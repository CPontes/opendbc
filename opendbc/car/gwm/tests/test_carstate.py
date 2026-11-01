"""Engagement and signal decoding, from real CAN frames through the parsers the car uses.

Expected signs are the ones measured on the road, not read from the code: steering angle and torques positive to the left.
"""
from collections import defaultdict

import pytest

from opendbc.can.packer import CANPacker
from opendbc.car import Bus, structs
from opendbc.car.gwm.interface import CarInterface
from opendbc.car.gwm.values import CAR, DBC

ButtonType = structs.CarState.ButtonEvent.Type
GearShifter = structs.CarState.GearShifter
STALK = "STEER_AND_AP_STALK"
CAM_MSGS = ("ACC",)
ACTIVE = {"CRUISE_STATE_2": 1}


class Car:
  def __init__(self, safety_param=0, long=False):
    CP = CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6)
    CP.safetyConfigs[-1].safetyParam |= safety_param
    CP.openpilotLongitudinalControl = long
    self.CI = CarInterface(CP)
    self.packer = CANPacker(DBC[CAR.GWM_HAVAL_H6][Bus.pt])
    self.signals = defaultdict(dict)
    self.nanos = 0
    self.CI.update([])  # parsers learn the messages on the first read, a frame sent before that is lost

  def step(self, frames=1, **msgs):
    for name, values in msgs.items():
      self.signals[name].update(values)
    for _ in range(frames):
      self.nanos += 10_000_000
      packets = [self.packer.make_can_msg(name, 2 if name in CAM_MSGS else 0, values) for name, values in self.signals.items()]
      ret = self.CI.update([(self.nanos, packets)])
    return ret

  def pull(self):
    return self.step(**{STALK: {"AP_ENABLE_COMMAND": 1}})

  def release(self):
    return self.step(**{STALK: {"AP_ENABLE_COMMAND": 0}})

  def lever(self):
    self.pull()
    return self.release()

  def cancel(self, value=1):
    return self.step(**{STALK: {"AP_CANCEL_COMMAND": value}})

  def brake(self, value=1):
    return self.step(BRAKE2={"PEDAL_BRAKE_PRESSED": value})


def resumes(ret):
  return [be for be in ret.buttonEvents if be.type == ButtonType.resumeCruise]


class TestEngagement:
  safety_param = 0

  def test_starts_off(self):
    ret = Car(self.safety_param).step()
    assert not ret.cruiseState.enabled and not ret.cruiseState.available

  def test_lever_engages_on_release_only(self):
    car = Car(self.safety_param)
    assert not car.pull().cruiseState.enabled
    assert not car.step(frames=20).cruiseState.enabled
    ret = car.release()
    assert ret.cruiseState.enabled and ret.cruiseState.available

  def test_engaging_is_not_a_resume(self):
    assert not resumes(Car(self.safety_param).lever())

  def test_stays_engaged(self):
    car = Car(self.safety_param)
    car.lever()
    assert car.step(frames=200).cruiseState.enabled

  def test_cancel_disengages(self):
    car = Car(self.safety_param)
    car.lever()
    ret = car.cancel()
    assert not ret.cruiseState.enabled and not ret.cruiseState.available
    assert not car.cancel(0).cruiseState.enabled

  def test_cancel_keeps_it_off_while_held(self):
    car = Car(self.safety_param)
    car.cancel()
    car.pull()
    # the release itself turns it on for this frame, the next one with cancel held turns it off
    car.release()
    assert not car.step().cruiseState.enabled

  @pytest.mark.xfail(strict=True, reason="STATE-01: lever released on the same frame cancel ends keeps it engaged")
  def test_cancel_wins_over_the_lever(self):
    car = Car(self.safety_param)
    car.lever()
    car.pull()
    car.step(**{STALK: {"AP_CANCEL_COMMAND": 1}})
    car.step(**{STALK: {"AP_CANCEL_COMMAND": 1, "AP_ENABLE_COMMAND": 0}})
    assert not car.cancel(0).cruiseState.enabled

  def test_engages_again_after_cancel(self):
    car = Car(self.safety_param)
    car.lever()
    car.cancel()
    car.cancel(0)
    assert car.lever().cruiseState.enabled


class TestBrake:
  def test_brake_disengages(self):
    car = Car()
    car.lever()
    ret = car.brake()
    assert ret.brakePressed and not ret.cruiseState.enabled
    assert not car.brake(0).cruiseState.enabled

  def test_lever_while_engaged_is_no_event(self):
    car = Car()
    car.lever()
    ret = car.lever()
    assert ret.cruiseState.enabled and not resumes(ret)

  def test_lever_engages_again_after_brake(self):
    car = Car()
    car.lever()
    car.brake()
    car.brake(0)
    ret = car.lever()
    assert ret.cruiseState.enabled and not resumes(ret)


class TestSignals:
  @pytest.mark.parametrize("mode,gear,parking", [(0, GearShifter.park, True), (1, GearShifter.drive, False),
                                                 (2, GearShifter.neutral, False), (3, GearShifter.reverse, False)])
  def test_gear(self, mode, gear, parking):
    ret = Car().step(CAR_OVERALL_SIGNALS={"DRIVE_MODE": mode})
    assert ret.gearShifter == gear and ret.parkingBrake == parking

  def test_gas(self):
    car = Car()
    assert not car.step(CAR_OVERALL_SIGNALS2={"GAS_POSITION": 0}).gasPressed
    assert car.step(CAR_OVERALL_SIGNALS2={"GAS_POSITION": 1}).gasPressed

  @pytest.mark.parametrize("direction,sign", [(0, 1), (1, -1)])
  def test_steering_angle_is_positive_to_the_left(self, direction, sign):
    ret = Car().step(**{STALK: {"STEERING_ANGLE": 12.5, "STEERING_DIRECTION": direction}})
    assert ret.steeringAngleDeg == pytest.approx(sign * 12.5, abs=0.1)

  @pytest.mark.parametrize("direction,sign", [(0, 1), (1, -1)])
  def test_steering_rate(self, direction, sign):
    ret = Car().step(**{STALK: {"STEERING_RATE": 12, "RATE_DIRECTION": direction}})
    assert ret.steeringRateDeg == pytest.approx(sign * 12, abs=0.01)

  @pytest.mark.parametrize("torque,pressed", [(0, False), (50, False), (-50, False), (51, True), (-51, True)])
  def test_driver_torque(self, torque, pressed):
    ret = Car().step(RX_STEER_RELATED={"B_RX_DRIVER_TORQUE": torque})
    assert ret.steeringTorque == torque and ret.steeringPressed == pressed

  def test_eps_torque(self):
    ret = Car().step(RX_STEER_RELATED={"B_RX_DRIVER_TORQUE": 10, "B_RX_EPS_TORQUE": -120})
    assert ret.steeringTorqueEps == -120 and ret.steeringTorque == 10

  @pytest.mark.parametrize("door", ["DOOR_REAR_RIGHT_OPEN", "DOOR_FRONT_RIGHT_OPEN", "DOOR_REAR_LEFT_OPEN", "DOOR_DRIVER_OPEN"])
  def test_doors(self, door):
    car = Car()
    assert not car.step().doorOpen
    assert car.step(DOOR_DRIVER={door: 1}).doorOpen

  def test_seatbelt(self):
    car = Car()
    assert not car.step(SEATBELT={"SEAT_BELT_DRIVER_STATE": 0}).seatbeltUnlatched
    assert car.step(SEATBELT={"SEAT_BELT_DRIVER_STATE": 1}).seatbeltUnlatched

  @pytest.mark.parametrize("signal,attr,other", [("LEFT_TURN_SIGNAL", "leftBlinker", "rightBlinker"), ("RIGHT_TURN_SIGNAL", "rightBlinker", "leftBlinker")])
  def test_blinker_outlasts_the_lamp(self, signal, attr, other):
    car = Car()
    ret = car.step(LIGHTS={signal: 1})
    assert getattr(ret, attr) and not getattr(ret, other)
    assert getattr(car.step(frames=48, LIGHTS={signal: 0}), attr)
    assert not getattr(car.step(frames=2), attr)

  @pytest.mark.parametrize("signal,attr,other", [("BSM_LEFT", "leftBlindspot", "rightBlindspot"), ("BSM_RIGHT", "rightBlindspot", "leftBlindspot")])
  def test_blind_spot(self, signal, attr, other):
    ret = Car().step(RADAR_BEHIND={signal: 1})
    assert getattr(ret, attr) and not getattr(ret, other)

  def test_wheel_speed(self):
    car = Car()
    speeds = {"FRONT_LEFT_WHEEL_SPEED": 36, "FRONT_RIGHT_WHEEL_SPEED": 36, "REAR_LEFT_WHEEL_SPEED": 36, "REAR_RIGHT_WHEEL_SPEED": 36}
    assert car.step().standstill
    ret = car.step(WHEEL_SPEEDS=speeds)
    assert not ret.standstill and ret.vEgoRaw == pytest.approx(10., abs=0.01)

  def test_acc_fault(self):
    car = Car()
    assert car.step(ACC={"CRUISE_STATE_2": 0}).accFaulted
    assert not car.step(ACC=ACTIVE).accFaulted

  def test_set_speed_only_with_openpilot_longitudinal(self):
    assert Car().step(ACC={"ACC_SPEED_SELECTION": 72}).cruiseState.speed == -1
    assert Car(long=True).step(ACC={"ACC_SPEED_SELECTION": 72}).cruiseState.speed == pytest.approx(20., abs=0.01)


class TestFaults:
  def test_eps_fault_bit_needs_a_second(self):
    car = Car()
    assert not car.step(frames=100, RX_STEER_RELATED={"EPS_FAULT_PERMANENT": 1}).steerFaultTemporary
    assert car.step().steerFaultTemporary
    assert not car.step(RX_STEER_RELATED={"EPS_FAULT_PERMANENT": 0}).steerFaultTemporary

  def test_eps_fault_bit_count_restarts(self):
    car = Car()
    car.step(frames=100, RX_STEER_RELATED={"EPS_FAULT_PERMANENT": 1})
    car.step(RX_STEER_RELATED={"EPS_FAULT_PERMANENT": 0})
    assert not car.step(frames=100, RX_STEER_RELATED={"EPS_FAULT_PERMANENT": 1}).steerFaultTemporary

  def test_eps_ignoring_the_request_needs_a_second(self):
    car = Car()
    car.CI.CC.lat_active = True
    assert not car.step(frames=100, RX_STEER_RELATED={"A_RX_STEER_REQUESTED": 0}).steerFaultTemporary
    assert car.step().steerFaultTemporary
    # interface reads the signal before the parsers update: one frame late
    assert not car.step(frames=2, RX_STEER_RELATED={"A_RX_STEER_REQUESTED": 1}).steerFaultTemporary

  def test_eps_obeying_is_no_fault(self):
    car = Car()
    car.CI.CC.lat_active = True
    assert not car.step(frames=150, RX_STEER_RELATED={"A_RX_STEER_REQUESTED": 1}).steerFaultTemporary

  def test_no_fault_when_openpilot_is_not_asking(self):
    car = Car()
    car.CI.CC.lat_active = False
    assert not car.step(frames=150, RX_STEER_RELATED={"A_RX_STEER_REQUESTED": 0}).steerFaultTemporary

  def test_count_restarts_when_the_request_stops(self):
    car = Car()
    car.CI.CC.lat_active = True
    car.step(frames=100, RX_STEER_RELATED={"A_RX_STEER_REQUESTED": 0})
    car.CI.CC.lat_active = False
    car.step()
    car.CI.CC.lat_active = True
    assert not car.step(frames=100).steerFaultTemporary


class TestFollowDistance:
  @pytest.mark.parametrize("distance,personality", [(1, 1), (2, 2), (3, 3), (4, 3)])
  def test_no_press_when_they_agree(self, distance, personality):
    car = Car()
    car.CI.current_personality = personality
    before = car.CI.press_gac_button
    car.step(frames=5, ACC={"CAR_DISTANCE_SELECTION": distance})
    assert car.CI.press_gac_button == before

  @pytest.mark.parametrize("distance,personality", [(1, 2), (2, 1), (3, 2), (4, 1)])
  def test_presses_until_they_agree(self, distance, personality):
    car = Car()
    car.CI.current_personality = personality
    states = set()
    for _ in range(4):
      car.step(ACC={"CAR_DISTANCE_SELECTION": distance})
      states.add(car.CI.press_gac_button)
    assert states == {True, False}

  @pytest.mark.xfail(strict=True, reason="STATE-01: a button release is reported on every frame the press state is False")
  def test_no_button_event_without_a_press(self):
    car = Car()
    car.CI.current_personality = 3
    assert not car.step(frames=5, ACC={"CAR_DISTANCE_SELECTION": 3}).buttonEvents
