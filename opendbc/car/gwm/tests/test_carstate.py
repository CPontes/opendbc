"""Engagement and signal decoding, from real CAN frames through the parsers the car uses.

Expected signs are the ones measured on the road, not read from the code: steering angle and torques positive to the left.
"""
from collections import defaultdict

import pytest

from opendbc.can.packer import CANPacker
from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.gwm.interface import CarInterface
from opendbc.car.gwm.values import CAR, DBC

ButtonType = structs.CarState.ButtonEvent.Type
GearShifter = structs.CarState.GearShifter
STALK = "STEER_AND_AP_STALK"
CAM_MSGS = ("ACC", "STEER_CMD", "ACC_CMD", "LATERAL_STATE")  # ACC_CMD is only forwarded
STOCK_MSGS = (*CAM_MSGS, STALK, "RX_STEER_RELATED")  # always on the bus of a running car
DRIVING = {"CRUISE_STATE_2": 3}
ACC_OFF = {"CRUISE_STATE_2": 1}
MOVING = {f"{pos}_WHEEL_SPEED": 50 for pos in ("FRONT_LEFT", "FRONT_RIGHT", "REAR_LEFT", "REAR_RIGHT")}
WHEELS = ("FRONT_LEFT_WHEEL_SPEED", "FRONT_RIGHT_WHEEL_SPEED", "REAR_LEFT_WHEEL_SPEED", "REAR_RIGHT_WHEEL_SPEED")


class Car:
  def __init__(self, safety_param=0):
    CP = CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6)
    CP.safetyConfigs[-1].safetyParam |= safety_param
    self.CI = CarInterface(CP)
    self.packer = CANPacker(DBC[CAR.GWM_HAVAL_H6][Bus.pt])
    self.signals = defaultdict(dict, {name: {} for name in STOCK_MSGS})
    self.nanos = 0
    self.frame = self.acc_frames = 0
    self.CI.update([])  # parsers learn the messages on the first read, a frame sent before that is lost

  def step(self, frames=1, **msgs):
    for name, values in msgs.items():
      self.signals[name].update(values)
    for i in range(frames):
      self.nanos += 10_000_000
      self.frame += 1
      # the camera sends the ACC at 10 Hz; a change goes out at once
      acc = self.frame % 10 == 0 or (i == 0 and "ACC" in msgs)
      self.acc_frames += acc
      packets = [self.packer.make_can_msg(name, 2 if name in CAM_MSGS else 0, values) for name, values in self.signals.items()
                 if name != "ACC" or acc]
      ret = self.CI.update([(self.nanos, packets)])
    return ret

  def pull(self):
    return self.step(**{STALK: {"AP_ENABLE_COMMAND": 1}})

  def release(self):
    return self.step(**{STALK: {"AP_ENABLE_COMMAND": 0}})

  def lever(self):
    self.pull()
    return self.release()

  def apply(self, lat_active=False, torque=0., cancel=False):
    CC = structs.CarControl()
    CC.enabled = CC.latActive = lat_active
    CC.actuators.torque = torque
    CC.cruiseControl.cancel = cancel
    return self.CI.apply(CC.as_reader(), self.nanos)[1]

  def sent(self, sends, name):
    return [frame for frame in sends if frame[0] == self.packer.dbc.name_to_msg[name].address]

  def field(self, name, frame, signal):
    sig = self.packer.dbc.name_to_msg[name].sigs[signal]
    raw = sig.get_raw_value(frame[1])
    raw -= (1 << sig.size) if sig.is_signed and raw & (1 << (sig.size - 1)) else 0
    return raw * sig.factor + sig.offset

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

  def test_cancel_wins_over_the_lever(self):
    car = Car(self.safety_param)
    car.lever()
    car.pull()
    car.step(**{STALK: {"AP_CANCEL_COMMAND": 1}})
    car.step(**{STALK: {"AP_CANCEL_COMMAND": 1, "AP_ENABLE_COMMAND": 0}})
    assert not car.cancel(0).cruiseState.enabled

  def test_cancel_during_the_pull_is_not_an_engage(self):
    # the panda dropped this pull, engaging here leaves every torque command blocked
    car = Car(self.safety_param)
    car.pull()
    car.cancel()
    car.cancel(0)
    assert not car.release().cruiseState.enabled

  def test_cancel_with_the_pull_is_not_an_engage(self):
    car = Car(self.safety_param)
    car.step(**{STALK: {"AP_CANCEL_COMMAND": 1, "AP_ENABLE_COMMAND": 1}})
    car.cancel(0)
    assert not car.release().cruiseState.enabled

  def test_pull_after_a_cancelled_pull_engages(self):
    car = Car(self.safety_param)
    car.pull()
    car.cancel()
    car.cancel(0)
    car.release()
    assert car.lever().cruiseState.enabled

  def test_pull_engages_again_after_openpilot_dropped_out(self):
    # 6 pulls went nowhere for 52 s in the logs: enabled never fell, so openpilot saw no new engage
    car = Car(self.safety_param)
    car.lever()
    car.apply(cancel=True)
    assert car.step().cruiseState.enabled
    assert not car.pull().cruiseState.enabled
    ret = car.release()
    assert ret.cruiseState.enabled and not resumes(ret)

  def test_pull_while_engaged_stays_engaged(self):
    car = Car(self.safety_param)
    car.lever()
    assert car.pull().cruiseState.enabled
    assert car.release().cruiseState.enabled

  def test_engages_again_after_cancel(self):
    car = Car(self.safety_param)
    car.lever()
    car.cancel()
    car.cancel(0)
    assert car.lever().cruiseState.enabled


class TestAccOff:
  """The ACC off (CRUISE_STATE_2 = 1) for 0.5 s ends the engagement, as in the panda: it dropped out, or never took the pull."""

  def test_acc_off_for_half_a_second_disengages(self):
    car = Car()
    car.step(ACC=DRIVING)
    assert car.lever().cruiseState.enabled
    car.signals["ACC"].update(ACC_OFF)
    start = car.acc_frames
    while car.step().cruiseState.enabled:
      assert car.acc_frames - start < 5
    assert car.acc_frames - start == 5
    assert not car.step(frames=10, ACC=DRIVING).cruiseState.enabled
    assert car.lever().cruiseState.enabled

  def test_a_pull_the_acc_never_takes(self):
    # counted from the pull, like the panda: holding the lever doesn't give the ACC more time
    car = Car()
    car.step(ACC=ACC_OFF)
    car.pull()
    car.step(frames=30)
    assert car.release().cruiseState.enabled
    assert not car.step(frames=21).cruiseState.enabled

  def test_acc_off_before_the_pull_does_not_count(self):
    car = Car()
    car.step(frames=100, ACC=ACC_OFF)
    assert car.lever().cruiseState.enabled
    assert car.step(frames=10, ACC=DRIVING).cruiseState.enabled
    assert car.step(frames=200).cruiseState.enabled

  @pytest.mark.parametrize("state", [0, 2, 3, 5])
  def test_other_acc_states_keep_it(self, state):
    car = Car()
    car.lever()
    assert car.step(frames=200, ACC={"CRUISE_STATE_2": state}).cruiseState.enabled


class TestBrake:
  def test_brake_during_the_pull_is_not_an_engage(self):
    # the foot leaving the brake while the lever is pulled: 2 of 114 pulls in the logs, panda refused both
    car = Car()
    car.brake()
    car.pull()
    car.brake(0)
    ret = car.release()
    assert not ret.brakePressed and not ret.cruiseState.enabled

  def test_brake_pressed_inside_the_pull_is_not_an_engage(self):
    car = Car()
    car.pull()
    car.brake()
    car.brake(0)
    assert not car.release().cruiseState.enabled

  def test_brake_one_frame_before_the_pull_is_not_an_engage(self):
    # the panda reads the stalk while the brake frame is still on its way
    car = Car()
    car.brake()
    car.step(BRAKE2={"PEDAL_BRAKE_PRESSED": 0}, **{STALK: {"AP_ENABLE_COMMAND": 1}})
    assert not car.release().cruiseState.enabled

  def test_pull_after_the_brake_engages(self):
    car = Car()
    car.brake()
    car.brake(0)
    car.step()
    assert car.lever().cruiseState.enabled

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
    assert car.step(CAR_OVERALL_SIGNALS2={"GAS_POSITION": 0.4}).gasPressed  # smallest step of the signal

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
    assert car.step().standstill
    ret = car.step(WHEEL_SPEEDS=dict.fromkeys(WHEELS, 36))
    assert not ret.standstill and ret.vEgoRaw == pytest.approx(10., abs=0.01)

  def test_acc_fault(self):
    car = Car()
    assert car.step(ACC={"CRUISE_STATE_2": 0}).accFaulted
    assert not car.step(ACC=DRIVING).accFaulted

  def test_set_speed_is_only_shown(self):
    ret = Car().step(ACC={"ACC_SPEED_SELECTION": 72})
    assert ret.cruiseState.speed == -1 and ret.cruiseState.speedCluster == pytest.approx(20., abs=0.01)


class TestFaults:
  def test_eps_fault_bit_needs_a_second(self):
    car = Car()
    assert not car.step(frames=100, RX_STEER_RELATED={"EPS_FAULT_PERMANENT": 1}).steerFaultTemporary
    ret = car.step()
    assert ret.steerFaultTemporary and not ret.steerFaultPermanent
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


class TestCommands:
  CAMERA_BUS = 2

  @staticmethod
  def driving(**stalk):
    car = Car()
    car.step(ACC={"CRUISE_STATE_2": 3}, WHEEL_SPEEDS=MOVING, **{STALK: stalk})
    return car

  def test_cancel_reaches_the_camera_ahead_of_the_stock_frame(self):
    car = self.driving(COUNTER=6, STEERING_ANGLE=12.5)
    frames = car.sent(car.apply(cancel=True), STALK)
    assert len(frames) == 1 and frames[0][2] == self.CAMERA_BUS
    assert car.field(STALK, frames[0], "AP_CANCEL_COMMAND") == 1
    assert car.field(STALK, frames[0], "COUNTER") == 7
    assert car.field(STALK, frames[0], "STEERING_ANGLE") == pytest.approx(12.5, abs=0.1)

  def test_cancel_counter_wraps_like_the_stock_one(self):
    # measured: the stock counter goes 13, 14, 0. 15 never shows up in 1.46 million frames
    car = self.driving(COUNTER=14)
    assert car.field(STALK, car.sent(car.apply(cancel=True), STALK)[0], "COUNTER") == 0

  def test_cancel_carries_no_other_button(self):
    buttons = ("AP_ENABLE_COMMAND", "AP_INCREASE_SPEED_COMMAND", "AP_DECREASE_SPEED_COMMAND",
               "AP_INCREASE_DISTANCE_COMMAND", "AP_REDUCE_DISTANCE_COMMAND")
    car = self.driving(**dict.fromkeys(buttons, 1))
    frame = car.sent(car.apply(cancel=True), STALK)[0]
    assert car.field(STALK, frame, "AP_CANCEL_COMMAND") == 1
    assert not any(car.field(STALK, frame, button) for button in buttons)

  def test_set_speed_reads_the_acc(self):
    car = Car()
    ret = car.step(ACC={"ACC_SPEED_SELECTION": 70})
    assert ret.cruiseState.speedCluster == pytest.approx(70 / 3.6)
    assert ret.cruiseState.speed == -1  # the planner still doesn't take it as a target

  def test_acc_driving_reads_the_state(self):
    car = Car()
    car.step(ACC={"CRUISE_STATE_2": 3})
    assert car.CI.CS.acc_driving
    car.step(ACC={"CRUISE_STATE_2": 2})
    assert not car.CI.CS.acc_driving

  def test_no_cancel_no_frame(self):
    car = Car()
    car.step()
    assert not car.sent(car.apply(), STALK)

  def test_rates(self):
    car = Car()
    car.step()
    sends = [car.apply() for _ in range(20)]
    assert [len(car.sent(s, "STEER_CMD")) for s in sends[:4]] == [1, 0, 1, 0]
    assert [len(car.sent(s, "RX_STEER_RELATED")) for s in sends[:4]] == [1, 0, 1, 0]
    assert [len(car.sent(s, "LATERAL_STATE")) for s in sends[:6]] == [1, 0, 0, 0, 0, 1]

  def test_torque_goes_out_with_its_sign(self):
    for torque, sign in ((0.1, 1), (-0.1, -1)):
      car = Car()
      car.step(WHEEL_SPEEDS=dict.fromkeys(WHEELS, 54))
      for _ in range(100):
        sends = car.apply(lat_active=True, torque=torque)
      steer = car.sent(sends, "STEER_CMD") or car.sent(car.apply(lat_active=True, torque=torque), "STEER_CMD")
      assert sign * car.field("STEER_CMD", steer[0], "TORQUE_CMD") > 0

  def test_one_unit_of_torque_is_never_sent(self):
    # repeated +-1 faults the EPS
    car = Car()
    car.step(WHEEL_SPEEDS=dict.fromkeys(WHEELS, 54))
    car.CI.CC.params.STEER_MAX = 100
    torque = car.field("STEER_CMD", car.sent(car.apply(lat_active=True, torque=0.01), "STEER_CMD")[0], "TORQUE_CMD")
    assert torque == 2

  def test_eps_torque_echoes_the_command_or_the_driver(self):
    car = Car()
    car.step(RX_STEER_RELATED={"B_RX_DRIVER_TORQUE": 40}, WHEEL_SPEEDS=dict.fromkeys(WHEELS, 54))
    frames = car.sent(car.apply(), "RX_STEER_RELATED")
    assert car.field("RX_STEER_RELATED", frames[0], "B_RX_DRIVER_TORQUE") == 40
    car.step()
    car.apply()
    for _ in range(40):
      sends = car.apply(lat_active=True, torque=-0.1)
    frames = car.sent(sends, "RX_STEER_RELATED") or car.sent(car.apply(lat_active=True, torque=-0.1), "RX_STEER_RELATED")
    assert car.field("RX_STEER_RELATED", frames[0], "B_RX_DRIVER_TORQUE") == 2 * car.CI.CC.apply_torque_last


class TestParams:
  @pytest.mark.parametrize("alpha_long", [False, True])
  def test_the_stock_acc_drives(self, alpha_long):
    CP = CarInterface.get_params(CAR.GWM_HAVAL_H6, gen_empty_fingerprint(), [], alpha_long, False, False)
    assert not CP.openpilotLongitudinalControl and not CP.alphaLongitudinalAvailable
    assert CP.safetyConfigs[-1].safetyParam == 0 and CP.radarUnavailable
