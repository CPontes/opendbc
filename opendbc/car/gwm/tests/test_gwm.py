import math
from types import SimpleNamespace

import numpy as np
import pytest

from opendbc.can.dbc import DBC
from opendbc.car import structs
from opendbc.car.gwm import gwmcan
from opendbc.car.gwm.carstate import CarState
from opendbc.car.gwm.interface import CarInterface
from opendbc.car.gwm.values import CAR, DBC as GWM_DBC, GwmFlags
from opendbc.car import Bus


def zeros(dbc, msg):
  return bytes(dbc.name_to_msg[msg].size)


class TestGwmLeadDistance:
  def setup_method(self):
    self.CP = CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6)
    self.cp = CarState.get_can_parsers(self.CP)[Bus.cam]

  def frame(self, raw):
    dat = bytearray(64)
    dat[33], dat[34] = 0xF0 | (raw >> 8), raw & 0xFF  # the upper half of the byte is something else
    return (0x2B8, bytes(dat), 2)

  def test_the_distance_of_the_haval(self):
    self.cp.update([(1000, [self.frame(1000)])])
    assert self.cp.vl["UNDEFINED_INTERESTING_SIGNALS"]["LEAD_DISTANCE"] == pytest.approx(1000 * .0534 + 3.7)

  def test_none_reads_the_zero_of_the_field(self):
    self.cp.update([(1000, [self.frame(0)])])
    assert self.cp.vl["UNDEFINED_INTERESTING_SIGNALS"]["LEAD_DISTANCE"] == pytest.approx(3.7)

  def test_losing_it_is_no_fault(self):
    # it only helps: a car or a moment without it must not take the ACC down
    assert self.cp.message_states[0x2B8].ignore_alive


class TestGwmCarController:
  def setup_method(self):
    CP = CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6)
    self.cc = CarInterface(CP).CC
    dbc = DBC(GWM_DBC[CP.carFingerprint][Bus.pt])
    self.CS = SimpleNamespace(
      out=structs.CarState(),
      camera_stock_data=zeros(dbc, "STEER_CMD"),
      eps_stock_data=zeros(dbc, "RX_STEER_RELATED"),
      steer_and_ap_stalk_msg=zeros(dbc, "STEER_AND_AP_STALK"),
      hud_stock_data=zeros(dbc, "LATERAL_STATE"),
      longitudinal_stock_data=zeros(dbc, "ACC_CMD"),
    )

  def _steer(self, torque, driver_torque, frames=40, v_ego=15.):
    self.CS.out.vEgo = v_ego
    CC = structs.CarControl()
    CC.latActive = True
    CC.enabled = True
    CC.actuators.torque = torque
    CC = CC.as_reader()
    self.CS.out.steeringTorque = driver_torque
    applied = 0
    for _ in range(frames):
      self.CS.out.steeringTorqueEps = applied
      actuators, _ = self.cc.update(CC, self.CS, 0)
      applied = actuators.torqueOutputCan
    return applied

  def test_keeps_torque_when_driver_helps(self):
    # driver pushing the same way as openpilot (torque limited in a tight curve)
    assert self._steer(-1.0, -150) < 0
    assert self._steer(1.0, 150) > 0

  def test_lets_go_when_driver_opposes(self):
    assert self._steer(-1.0, 150) == 0
    assert self._steer(1.0, -150) == 0

  def test_no_request_when_driver_opposes(self):
    self._steer(1.0, -150)
    assert not self.cc.lat_active
    self._steer(1.0, 150)
    assert self.cc.lat_active

  def test_light_touch_keeps_torque(self):
    assert self._steer(1.0, -60) > 0

  def test_full_torque_at_any_speed(self):
    # slow corners need all of it: low speed wobble is filtered in the request, not here
    assert abs(self._steer(1.0, 0, frames=200, v_ego=10 / 3.6)) == 253
    assert abs(self._steer(1.0, 0, frames=200, v_ego=40 / 3.6)) == 253


class TestGwmSoftHandoff(TestGwmCarController):
  # a light push against openpilot fades its torque out at the panda's rate; a firm one still lets go at once
  def setup_method(self):
    super().setup_method()
    CP = CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6)
    CP.flags |= GwmFlags.SOFT_HANDOFF.value
    self.cc = CarInterface(CP).CC

  def _run(self, steps):
    """steps: (frames, driver torque); returns what went out at each 50 Hz frame and whether it was requested"""
    CC = structs.CarControl()
    CC.latActive = CC.enabled = True
    CC.actuators.torque = 1.0
    CC = CC.as_reader()
    self.CS.out.vEgo = 15.
    out, applied = [], 0
    for frames, driver in steps:
      self.CS.out.steeringTorque = driver
      for _ in range(frames):
        self.CS.out.steeringTorqueEps = applied
        actuators, _ = self.cc.update(CC, self.CS, 0)
        if self.cc.frame % 2 == 1:  # the frame that just ran sent the steering
          applied = actuators.torqueOutputCan
          out.append((applied, self.cc.lat_active))
    return out

  def test_a_light_push_fades_at_the_panda_rate(self):
    out = self._run([(200, 0), (20, -120)])
    assert out[-11][0] == 253
    fade = [o[0] for o in out[-10:]]
    assert all(0 <= a - b <= 6 for a, b in zip([253] + fade, fade, strict=False))
    assert fade[-1] == 253 - 6 * 10
    assert all(o[1] for o in out[-10:])  # still requested while there is torque

  def test_held_it_reaches_zero_and_lets_go(self):
    out = self._run([(200, 0), (100, -120)])  # 1 s
    assert out[-1] == (0, False)
    assert max(i for i, o in enumerate(out) if o[0] > 0) - out.index((253, True)) < 200 + 43

  def test_a_short_push_keeps_most_and_comes_back_from_there(self):
    out = self._run([(200, 0), (30, -120), (1, 0)])
    assert out[-1][0] >= 253 - 6 * 15 and out[-1][1]
    out = self._run([(60, 0)])
    assert out[-1][0] == 253  # back to the top in 0.6 s: from zero it takes 1.3 s

  def test_just_under_firm_still_fades(self):
    out = self._run([(200, 0), (2, -149)])
    assert out[-1] == (253 - 6, True)

  def test_a_push_around_1_nm_never_cuts(self):
    out = self._run([(200, 0)] + [(2, -101), (2, -99)] * 10)
    assert all(o[1] for o in out[-20:]) and min(o[0] for o in out[-20:]) >= 253 - 6 * 10

  def test_the_end_of_the_fade_is_zero_never_one(self):
    # the last step can land on 1: the EPS faults on a repeated 1, and 1 is no torque anyway
    self.cc.apply_torque_last = 7
    out = self._run([(2, -120)])
    assert out[-1] == (0, False)  # 7 - 6 = 1 goes out as 0, and the request with it

  def test_a_firm_push_lets_go_at_once(self):
    assert self._run([(200, 0), (2, -150)])[-1] == (0, False)

  def test_a_light_push_turning_firm_lets_go_at_once(self):
    out = self._run([(200, 0), (10, -120), (2, -160)])
    assert out[-2][0] > 0 and out[-1] == (0, False)

  def test_lets_go_when_driver_opposes(self):
    # with the soft handoff a firm push still lets go
    assert self._steer(-1.0, 150) == 0
    assert self._steer(1.0, -150) == 0

  def test_no_request_when_driver_opposes(self):
    self._steer(1.0, -150)
    assert not self.cc.lat_active
    self._steer(1.0, 150)
    assert self.cc.lat_active

  def test_without_openpilot_steering_nothing_fades(self):
    CC = structs.CarControl()
    CC.enabled = True
    CC.actuators.torque = 1.0
    self._run([(200, 0)])
    self.CS.out.steeringTorque = -120
    actuators, _ = self.cc.update(CC.as_reader(), self.CS, 0)
    actuators, _ = self.cc.update(CC.as_reader(), self.CS, 0)
    assert actuators.torqueOutputCan == 0 and not self.cc.lat_active


class TestGwmSoftHandoffOff(TestGwmCarController):
  def test_off_by_default_a_light_push_cuts(self):
    self._steer(1.0, 0, frames=200)
    assert self._steer(1.0, -120, frames=2) == 0


class TestGwmSteerFault:
  def _fault(self, lat_active, frames=150):
    CI = CarInterface(CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6))
    CI.CC.lat_active = lat_active
    # no CAN data: the EPS does not confirm the steering request
    for _ in range(frames):
      CS = CI.update([])
    return CS.steerFaultTemporary

  def test_fault_when_eps_ignores_the_request(self):
    assert self._fault(True)

  def test_no_fault_without_a_request(self):
    assert not self._fault(False)


class TestGwmResume:
  def setup_method(self):
    CP = CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6)
    self.cc = CarInterface(CP).CC
    dbc = DBC(GWM_DBC[CP.carFingerprint][Bus.pt])
    self.CS = SimpleNamespace(out=structs.CarState(), camera_stock_data=None, eps_stock_data=None, hud_stock_data=None, longitudinal_stock_data=None,
                              steer_and_ap_stalk_msg=zeros(dbc, "STEER_AND_AP_STALK"), acc_waiting=True, acc_driving=True, stalk_idle=True)

  def _stalk_frames(self, frames, resume=True, cancel=False):
    CC = structs.CarControl()
    CC.enabled = True
    CC.cruiseControl.resume = resume
    CC.cruiseControl.cancel = cancel
    CC = CC.as_reader()
    enable = other = 0
    for _ in range(frames):
      for addr, dat, _bus in self.cc.update(CC, self.CS, 0)[1]:
        if addr == 0xA1:
          enable += bool(dat[5] & 0x80)
          other += bool(dat[5] & 0x40)
    return enable, other

  def test_one_pull_per_request(self):
    assert self._stalk_frames(50) == (13, 0)

  def test_each_request_is_one_pull(self):
    self._stalk_frames(50)
    self._stalk_frames(1, resume=False)
    assert self._stalk_frames(50) == (13, 0)

  def test_only_with_the_acc_waiting(self):
    self.CS.acc_waiting = False
    assert self._stalk_frames(20) == (0, 0)
    self.CS.acc_waiting = True
    assert self._stalk_frames(20) == (0, 0)  # the request was missed, it isn't taken late

  def test_a_driver_button_ends_the_pull(self):
    assert self._stalk_frames(5) == (5, 0)
    self.CS.stalk_idle = False
    assert self._stalk_frames(20) == (0, 0)
    self.CS.stalk_idle = True
    assert self._stalk_frames(20) == (0, 0)

  def test_the_end_of_the_request_ends_the_pull(self):
    assert self._stalk_frames(5) == (5, 0)
    assert self._stalk_frames(20, resume=False) == (0, 0)

  def test_cancel_wins(self):
    assert self._stalk_frames(5) == (5, 0)
    assert self._stalk_frames(5, cancel=True) == (0, 5)

  def test_no_original_frame_no_pull(self):
    self.CS.steer_and_ap_stalk_msg = None
    assert self._stalk_frames(5) == (0, 0)

  def test_cancel_is_a_push_like_the_drivers(self):
    assert self._stalk_frames(13 + 100, resume=False, cancel=True) == (0, 13)

  def test_cancel_pushes_again_if_the_acc_stayed_on(self):
    assert self._stalk_frames(2 * 113, resume=False, cancel=True) == (0, 26)

  def test_cancel_stops_when_the_acc_lets_go(self):
    assert self._stalk_frames(5, resume=False, cancel=True) == (0, 5)
    self.CS.acc_driving = False
    assert self._stalk_frames(200, resume=False, cancel=True) == (0, 0)

  def test_cancel_only_with_the_acc_driving(self):
    self.CS.acc_driving = False
    assert self._stalk_frames(50, resume=False, cancel=True) == (0, 0)

  def test_cancel_never_lets_a_stopped_car_go(self):
    self.CS.out.standstill = True
    assert self._stalk_frames(50, resume=False, cancel=True) == (0, 0)

  def test_a_new_request_is_a_new_push(self):
    assert self._stalk_frames(20, resume=False, cancel=True) == (0, 13)
    self._stalk_frames(1, resume=False)
    assert self._stalk_frames(20, resume=False, cancel=True) == (0, 13)

  def _minus(self, frames, decrease=True, cancel=False):
    CC = structs.CarControl()
    CC.enabled = True
    CC.cruiseControl.decrease = decrease
    CC.cruiseControl.cancel = cancel
    CC = CC.as_reader()
    presses = 0
    for _ in range(frames):
      for addr, dat, _bus in self.cc.update(CC, self.CS, 0)[1]:
        if addr == 0xA1:
          presses += bool(dat[6] & 0x04)
    return presses

  def test_minus_is_a_press_like_the_drivers_then_a_gap(self):
    assert self._minus(13 + 35) == 13
    assert self._minus(13 + 35) == 13

  def test_minus_only_with_the_acc_driving_a_moving_car(self):
    self.CS.acc_driving = False
    assert self._minus(50) == 0
    self.CS.acc_driving = True
    self.CS.out.standstill = True
    assert self._minus(50) == 0

  def test_minus_stops_with_the_end_of_the_request(self):
    assert self._minus(5) == 5
    assert self._minus(50, decrease=False) == 0
    assert self._minus(5) == 5  # a new press from the start, not the rest of the old one

  def test_a_driver_button_stops_the_minus(self):
    assert self._minus(5) == 5
    self.CS.stalk_idle = False
    assert self._minus(50) == 0

  def test_cancel_wins_over_the_minus(self):
    assert self._minus(50, cancel=True) == 0

  def test_no_original_frame_no_minus(self):
    self.CS.steer_and_ap_stalk_msg = None
    assert self._minus(5) == 0

  def test_no_original_frame_no_cancel(self):
    self.CS.steer_and_ap_stalk_msg = None
    assert self._stalk_frames(5, resume=False, cancel=True) == (0, 0)


class TestGwmLongitudinal:
  def setup_method(self):
    CP = CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6)
    CP.openpilotLongitudinalControl = True
    self.cc = CarInterface(CP).CC
    dbc = DBC(GWM_DBC[CP.carFingerprint][Bus.pt])
    self.CS = SimpleNamespace(out=structs.CarState(), camera_stock_data=None, eps_stock_data=None, hud_stock_data=None,
                              longitudinal_stock_data=zeros(dbc, "ACC_CMD"), steer_and_ap_stalk_msg=zeros(dbc, "STEER_AND_AP_STALK"),
                              acc_waiting=False, acc_driving=False, stalk_idle=True, rolling_back=False)
    self.CS.out.vEgo = 11.

  def send(self, accel, pitch=0., state=None, active=True):
    frame = self.frame(accel, pitch, state, active)
    return frame[9] & 0x1F, 181 - frame[13], ((frame[27] & 0x1F) << 8 | frame[28]) - 192

  def frame(self, accel, pitch=0., state=None, active=True):
    CC = structs.CarControl()
    CC.enabled = active
    CC.longActive = active
    CC.actuators.accel = accel
    CC.actuators.longControlState = state if state is not None else structs.CarControl.Actuators.LongControlState.pid
    CC.orientationNED = [0., pitch, 0.]
    frame = None
    for _ in range(2):
      for addr, dat, _bus in self.cc.update(CC.as_reader(), self.CS, 0)[1]:
        if addr == 0x143:
          frame = dat
    return frame

  def settle(self, accel, pitch=0.):
    for _ in range(300):
      out = self.send(accel, pitch)
    return out

  def stock(self, mode):
    frame = bytearray(self.CS.longitudinal_stock_data)
    frame[9] = mode
    self.CS.longitudinal_stock_data = bytes(frame)

  def pedal(self, pressed=True):
    self.CS.out.brakePressed = pressed

  def braking_with_the_acc(self, accel=-2.):
    self.CS.acc_driving = True
    self.stock(12)
    self.settle(accel)

  def test_the_pedal_takes_over_from_our_brake(self):
    self.braking_with_the_acc()
    held = self.send(-2.)[1]
    self.pedal()
    assert self.send(0., active=False)[:2] == (13, held)
    assert self.send(0., active=False)[:2] == (13, held)

  def test_the_brake_ends_when_the_car_drops_the_acc(self):
    self.braking_with_the_acc()
    self.pedal()
    self.send(0., active=False)
    self.CS.acc_driving = False
    self.stock(4)
    assert self.send(0., active=False) == (4, 0, 0)

  def test_the_brake_stays_less_than_the_panda_lets_it(self):
    self.braking_with_the_acc()
    self.pedal()
    modes = [self.send(0., active=False)[0] for _ in range(20)]
    assert modes.count(13) == 9  # 50 Hz: 0.18 s, the panda stops at 0.2
    assert 12 not in modes

  def test_out_of_the_brake_there_is_no_gas_frame(self):
    self.braking_with_the_acc()
    self.pedal()
    self.CS.acc_driving = False
    assert self.send(0., active=False) == (4, 0, 0)

  def test_the_brake_is_not_held_with_the_gas_pedal(self):
    self.braking_with_the_acc()
    self.pedal()
    self.CS.out.gasPressed = True
    assert self.send(0., active=False)[1] == 0

  def test_letting_go_of_the_pedal_ends_it(self):
    self.braking_with_the_acc()
    self.pedal()
    self.send(0., active=False)
    self.pedal(False)
    assert self.send(0., active=False)[1] == 0

  def test_the_pedal_out_of_the_gas_holds_nothing(self):
    self.CS.acc_driving = True
    self.stock(12)
    self.settle(.5)
    self.pedal()
    assert self.send(0., active=False) == (4, 0, 0)

  def test_a_brake_lost_without_the_pedal_holds_nothing(self):
    self.braking_with_the_acc()
    assert self.send(0., active=False)[1] == 0

  def test_the_mode_of_the_acc_is_kept_out_of_the_pedal(self):
    self.stock(8)
    assert self.send(0., active=False)[0] == 8
    self.pedal()
    assert self.send(0., active=False)[0] == 8

  def test_a_deepening_brake_holds_what_the_panda_saw(self):
    # the panda takes the level at the pedal, frames before openpilot lets go: asking for more would be refused, and no 0x143 at all
    self.braking_with_the_acc(-1.)
    seen = self.send(-1.)[1]
    self.send(-1.5)
    self.pedal()
    self.send(-2.)
    assert self.send(0., active=False)[:2] == (13, seen)

  def test_an_easing_brake_holds_the_last_of_it(self):
    self.braking_with_the_acc(-2.)
    self.send(-1.8)
    self.send(-1.6)
    last = self.send(-1.4)[1]
    self.pedal()
    assert self.send(0., active=False)[:2] == (13, last)

  def test_the_pedal_with_openpilot_still_on_holds_nothing(self):
    self.braking_with_the_acc(-1.)
    self.pedal()
    assert self.send(.5)[0] == 12

  def test_openpilot_back_on_ends_it(self):
    self.braking_with_the_acc()
    self.pedal()
    self.send(0., active=False)
    assert self.send(.5)[0] == 12

  def test_the_brake_that_holds_a_descent_stays_too(self):
    # the field at its zero is still the car holding the slope: 82 to 87% of the brake frames on descents
    self.braking_with_the_acc(-.5)
    for _ in range(80):
      self.send(-.01)
    self.pedal()
    assert self.send(0., active=False)[:2] == (13, 41)

  def test_a_frame_of_gas_among_the_last_holds_nothing(self):
    # the panda took its level from the frame of gas: every frame of a try would be refused
    self.braking_with_the_acc(-.5)
    self.send(.5)
    self.send(-.5)
    self.pedal()
    assert self.send(0., active=False) == (4, 0, 0)

  def test_the_camera_on_the_brake_also_goes_to_the_pedal(self):
    self.stock(13)
    self.pedal()
    assert self.send(0., active=False) == (4, 0, 0)

  def test_the_pedal_mode_goes_with_its_own_fields(self):
    # in 21 422 frames the camera's mode 4 is always 0, 3, 0, and these fields decide how the car brakes
    self.stock(12)
    frame = bytearray(self.CS.longitudinal_stock_data)
    frame[27] = (4 << 5)
    frame[29] = 0x80
    self.CS.longitudinal_stock_data = bytes(frame)
    self.pedal()
    out = self.frame(0., active=False)
    assert (out[9] & 0x1F, (out[12] >> 5) & 1, out[27] >> 5, out[29] >> 7) == (4, 0, 3, 0)

  def test_the_output_tells_the_brake_of_the_handoff(self):
    self.braking_with_the_acc()
    self.pedal()
    CC = structs.CarControl()
    new_actuators = self.cc.update(CC.as_reader(), self.CS, 0)[0]
    new_actuators = self.cc.update(CC.as_reader(), self.CS, 0)[0]
    assert new_actuators.accel == -2.

  def test_on_a_descent_the_brake_asks_only_for_the_plan(self):
    # in the brake mode the car holds the deceleration by itself on any slope: real = 0.95 x asked - 0.03 x slope, 17 542 frames.
    # Adding the slope again braked at -1.3 with the plan asking for +0.5
    descent = -0.13  # rad, 7.4 degrees
    for _ in range(300):
      self.send(-1., descent)
    assert self.send(-1., descent)[1] == round(41 + 1. / 3.5 * 66)

  def test_a_descent_still_chooses_the_brake(self):
    descent = -0.13
    for _ in range(300):
      out = self.send(.5, descent)
    assert out[0] == 13 and out[1] == 41  # holds the speed, doesn't add braking

  def test_no_dead_zone_in_the_gas(self):
    assert self.send(0.3)[2] > 300
    assert self.send(0.5)[2] == 725
    assert self.send(1.)[2] == 1440

  def test_gas_and_brake_stay_inside_what_the_panda_allows(self):
    for accel in np.arange(-5., 5., .1):
      _, brake, gas = self.send(float(accel))
      assert brake <= 107 and -192 <= gas <= 4577

  def test_more_acceleration_never_means_less_gas(self):
    assert list(gwmcan.ACC_GAS) == sorted(gwmcan.ACC_GAS) and len(gwmcan.ACC_GAS) == len(gwmcan.ACC_GAS_ACCEL)

  def test_a_request_around_zero_does_not_flip_gas_and_brake(self):
    modes = [self.send(a)[0] for a in [0.05, -0.05, -0.2, 0.05, -0.1, -0.2, 0.] * 10]
    assert set(modes) == {12}
    self.settle(-0.3)
    modes = [self.send(a)[0] for a in [-0.2, -0.3, -0.15, -0.26, -0.11] * 10]
    assert set(modes) == {13}

  def test_the_brake_lets_go_once_gas_is_asked(self):
    assert self.send(-0.5)[0] == 13
    assert self.send(0.2)[0] == 12

  def test_an_eased_brake_stays_on_the_brake_for_a_while(self):
    self.settle(-0.5)
    modes = [self.send(-0.05)[0] for _ in range(95)]  # 1.9 s
    assert set(modes) == {13}
    assert [self.send(-0.05)[0] for _ in range(10)][-1] == 12

  def test_leaving_a_stop_never_waits_on_the_brake(self):
    self.CS.out.vEgo = 1.
    self.settle(-0.5)
    assert self.send(0.05)[0] == 12

  def test_rolling_slowly_stays_on_the_brake_until_the_request_turns(self):
    self.CS.out.vEgo = 1.
    self.settle(-0.5)
    assert {self.send(-0.05)[0] for _ in range(300)} == {13}
    assert self.send(0.)[0] == 12

  def test_leaving_a_stop_on_a_descent_does_not_wait_for_the_slope(self):
    self.CS.out.vEgo = 1.
    descent = math.atan(-.03)
    self.settle(-0.5, descent)
    assert self.send(0.25, descent)[0] == 12

  def test_out_of_the_brake_the_gas_climbs_from_coasting(self):
    self.settle(-0.5)
    mode, _, gas = self.send(0.2)
    assert mode == 12 and gas < -100
    assert [self.send(0.2)[2] for _ in range(50)][-1] == 330  # 1 s later

  def test_a_real_request_for_gas_is_not_held_back(self):
    self.settle(-0.5)
    assert self.send(1.)[2] == 1440

  def test_leaving_a_stop_has_no_climb(self):
    self.CS.out.vEgo = 1.
    self.settle(-0.5)
    assert self.send(0.2)[2] == 330

  def test_past_the_climb_gas_moves_freely(self):
    self.settle(-0.5)
    for _ in range(100):
      self.send(0.2)
    assert self.send(0.35)[2] == 545

  def test_held_on_the_brake_with_nothing_asked_it_does_not_brake(self):
    self.settle(-0.5)
    mode, brake, gas = self.send(0.05)
    assert mode == 13 and brake == 41 and gas == 0

  def test_a_new_request_in_the_hold_builds_up_and_a_hard_one_is_not_held_back(self):
    self.settle(-0.5)
    for _ in range(50):
      self.send(0.05)
    assert self.brake_at(15, -0.2) > 44  # 0.3 s in
    self.setup_method()
    self.settle(-0.5)
    for _ in range(50):
      self.send(0.05)
    assert self.send(-3.5)[1] == 107

  def test_braking_again_restarts_the_hold(self):
    self.settle(-0.5)
    for _ in range(90):
      self.send(-0.05)
    self.send(-0.5)
    assert {self.send(-0.05)[0] for _ in range(90)} == {13}

  def test_a_climb_asks_for_more_and_a_descent_brakes(self):
    climb = math.atan(.07)
    flat = self.settle(0.5)
    self.setup_method()
    assert self.settle(0.5, climb)[2] > flat[2]
    self.setup_method()
    mode, brake, gas = self.settle(0., math.atan(-.14))
    assert mode == 13 and brake == 41 and gas == 0  # with no braking asked, on the brake mode the car holds the speed: -0.04 m/s² on descents

  def test_mild_deceleration_eases_off_the_engine_before_the_brake(self):
    mode, brake, gas = self.send(-0.14)
    assert mode == 12 and brake == 0 and gas == -192

  def test_braking_flags_are_the_cameras_so_regen_works(self):
    self.send(-0.5)
    frame = self.frame(-0.5)
    assert frame[9] & 0x1F == 13 and frame[27] >> 5 == 3 and frame[29] >> 7 == 0
    frame = self.frame(1.)
    assert frame[9] & 0x1F == 12 and frame[27] >> 5 == 4 and frame[29] >> 7 == 1

  def brake_at(self, frames, accel):
    return [self.send(accel)[1] for _ in range(frames)][-1]

  def test_moderate_braking_builds_slowly(self):
    assert self.brake_at(5, -0.9) < 50  # 0.1 s in, at 1 m/s^3 from -0.25
    self.setup_method()
    assert self.brake_at(60, -0.9) >= 57

  def test_hard_braking_is_never_held_back(self):
    assert self.send(-2.)[1] >= 78
    self.setup_method()
    self.send(-0.5)
    assert self.send(-3.5)[1] == 107

  def test_a_climb_never_takes_braking_away(self):
    flat = self.settle(-1.)
    self.setup_method()
    assert self.settle(-1., math.atan(.1))[1] == flat[1]

  def test_the_slope_is_left_out_of_a_stop(self):
    stopping = structs.CarControl.Actuators.LongControlState.stopping
    self.settle(0., math.atan(.1))
    assert self.send(-0.75, math.atan(.1), stopping)[0] == 13

  def stop(self, accel=-0.75, v_ego=0.2, frames=1, active=True):
    stopping = structs.CarControl.Actuators.LongControlState.stopping
    self.CS.out.vEgo = v_ego
    return [self.send(accel, state=stopping, active=active) for _ in range(frames)][-1]

  def test_the_end_of_a_stop_is_the_stock_hold(self):
    self.stop(v_ego=1.)
    assert self.stop(v_ego=1.)[0] == 13
    frame = self.frame(-0.75, state=structs.CarControl.Actuators.LongControlState.stopping)
    self.CS.out.vEgo = 0.2
    frame = self.frame(-0.75, state=structs.CarControl.Actuators.LongControlState.stopping)
    # the camera's own: 25, standstill 1, 3 and 0, no gas
    assert frame[9] & 0x1F == 25 and frame[12] >> 5 & 1 == 1 and frame[27] >> 5 == 3 and frame[29] >> 7 == 0
    assert ((frame[27] & 0x1F) << 8 | frame[28]) - 192 == 0

  def test_the_hold_climbs_to_the_stock_request_in_about_two_seconds(self):
    assert self.stop()[1] == 55  # -0.75 to start with
    brakes = [self.stop()[1] for _ in range(150)]
    assert brakes[49] in (65, 66, 67)  # 1 s in: -1.35 m/s^2
    assert brakes.index(81) in range(105, 120)  # -2.12, like the camera's 100
    assert max(brakes) == 81 and brakes == sorted(brakes)

  def test_the_hold_never_eases_a_harder_request(self):
    assert self.stop(-3.)[1] == 98
    assert self.stop(-3.5)[1] == 107

  def test_once_holding_a_speed_reading_does_not_end_it(self):
    self.stop(frames=10)
    assert self.stop(v_ego=0.4)[0] == 25

  def test_a_stop_above_the_hold_speed_is_not_held(self):
    assert {self.stop(v_ego=v)[0] for v in (5., 2., 1., 0.3)} == {13}

  def test_leaving_the_hold_goes_straight_to_the_gas(self):
    self.stop(frames=200)
    self.CS.out.vEgo = 0.
    mode, brake, gas = self.send(0.3)
    assert mode == 12 and brake == 0 and gas > 0

  def test_the_hold_is_only_for_openpilot_driving(self):
    assert self.stop(active=False)[1:] == (0, 0)

  def rolling_back(self, gear=structs.CarState.GearShifter.drive, gas=False):
    self.CS.rolling_back = True
    self.CS.out.gearShifter = gear
    self.CS.out.gasPressed = gas
    self.CS.out.vEgo = 0.1

  def test_rolling_back_in_drive_holds_at_once(self):
    self.settle(0.3)
    self.rolling_back()
    assert self.send(0.3)[:2] == (25, 81)
    self.setup_method()
    self.stop(frames=2)
    self.rolling_back()
    assert self.stop(v_ego=0.1)[:2] == (25, 81)

  def test_rolling_back_never_eases_a_harder_request(self):
    self.rolling_back()
    assert self.send(-3.)[1] == 98

  def test_once_stopped_openpilot_drives_on(self):
    self.rolling_back()
    self.send(0.3)
    self.CS.rolling_back = False
    self.CS.out.vEgo = 0.
    assert self.send(0.3)[0] == 12

  def test_reverse_and_the_gas_pedal_are_no_rolling_back(self):
    self.rolling_back(gear=structs.CarState.GearShifter.reverse)
    assert self.send(0.3)[0] == 12
    self.setup_method()
    self.rolling_back(gas=True)
    assert self.send(0.3)[0] == 12

  def test_the_pedal_takes_over_from_the_hold_in_its_mode(self):
    self.CS.acc_driving = True
    self.stock(12)
    self.stop(frames=200)
    self.pedal()
    frames = [self.frame(0., active=False) for _ in range(9)]
    assert {(f[9] & 0x1F, 181 - f[13], f[12] >> 5 & 1) for f in frames} == {(25, 81, 1)}

  def test_inactive_asks_for_nothing(self):
    assert self.send(1., active=False)[1:] == (0, 0)
    self.send(-2.)
    assert self.send(-2., active=False)[1:] == (0, 0)
