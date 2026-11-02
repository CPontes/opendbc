import math
from types import SimpleNamespace

import numpy as np

from opendbc.can.dbc import DBC
from opendbc.car import structs
from opendbc.car.gwm import gwmcan
from opendbc.car.gwm.interface import CarInterface
from opendbc.car.gwm.values import CAR, DBC as GWM_DBC
from opendbc.car import Bus


def zeros(dbc, msg):
  return bytes(dbc.name_to_msg[msg].size)


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
                              acc_waiting=False, acc_driving=False, stalk_idle=True)

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

  def test_the_brake_lets_go_once_the_request_comes_back(self):
    assert self.send(-0.5)[0] == 13
    assert self.send(-0.05)[0] == 12

  def test_a_climb_asks_for_more_and_a_descent_brakes(self):
    climb = math.atan(.07)
    flat = self.settle(0.5)
    self.setup_method()
    assert self.settle(0.5, climb)[2] > flat[2]
    self.setup_method()
    mode, brake, gas = self.settle(0., math.atan(-.14))
    assert mode == 13 and brake > 41 and gas == 0

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

  def test_inactive_asks_for_nothing(self):
    assert self.send(1., active=False)[1:] == (0, 0)
    self.send(-2.)
    assert self.send(-2., active=False)[1:] == (0, 0)
