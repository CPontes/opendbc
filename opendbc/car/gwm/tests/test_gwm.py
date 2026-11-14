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


# A reversal recorded on the Haval (LAT-01, 08/11): openpilot's request swings from full left to full right in 0.2 s while
# the driver pushes left the whole time (1.2 to 2.4 Nm). 100 Hz, from 0.3 s before the rule of today let go to 0.5 s after;
# it had 241 out, and dropped it to 0 at once.
REVERSAL_REQUEST = [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
  1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.953, 0.822, 0.690, 0.557, 0.425, 0.282, 0.133,
  -0.015, -0.287, -0.289, -0.522, -0.535, -0.660, -0.753, -0.798, -0.834, -0.867, -0.892, -0.974, -1.000, -1.000,
  -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000,
  -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000,
  -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000, -1.000]
REVERSAL_DRIVER = [122, 131, 131, 133, 133, 140, 140, 135, 135, 139, 139, 128, 128, 145, 145, 159, 159, 156, 156,
  161, 161, 169, 169, 172, 172, 152, 152, 159, 159, 171, 171, 189, 189, 203, 203, 168, 168, 168, 168, 196, 196, 185,
  185, 201, 201, 225, 225, 194, 194, 187, 187, 198, 198, 204, 204, 191, 191, 190, 190, 200, 200, 201, 201, 191, 191,
  219, 219, 200, 200, 191, 191, 240, 240, 236, 236, 224, 224, 227, 227, 236]


class TestGwmOpposeBoth(TestGwmCarController):
  # with GwmFlags.OPPOSE_BOTH the driver only counts as against openpilot when against the request and the torque going out;
  # every test of TestGwmCarController runs here too
  FLAGS = GwmFlags.OPPOSE_BOTH.value

  def setup_method(self):
    super().setup_method()
    CP = CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6)
    CP.flags |= self.FLAGS
    self.cc = CarInterface(CP).CC

  def _replay(self, request, driver, warmup=200, tail=150, disengage_at=None):
    """the recorded request and driver at 100 Hz, after warmup frames of the first sample and before tail frames of the last;
    the EPS follows what went out. Returns (torque out, requested) at each 50 Hz frame of the recorded part and the tail.
    disengage_at: the 100 Hz frame, counted from the end of the warmup, from which openpilot no longer steers"""
    request = [request[0]] * warmup + list(request) + [request[-1]] * tail
    driver = [driver[0]] * warmup + list(driver) + [driver[-1]] * tail
    self.CS.out.vEgo = 15.
    out, applied = [], 0
    for k, (r, d) in enumerate(zip(request, driver, strict=True)):
      CC = structs.CarControl()
      CC.latActive = CC.enabled = disengage_at is None or k < warmup + disengage_at
      CC.actuators.torque = r
      self.CS.out.steeringTorque = d
      self.CS.out.steeringTorqueEps = applied
      actuators, _ = self.cc.update(CC.as_reader(), self.CS, 0)
      if self.cc.frame % 2 == 1:  # the frame that just ran sent the steering
        applied = actuators.torqueOutputCan
        if k >= warmup:
          out.append((applied, self.cc.lat_active))
    return out

  def test_the_reversal_comes_down_at_the_panda_rate(self):
    out = self._replay(REVERSAL_REQUEST, REVERSAL_DRIVER)
    torque = [o[0] for o in out]
    assert torque[0] == 253
    # never a step, never back up; the last step may land on 0 from 7 (a 1 goes out as 0: the EPS faults on a repeated 1)
    assert all(0 <= a - b <= 6 or (b == 0 and a <= 7) for a, b in zip(torque, torque[1:], strict=False))
    assert min(torque) >= 0  # never pushes on the driver, who is pushing left
    first_zero = torque.index(0)
    assert all(o[1] for o in out[:first_zero])  # requested while there is torque
    assert all(o == (0, False) for o in out[first_zero + 1:])  # then lets go: the driver is against both
    assert torque[first_zero - 1] == 7 and 2 not in torque  # from 253 the last step lands on 1: it goes out as 0, never as 2

  def test_from_250_it_never_crosses(self):
    # from 253 the last step of 6 lands on 1, which goes out as 0 anyway: from 250 it lands on 4, and the next step shows the
    # target. Coming down to zero it stops there; towards the request on the other side it would push the driver (the QA)
    torque = [o[0] for o in self._replay([250 / 253, -1.], [150, 150], tail=200)]
    assert torque[0] == 250 and 4 in torque
    assert min(torque) >= 0 and torque[-1] == 0

  def test_a_light_hand_is_not_eased(self):
    # under 1 Nm the driver isn't against openpilot at all: the torque follows the request across zero, without stopping there
    torque = [o[0] for o in self._replay([250 / 253, -1.], [60, 60], tail=200)]
    last_positive = max(i for i, t in enumerate(torque) if t > 0)
    assert torque[last_positive + 1] < 0 and min(torque) < -100

  def test_disengaged_mid_descent_it_is_zero_in_the_same_send(self):
    # openpilot no longer steers: the torque goes to 0 with the request in the first send after, the descent doesn't go on
    out = self._replay(REVERSAL_REQUEST, REVERSAL_DRIVER, disengage_at=len(REVERSAL_REQUEST))
    eased = len(REVERSAL_REQUEST) // 2  # the recorded part, at 50 Hz: the help was coming down
    assert 0 < out[eased - 1][0] < 253
    assert all(o == (0, False) for o in out[eased:])

  def test_against_the_request_and_the_torque_lets_go_at_once(self):
    self._steer(1.0, 0, frames=200)
    assert self._steer(1.0, -150, frames=2) == 0 and not self.cc.lat_active

  def test_the_panda_takes_every_frame_of_the_reversal(self):
    # in a closed loop the EPS follows what went out, as in the car, and the panda takes every frame. (In the log replay the
    # panda refused 8 frames of one eased reversal: there it had already taken control away, the ACC off, while openpilot
    # still steered; the torque was within what it measured. LAT-01, the QA's notice)
    from opendbc.car.structs import CarParams
    from opendbc.safety.tests.libsafety import libsafety_py
    from opendbc.safety.tests.test_gwm import GwmPacker, opendbc as safety_dbc
    safety = libsafety_py.libsafety
    safety.set_safety_hooks(CarParams.SafetyModel.gwm, 0)
    safety.init_tests()
    safety.set_controls_allowed(True)
    packer = GwmPacker(safety_dbc)
    request = [REVERSAL_REQUEST[0]] * 200 + REVERSAL_REQUEST + [REVERSAL_REQUEST[-1]] * 150
    driver = [REVERSAL_DRIVER[0]] * 200 + REVERSAL_DRIVER + [REVERSAL_DRIVER[-1]] * 150
    self.CS.out.vEgo = 15.
    applied, sent, refused = 0, 0, 0
    for k, (r, d) in enumerate(zip(request, driver, strict=True)):
      safety.set_timer(k * 10_000)
      assert safety.safety_rx_hook(packer.make_can_msg_safety("RX_STEER_RELATED", 0, {"B_RX_EPS_TORQUE": applied}))
      CC = structs.CarControl()
      CC.latActive = CC.enabled = True
      CC.actuators.torque = r
      self.CS.out.steeringTorque = d
      self.CS.out.steeringTorqueEps = applied
      actuators, sends = self.cc.update(CC.as_reader(), self.CS, 0)
      for addr, dat, bus in sends:
        if addr == 0x12B:
          sent += 1
          refused += not safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat))
      if self.cc.frame % 2 == 1:  # the frame that just ran sent the steering
        applied = actuators.torqueOutputCan
    assert sent > 150 and refused == 0
    assert safety.get_controls_allowed()

  def test_with_no_torque_out_the_request_decides(self):
    # zero going out is on both sides: as today
    assert self._steer(1.0, -150) == 0
    assert self._steer(1.0, 150) > 0


class TestGwmOpposeBothOff(TestGwmOpposeBoth):
  FLAGS = 0

  def test_the_reversal_comes_down_at_the_panda_rate(self):
    # off by default: the rule of today drops the help at once, as recorded
    torque = [o[0] for o in self._replay(REVERSAL_REQUEST, REVERSAL_DRIVER)]
    assert max(a - b for a, b in zip(torque, torque[1:], strict=False)) >= 200 and torque[-1] == 0

  def test_from_250_it_never_crosses(self):
    # off: from 250 too it drops at once, and never to the other side
    torque = [o[0] for o in self._replay([250 / 253, -1.], [150, 150], tail=200)]
    assert torque[0] == 250 and max(a - b for a, b in zip(torque, torque[1:], strict=False)) >= 200
    assert min(torque) >= 0 and torque[-1] == 0

  def test_disengaged_mid_descent_it_is_zero_in_the_same_send(self):
    # off: there is no descent, it had already let go; disengaged, it stays at 0
    out = self._replay(REVERSAL_REQUEST, REVERSAL_DRIVER, disengage_at=len(REVERSAL_REQUEST))
    assert all(o == (0, False) for o in out[len(REVERSAL_REQUEST) // 2:])


class TestGwmOpposeBothWithSoftHandoff(TestGwmOpposeBoth):
  # with both: the reversal still comes down at the panda's rate; a light push against both fades as the soft handoff does
  FLAGS = GwmFlags.OPPOSE_BOTH.value | GwmFlags.SOFT_HANDOFF.value

  def test_against_the_request_and_the_torque_lets_go_at_once(self):
    self._steer(1.0, 0, frames=200)
    assert self._steer(1.0, -160, frames=2) == 0 and not self.cc.lat_active  # firm: at once, as the soft handoff


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

  def _buttons(self, frames, decrease=False, increase=False):
    CC = structs.CarControl()
    CC.enabled = True
    CC.cruiseControl.decrease = decrease
    CC.cruiseControl.increase = increase
    CC = CC.as_reader()
    out = []
    for _ in range(frames):
      sent = [dat for addr, dat, _bus in self.cc.update(CC, self.CS, 0)[1] if addr == 0xA1]
      out.append("+" if sent and sent[0][6] & 0x08 else "-" if sent and sent[0][6] & 0x04 else ".")
    return "".join(out)

  def test_plus_is_a_press_like_the_drivers_then_a_gap(self):  # CRUISE-01
    assert self._buttons(13 + 35 + 13, increase=True) == "+" * 13 + "." * 35 + "+" * 13

  def test_plus_only_with_the_acc_driving_a_moving_car_and_nobody_on_the_stalk(self):
    for attr, value in (("acc_driving", False), ("stalk_idle", False)):
      setattr(self.CS, attr, value)
      assert self._buttons(20, increase=True) == "." * 20, attr
      setattr(self.CS, attr, True)
    self.CS.out.standstill = True
    assert self._buttons(20, increase=True) == "." * 20
    self.CS.out.standstill = False
    self.CS.steer_and_ap_stalk_msg = None
    assert self._buttons(20, increase=True) == "." * 20

  def test_the_minus_wins_over_the_plus(self):
    assert set(self._buttons(13, decrease=True, increase=True)) == {"-"}

  def test_a_plus_never_right_after_a_minus(self):
    seq = self._buttons(5, decrease=True) + self._buttons(60, increase=True)
    assert seq.startswith("-----") and "+" in seq
    assert seq.index("+") - seq.rindex("-") - 1 >= 35  # the gap of the minus, before any plus

  def test_no_original_frame_no_cancel(self):
    self.CS.steer_and_ap_stalk_msg = None
    assert self._stalk_frames(5, resume=False, cancel=True) == (0, 0)


# LONG-06: a touch of the gas at a stop, as openpilot sent it on route 4b (14 min), one tuple per 100 Hz frame: the longitudinal
# state, the acceleration, longActive and the gas pedal. The pedal for 0.28 s, openpilot back with the longitudinal still off for a
# frame, the car never moving. Only the numbers of the request and the pedal
TAP_4B = ([("stopping", -0.75, True, False)] * 30 + [("stopping", 0., False, True)] + [("off", 0., False, True)] * 27 +
          [("off", -0.01, True, False)] + [("stopping", round(-0.02 - 0.01 * k, 2), True, False) for k in range(32)])
TAP_4B_BACK = 58  # the first frame back
TAP_4B_CAMERA_LETS_GO = 43  # the camera went from its hold to the gas 0.13 s into the pedal
TAP_4B_PEDAL = 0.07  # how far down the pedal went, at most (18 in the byte)


class TestGwmLongitudinal:
  FLAGS = 0

  def setup_method(self):
    CP = CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6)
    CP.openpilotLongitudinalControl = True
    CP.flags |= int(self.FLAGS)
    self.cc = CarInterface(CP).CC
    dbc = DBC(GWM_DBC[CP.carFingerprint][Bus.pt])
    self.CS = SimpleNamespace(out=structs.CarState(), camera_stock_data=None, eps_stock_data=None, hud_stock_data=None,
                              longitudinal_stock_data=zeros(dbc, "ACC_CMD"), steer_and_ap_stalk_msg=zeros(dbc, "STEER_AND_AP_STALK"),
                              acc_waiting=False, acc_driving=False, stalk_idle=True, rolling_back=False, gas_pedal=0.)
    self.CS.out.vEgo = 11.

  def send(self, accel, pitch=0., state=None, active=True):
    frame = self.frame(accel, pitch, state, active)
    return frame[9] & 0x1F, 181 - frame[13], ((frame[27] & 0x1F) << 8 | frame[28]) - 192

  def frame(self, accel, pitch=0., state=None, active=True, enabled=None):
    CC = structs.CarControl()
    CC.enabled = active if enabled is None else enabled
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

  def test_a_braking_frame_out_of_the_stop_keeps_the_hold(self):
    # a should_stop that flickers: one pid frame asking a little braking must not drop the hold to regen
    self.stop(frames=200)
    self.CS.out.vEgo = 0.
    assert self.send(-0.05)[:2] == (25, 81)
    assert self.stop(v_ego=0.)[:2] == (25, 81)

  def stopped(self):
    self.CS.out.vEgo = 0.
    self.CS.out.standstill = True

  def test_our_hold_never_carries_the_cameras_drive_off(self):
    # LONG-05: the camera sets the top bits of byte 9 to 2 as it leaves a stop. Passed on with our hold, the car let the brake
    # go with the car stopped and it rolled back on a climb (route 41, 37.5 min)
    stopping = structs.CarControl.Actuators.LongControlState.stopping
    self.stopped()
    self.stock(12 | 0x40)
    for _ in range(50):
      frame = self.frame(-0.75, state=stopping)
    assert frame[9] & 0x1F == 25 and frame[9] >> 5 == 0
    assert frame[8] == gwmcan.checksum(frame[9:16], 0xEF)  # the brake block's CRC is made after the bits are cleared

  def test_our_brake_never_carries_it_either(self):
    self.CS.out.vEgo = 5.
    self.stock(12 | 0x40)
    frame = self.frame(-1.5)
    assert frame[9] & 0x1F == 13 and frame[9] >> 5 == 0

  def test_the_handoff_to_the_pedal_never_carries_it(self):
    stopping = structs.CarControl.Actuators.LongControlState.stopping
    self.stopped()
    for _ in range(100):
      self.frame(-0.75, state=stopping)
    self.stock(12 | 0x40)
    self.CS.acc_driving = True
    self.pedal()
    frame = self.frame(0., active=False)  # openpilot let go at the pedal: our brake stays a moment, in its mode
    assert frame[9] & 0x1F in (13, 25) and frame[9] >> 5 == 0

  def test_our_gas_keeps_the_cameras_bits(self):
    # leaving a stop is the camera's own request too: on the gas it goes as the camera sends it
    self.CS.out.vEgo = 2.
    self.stock(12 | 0x40)
    frame = self.frame(0.5)
    assert frame[9] & 0x1F == 12 and frame[9] >> 5 == 2

  def test_a_stop_back_before_the_car_moves_keeps_the_hold(self):
    # behind a car creeping on, the planner goes to the gas for a frame and back to stopping (route 40, 2.8 min): the hold
    # comes back where it was, not from nothing, or the pressure falls with the car still stopped. Out of a gas frame the
    # stopping request starts near zero, as it did there
    self.stop(frames=200)
    self.stopped()
    for _ in range(3):
      assert self.send(0.1)[0] == 12
      assert self.stop(-0.01, v_ego=0.)[:2] == (25, 81)

  def test_a_frame_of_stopping_in_a_departure_is_no_jab_of_hold(self):
    # the other flicker: the car already leaving, one frame of stopping must not bring the whole hold back on a moving car
    self.stop(frames=200)
    self.stopped()
    self.send(0.5)
    self.CS.out.standstill = False
    for v in (0.02, 0.05, 0.08, 0.11, 0.14, 0.16):
      self.CS.out.vEgo = v
      assert self.send(0.5)[0] == 12
    assert self.stop(-0.01, v_ego=0.17)[1] < 50
    self.CS.out.vEgo = 0.18
    assert self.send(0.5)[0] == 12

  def test_with_the_pedal_the_kept_hold_does_not_come_back(self):
    # the panda takes its level at the pedal and refuses more: the hold stays at the last frame, not at the kept one
    self.stop(frames=200)
    self.stopped()
    self.send(0.1)
    self.pedal()
    assert self.stop(-0.01, v_ego=0.)[1] < 50

  def test_a_harder_stop_comes_back_at_the_stock_hold(self):
    self.stop(-3., frames=50)
    self.stopped()
    self.send(0.1)
    assert self.stop(-0.3, v_ego=0.)[:2] == (25, 81)  # -2.12, what held 860 of pressure on the climb

  def test_the_kept_hold_ends_when_the_car_moves_on(self):
    self.stop(frames=200)
    self.CS.out.vEgo = 1.
    self.send(0.3)
    assert self.stop()[:2] == (25, 55)  # a new stop builds up again from what is asked

  def test_the_kept_hold_ends_with_openpilot(self):
    self.stop(frames=200)
    self.stop(active=False)
    assert self.stop()[:2] == (25, 55)

  def test_the_hold_does_not_follow_a_car_moving_on(self):
    self.stop(frames=200)
    self.CS.out.vEgo = 1.
    assert self.send(-0.5)[0] == 13

  def test_the_hold_ends_with_openpilot(self):
    self.stop(frames=50)
    self.stop(active=False)
    assert self.stop(v_ego=0.4)[0] == 13

  def test_with_the_pedal_the_hold_stops_climbing(self):
    self.stop(frames=50)
    self.pedal()
    climbing = [self.stop()[1] for _ in range(20)]
    assert len(set(climbing)) == 1
    self.pedal(False)
    assert self.stop(frames=60)[1] > climbing[0]

  def test_a_stop_above_the_hold_speed_is_not_held(self):
    assert {self.stop(v_ego=v)[0] for v in (5., 2., 1., 0.3)} == {13}

  def test_leaving_the_hold_goes_straight_to_the_gas(self):
    self.stop(frames=200)
    self.CS.out.vEgo = 0.
    mode, brake, gas = self.send(0.3)
    assert mode == 12 and brake == 0 and gas > 0

  def test_the_hold_is_only_for_openpilot_driving(self):
    assert self.stop(active=False)[1:] == (0, 0)

  # LONG-06: back from a touch of the gas at a stop. On route 4b (14 min) the first frame back was a frame of gas and the hold climbed
  # again from 42; the pressure, already falling, went to 0 with the car stopped
  def touch_gas(self, frames=14, v_ego=0., enabled=True, first_active=False):
    # the driver on the gas for frames x 20 ms: openpilot lets go of the longitudinal and stays engaged
    off = structs.CarControl.Actuators.LongControlState.off
    self.CS.out.gasPressed = True
    self.CS.out.vEgo = v_ego
    self.CS.out.standstill = v_ego == 0.
    if first_active:  # openpilot lets go a frame after the pedal
      self.frame(0., state=structs.CarControl.Actuators.LongControlState.stopping)
    for _ in range(frames):
      self.frame(0., state=off, active=False, enabled=enabled)
    self.CS.out.gasPressed = False

  def back(self, accel=-0.01, state=structs.CarControl.Actuators.LongControlState.off):
    # the first frame back, the car stopped: openpilot sends it before the planner is stopping again
    self.CS.out.vEgo = 0.
    self.CS.out.standstill = True
    return self.send(accel, state=state)

  def held_stop(self):
    self.stopped()
    self.stop(v_ego=0., frames=200)

  def replay_touch(self, seq, panda=None):
    LongCtrlState = structs.CarControl.Actuators.LongControlState
    out = []
    for k, (state, accel, active, gas) in enumerate(seq):
      CC = structs.CarControl()
      CC.enabled = True
      CC.longActive = active
      CC.actuators.accel = accel
      CC.actuators.longControlState = getattr(LongCtrlState, state)
      self.CS.out.gasPressed = gas
      self.CS.gas_pedal = TAP_4B_PEDAL if gas else 0.
      if panda is not None:
        panda(k, gas)
      for addr, dat, bus in self.cc.update(CC.as_reader(), self.CS, 0)[1]:
        if addr == 0x143:
          out.append((k, dat[9] & 0x1F, 181 - dat[13], addr, dat, bus))
    return out

  def panda_touch(self):
    # the recorded touch of route 4b through the panda: the camera holds, then lets go to the gas 0.13 s into the pedal; the panda
    # sees the car stopped and the pedal. Every 0x143, whether the panda took it, and whether control is still allowed at the end
    from opendbc.car.gwm.values import GwmSafetyFlags
    from opendbc.car.structs import CarParams
    from opendbc.safety.tests.libsafety import libsafety_py
    from opendbc.safety.tests.test_gwm import GwmPacker, checksum, opendbc as safety_dbc
    safety = libsafety_py.libsafety
    safety.set_safety_hooks(CarParams.SafetyModel.gwm, GwmSafetyFlags.LONG_CONTROL)
    safety.init_tests()
    safety.set_controls_allowed(True)
    packer = GwmPacker(safety_dbc)
    self.held_stop()
    self.stock(25)

    def panda(k, gas):
      safety.set_timer(k * 10_000)
      if k == TAP_4B_CAMERA_LETS_GO:
        self.stock(12 | 0x40)  # with its drive-off bits, as it resumes
      speed = {f"{w}_WHEEL_SPEED": 0. for w in ("FRONT_LEFT", "FRONT_RIGHT", "REAR_LEFT", "REAR_RIGHT")}
      assert safety.safety_rx_hook(packer.make_can_msg_safety("WHEEL_SPEEDS", 0, speed, fix_checksum=checksum))
      pedal = TAP_4B_PEDAL * 100 if gas else 0
      assert safety.safety_rx_hook(packer.make_can_msg_safety("CAR_OVERALL_SIGNALS2", 0, {"GAS_POSITION": pedal}))

    out = self.replay_touch(TAP_4B, panda)
    refused = [k for k, _m, _b, addr, dat, bus in out if not safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat))]
    return out, refused, safety.get_controls_allowed()

  def press(self, frames, pedal=0.2, v_ego=0., enabled=True, brake=False):
    # the driver's gas for frames x 20 ms, openpilot letting go of the longitudinal and staying engaged: (mode, brake, drive-off bits)
    off = structs.CarControl.Actuators.LongControlState.off
    self.CS.out.gasPressed = True
    self.CS.out.brakePressed = brake
    self.CS.gas_pedal = pedal
    self.CS.out.vEgo = v_ego
    self.CS.out.standstill = v_ego == 0.
    out = []
    for _ in range(frames):
      f = self.frame(0., state=off, active=False, enabled=enabled)
      out.append((f[9] & 0x1F, 181 - f[13], f[9] >> 5))
    return out

  def lift(self):
    self.CS.out.gasPressed = False
    self.CS.out.brakePressed = False
    self.CS.gas_pedal = 0.
    return self.back()

  def test_without_the_tap_option_the_pedal_takes_the_hold_at_once(self):
    # LONG-07 off: once the camera resumes, the pedal's frame is the gas override, as before
    self.held_stop()
    self.stock(12 | 0x40)
    assert {m for m, *_ in self.press(5)} == {20}

  def test_without_the_option_a_touch_comes_back_as_before(self):
    # LONG-06 off, the default: the first frame back is a frame of gas and the hold climbs again from nothing, as on route 4b
    self.held_stop()
    back = [(m, b) for k, m, b, *_ in self.replay_touch(TAP_4B) if k >= TAP_4B_BACK]
    assert back[0][0] == 12 and back[1][0] == 25 and back[1][1] < 50

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

  def test_once_stopped_it_holds_a_moment_and_openpilot_drives_on(self):
    self.rolling_back()
    self.send(0.3)
    self.CS.rolling_back = False
    self.CS.out.vEgo = 0.
    modes = [self.send(0.3)[0] for _ in range(60)]
    assert modes[:49] == [25] * 49 and modes[-1] == 12  # 1 s at 50 Hz

  def test_rolling_back_again_restarts_the_moment(self):
    self.rolling_back()
    self.send(0.3)
    self.CS.rolling_back = False
    for _ in range(40):
      self.send(0.3)
    self.CS.rolling_back = True
    self.send(0.3)
    self.CS.rolling_back = False
    assert {self.send(0.3)[0] for _ in range(45)} == {25}

  def test_rolling_back_is_only_at_a_crawl(self):
    self.settle(0.3)
    self.rolling_back()
    self.CS.out.vEgo = 20.
    assert self.send(0.3)[0] == 12

  @pytest.mark.parametrize("gear", ["neutral", "park", "unknown"])
  def test_out_of_drive_is_no_rolling_back(self, gear):
    self.rolling_back(gear=getattr(structs.CarState.GearShifter, gear))
    assert self.send(0.3)[0] == 12

  def test_rolling_back_without_openpilot_asks_for_nothing(self):
    self.rolling_back()
    assert self.send(0.3, active=False)[1:] == (0, 0)

  def test_the_moment_ends_with_the_gas_pedal(self):
    self.rolling_back()
    self.send(0.3)
    self.CS.rolling_back = False
    self.CS.out.gasPressed = True
    self.send(0.3)
    self.CS.out.gasPressed = False
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


class TestGwmGasTouchHold(TestGwmLongitudinal):
  # LONG-06 with the option on (GwmFlags.GAS_TOUCH_HOLD): every longitudinal test above runs with it too
  FLAGS = GwmFlags.GAS_TOUCH_HOLD

  def test_back_from_a_touch_of_the_gas_is_the_stock_hold_at_once(self):
    self.held_stop()
    self.touch_gas()
    assert self.back()[:2] == (25, 81)  # no frame of gas, no hold from nothing
    assert {self.stop(-0.02 - 0.01 * k, v_ego=0.)[:2] for k in range(30)} == {(25, 81)}

  def test_the_recorded_touch_comes_back_at_the_hold_with_no_frame_of_gas(self):
    self.held_stop()
    assert self.cc.frame % 2 == 0  # the first frame back is a frame that sends, as it was on route 4b
    out = self.replay_touch(TAP_4B)
    assert 12 not in {m for _k, m, *_ in out}
    back = [(m, b) for k, m, b, *_ in out if k >= TAP_4B_BACK]
    assert out[[k for k, *_ in out].index(TAP_4B_BACK)][1:3] == (25, 81) and set(back) == {(25, 81)}

  def test_the_panda_takes_every_frame_of_the_recorded_touch(self):
    # the camera holds, then lets go to the gas 0.13 s into the pedal, as on route 4b; the panda sees the car stopped and the pedal
    out, refused, allowed = self.panda_touch()
    modes = {m for k, m, *_ in out if TAP_4B_BACK > k > TAP_4B_CAMERA_LETS_GO}
    assert modes == {20} and refused == [] and allowed  # on the pedal, the camera's own gas override

  def test_openpilot_letting_go_a_frame_after_the_pedal_also_comes_back_at_the_hold(self):
    self.held_stop()
    self.touch_gas(first_active=True)
    assert self.back()[:2] == (25, 81)

  def test_no_harder_hold_while_the_foot_is_on_the_gas(self):
    # early in a stop the hold is still climbing: the frame with the pedal, openpilot not yet letting go, never jumps to the stock hold
    self.stopped()
    climbing = self.stop(v_ego=0., frames=10)[1]
    self.CS.out.gasPressed = True
    assert self.send(0., state=structs.CarControl.Actuators.LongControlState.stopping)[1] < max(climbing + 3, 75)

  def test_a_long_press_that_never_moves_the_car_also_comes_back_at_the_hold(self):
    # what matters is the car not moving, not how long: 5 s on the gas with the car held still on a climb is the same risk
    self.held_stop()
    self.touch_gas(frames=250)
    assert self.back()[:2] == (25, 81)

  def test_a_touch_that_moved_the_car_starts_again_as_before(self):
    self.held_stop()
    self.touch_gas(v_ego=0.4)
    assert self.back()[0] == 12
    assert self.stop(-0.02, v_ego=0.)[1] < 50  # a new stop builds up from what is asked

  def test_back_with_the_wheels_turning_is_no_jab_of_hold(self):
    self.held_stop()
    self.touch_gas(v_ego=0.1)
    self.CS.out.vEgo = 0.1
    self.CS.out.standstill = False
    assert self.send(-0.01, state=structs.CarControl.Actuators.LongControlState.off)[0] == 12

  def test_a_departure_right_after_the_touch_goes_to_the_gas(self):
    self.held_stop()
    self.touch_gas()
    mode, brake, gas = self.back(0.3, structs.CarControl.Actuators.LongControlState.pid)
    assert mode == 12 and brake == 0 and gas > 0

  def test_the_brake_pedal_after_the_touch_changes_nothing(self):
    self.held_stop()
    self.touch_gas()
    self.pedal()
    for _ in range(10):
      self.frame(0., state=structs.CarControl.Actuators.LongControlState.off, active=False, enabled=True)
    self.pedal(False)
    assert self.back()[0] == 12

  def test_disengaged_during_the_touch_nothing_comes_back(self):
    self.held_stop()
    self.touch_gas(enabled=False)
    assert self.back()[0] == 12

  def test_a_touch_out_of_our_hold_changes_nothing(self):
    # stopped and leaving on the gas, not holding: the driver's touch is no stop of ours
    self.stopped()
    self.send(0.3)
    self.touch_gas()
    assert self.back()[0] == 12

  def test_openpilot_letting_go_later_with_no_pedal_is_as_before(self):
    # back from the touch, openpilot lets go of the longitudinal for a moment with nobody on the pedals: no touch, as before
    self.held_stop()
    self.touch_gas()
    self.back()
    self.frame(0., state=structs.CarControl.Actuators.LongControlState.off, active=False, enabled=True)
    assert self.back()[0] == 12

  def test_the_hold_back_from_the_touch_lasts_for_this_touch_only(self):
    self.held_stop()
    self.touch_gas()
    self.back()
    self.send(0.3)  # the car leaves for real
    self.CS.out.vEgo = 1.
    self.CS.out.standstill = False
    self.send(0.3)
    self.touch_gas(v_ego=1.)
    assert self.back()[0] == 12  # a touch on the move, then a stop: as before

  def test_without_the_option_a_touch_comes_back_as_before(self):
    # with the option: the stock hold at once (the test above)
    self.test_the_recorded_touch_comes_back_at_the_hold_with_no_frame_of_gas()


class TestGwmGasTapHold(TestGwmLongitudinal):
  # LONG-07 with the option on (GwmFlags.GAS_TAP_HOLD): every longitudinal test above runs with it too. The camera resumed, with its
  # drive-off bits, as it does a moment into the pedal
  FLAGS = GwmFlags.GAS_TAP_HOLD

  def resumed_stop(self):
    self.held_stop()
    self.stock(12 | 0x40)

  def test_a_touch_keeps_our_hold_with_no_brake(self):
    self.resumed_stop()
    assert set(self.press(10)) == {(25, 0, 0)}  # the hold mode, no brake (the panda takes none with the pedal), no drive-off bits
    assert self.lift()[:2] == (25, 81)  # and back at the stock hold, as with GAS_TOUCH_HOLD

  def test_the_hold_lasts_until_the_pedal_has_been_down_0_35_s(self):
    self.resumed_stop()
    modes = [m for m, *_ in self.press(19)]
    assert modes[:18] == [25] * 18 and modes[18] == 20  # 0.34 s held; at 0.36 s the gas override, in that frame

  def test_a_firm_pedal_goes_at_once(self):
    self.resumed_stop()
    assert self.press(1, pedal=0.36)[0][0] == 20
    self.setup_method()
    self.resumed_stop()
    assert self.press(1, pedal=0.35)[0][0] == 25  # up to 35% is a touch

  def test_the_wheels_turning_end_it(self):
    self.resumed_stop()
    self.press(3)
    assert self.press(1, v_ego=0.02)[0][0] == 20  # the panda's "moving": any wheel

  def test_the_brake_ends_it(self):
    self.resumed_stop()
    self.press(3)
    assert self.press(1, brake=True)[0][0] != 25

  def test_disengaged_ends_it(self):
    self.resumed_stop()
    self.press(3)
    assert self.press(1, enabled=False)[0][0] == 20

  def test_once_out_never_back_in_the_same_press(self):
    self.resumed_stop()
    self.press(1, pedal=0.4)
    assert {m for m, *_ in self.press(5)} == {20}
    self.setup_method()
    self.resumed_stop()
    self.press(2)
    self.press(1, v_ego=0.02)
    assert {m for m, *_ in self.press(5)} == {20}

  def test_a_new_press_starts_again(self):
    self.resumed_stop()
    self.press(1, pedal=0.4)
    self.lift()
    self.stop(v_ego=0., frames=20)
    assert {m for m, *_ in self.press(5)} == {25}

  def test_a_press_out_of_our_hold_is_as_before(self):
    # stopped and leaving on the gas, not holding: no hold for the driver's pedal
    self.stopped()
    self.stock(12 | 0x40)
    self.send(0.3)
    assert {m for m, *_ in self.press(5)} == {20}

  def test_openpilot_still_on_with_the_pedal_sends_no_brake(self):
    # the frame with the pedal before openpilot lets go: the hold with no brake, never a brake the panda would refuse
    self.resumed_stop()
    self.CS.out.gasPressed = True
    self.CS.gas_pedal = 0.2
    assert self.send(0., state=structs.CarControl.Actuators.LongControlState.stopping)[:2] == (25, 0)

  def test_the_recorded_touch_holds_through(self):
    self.held_stop()
    self.stock(25)
    LongCtrlState = structs.CarControl.Actuators.LongControlState
    out = []
    for k, (state, accel, active, gas) in enumerate(TAP_4B):
      if k == TAP_4B_CAMERA_LETS_GO:
        self.stock(12 | 0x40)
      CC = structs.CarControl()
      CC.enabled = True
      CC.longActive = active
      CC.actuators.accel = accel
      CC.actuators.longControlState = getattr(LongCtrlState, state)
      self.CS.out.gasPressed = gas
      self.CS.gas_pedal = TAP_4B_PEDAL if gas else 0.
      for addr, dat, _bus in self.cc.update(CC.as_reader(), self.CS, 0)[1]:
        if addr == 0x143:
          out.append((k, dat[9] & 0x1F, 181 - dat[13]))
    on_pedal = {(m, b) for k, m, b in out if 30 <= k < TAP_4B_BACK}
    back = {(m, b) for k, m, b in out if k >= TAP_4B_BACK}
    assert on_pedal == {(25, 0)} and back == {(25, 81)}

  def test_the_panda_takes_every_frame_of_the_recorded_touch_held(self):
    out, refused, allowed = self.panda_touch()
    modes = {m for k, m, *_ in out if TAP_4B_BACK > k > TAP_4B_CAMERA_LETS_GO}
    assert modes == {25} and refused == [] and allowed

  def test_without_the_tap_option_the_pedal_takes_the_hold_at_once(self):
    # with the option: the hold stays for a touch (the test above)
    self.test_a_touch_keeps_our_hold_with_no_brake()

  def test_without_the_option_a_touch_comes_back_as_before(self):
    # with the option: back at the stock hold (as with GAS_TOUCH_HOLD)
    self.test_a_touch_keeps_our_hold_with_no_brake()
