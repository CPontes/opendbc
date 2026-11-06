from types import SimpleNamespace


from opendbc.can.dbc import DBC
from opendbc.car import structs
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


class TestGwmCancel:
  def setup_method(self):
    CP = CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6)
    self.cc = CarInterface(CP).CC
    dbc = DBC(GWM_DBC[CP.carFingerprint][Bus.pt])
    self.CS = SimpleNamespace(out=structs.CarState(), camera_stock_data=None, eps_stock_data=None, hud_stock_data=None,
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

  def test_no_original_frame_no_cancel(self):
    self.CS.steer_and_ap_stalk_msg = None
    assert self._stalk_frames(5, resume=False, cancel=True) == (0, 0)
