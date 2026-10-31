from types import SimpleNamespace

from opendbc.can.dbc import DBC
from opendbc.car import structs
from opendbc.car.gwm.interface import CarInterface
from opendbc.car.gwm.values import CAR, DBC as GWM_DBC
from opendbc.car import Bus


def zeros(dbc, msg):
  return {sig: 0 for sig in dbc.name_to_msg[msg].sigs}


class TestGwmCarController:
  def setup_method(self):
    CP = CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6)
    self.cc = CarInterface(CP).CC
    dbc = DBC(GWM_DBC[CP.carFingerprint][Bus.pt])
    self.CS = SimpleNamespace(
      out=structs.CarState(),
      camera_stock_values=zeros(dbc, "STEER_CMD"),
      eps_stock_values=zeros(dbc, "RX_STEER_RELATED"),
      steer_and_ap_stalk_msg=zeros(dbc, "STEER_AND_AP_STALK"),
      hud_stock_values=zeros(dbc, "LATERAL_STATE"),
      longitudinal_stock_values=zeros(dbc, "ACC_CMD"),
    )

  def _steer(self, torque, driver_torque, frames=40):
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

  def test_light_touch_keeps_torque(self):
    assert self._steer(1.0, -60) > 0
