from opendbc.car import structs, get_safety_config, Bus
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.gwm.carcontroller import CarController
from opendbc.car.gwm.carstate import CarState

STEER_FAULT_FRAMES = 100  # 1 s of the EPS not taking a torque we asked for


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController

  def __init__(self, CP):
    super().__init__(CP)
    self.steer_fault_temporary_counter = 0

  def apply(self, CC, now_nanos):
    self.CS.cancel_requested = CC.cruiseControl.cancel
    return super().apply(CC, now_nanos)

  def update(self, can_packets):
    ret = super().update(can_packets)
    # A fault only when the EPS doesn't take a torque the controller actually sent: carcontroller stops requesting while the
    # driver steers against openpilot, and the EPS not obeying then is expected
    eps_obeying = self.can_parsers[Bus.main].vl["RX_STEER_RELATED"]["A_RX_STEER_REQUESTED"] == 1
    self.steer_fault_temporary_counter = self.steer_fault_temporary_counter + 1 if (self.CC.lat_active and not eps_obeying) else 0
    ret.steerFaultTemporary |= self.steer_fault_temporary_counter > STEER_FAULT_FRAMES
    return ret

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    ret.brand = 'gwm'
    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.gwm)]
    ret.dashcamOnly = False

    ret.steerActuatorDelay = 0.3
    ret.steerLimitTimer = 0.4
    ret.steerAtStandstill = False
    ret.steerControlType = structs.CarParams.SteerControlType.torque
    CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    # the stock ACC drives: openpilot longitudinal is a later step, and no radar reaches the harness
    ret.radarUnavailable = True
    return ret
