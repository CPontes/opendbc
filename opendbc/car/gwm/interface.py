from opendbc.car import structs, get_safety_config, Bus, create_button_events
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.gwm.carcontroller import CarController
from opendbc.car.gwm.carstate import CarState
from opendbc.car.gwm.values import GwmSafetyFlags

ButtonType = structs.CarState.ButtonEvent.Type
TransmissionType = structs.CarParams.TransmissionType

# The car has 4 follow distances and openpilot 3 personalities: both farthest distances take the farthest bars.
FOLLOW_DISTANCE_BARS = {1: 1, 2: 2, 3: 3, 4: 3}
FOLLOW_DISTANCE_WAIT = 50  # frames for selfdrived to answer a press with new bars


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController

  def __init__(self, CP):
      super().__init__(CP)
      self.isEPSobeying = True
      self.steer_fault_temporary_counter = 0
      self.current_personality = 0
      self.pcm_follow_distance = 0
      self.follow_distance_wait = 0

  def apply(self, CC, now_nanos):
    hud_control = CC.hudControl
    self.current_personality = hud_control.leadDistanceBars
    self.CS.cancel_requested = CC.cruiseControl.cancel
    return super().apply(CC, now_nanos)

  def update(self, can_packets):
    cp = self.can_parsers[Bus.main]
    self.isEPSobeying = cp.vl["RX_STEER_RELATED"]["A_RX_STEER_REQUESTED"] == 1
    # carcontroller stops requesting torque while the driver steers against openpilot: the EPS not obeying
    # then is expected, it would turn every firm takeover longer than 1 s into a fault
    self.steer_fault_temporary_counter = (self.steer_fault_temporary_counter + 1) if (self.CC.lat_active and not self.isEPSobeying) \
                                          else 0

    cp_cam = self.can_parsers[Bus.cam]
    self.pcm_follow_distance = cp_cam.vl["ACC"]["CAR_DISTANCE_SELECTION"]

    ret = super().update(can_packets)
    ret.steerFaultTemporary |= self.steer_fault_temporary_counter > 100
    # copy first: setting the field wipes the old list, an event from CarState would come out as unknown
    car_state_events = [be.copy() for be in ret.buttonEvents]
    ret.buttonEvents = [*car_state_events, *self.follow_distance_events()]

    return ret

  def follow_distance_events(self):
    # Only openpilot longitudinal has personalities. Each release is one step, so press once and wait for the bars.
    self.follow_distance_wait = max(self.follow_distance_wait - 1, 0)
    bars = FOLLOW_DISTANCE_BARS.get(self.pcm_follow_distance)
    if not self.CP.openpilotLongitudinalControl or bars is None or self.current_personality == 0:
      return []
    if bars == self.current_personality or self.follow_distance_wait > 0:
      return []
    self.follow_distance_wait = FOLLOW_DISTANCE_WAIT
    return create_button_events(0, 1, {1: ButtonType.gapAdjustCruise})

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

    ret.radarUnavailable = True

    ret.alphaLongitudinalAvailable = True
    if alpha_long:
      ret.openpilotLongitudinalControl = True
      ret.safetyConfigs[-1].safetyParam |= GwmSafetyFlags.LONG_CONTROL.value

      ret.longitudinalActuatorDelay = 0.25
      ret.stopAccel = -0.75
      ret.longitudinalTuning.kiBP = [0.]
      ret.longitudinalTuning.kiV = [0.4]

    return ret
