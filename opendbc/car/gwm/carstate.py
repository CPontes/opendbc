from opendbc.car import Bus, CanBusBase, structs
from opendbc.can.parser import CANParser
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarStateBase
from opendbc.car.gwm.values import DBC, GwmSafetyFlags

ButtonType = structs.CarState.ButtonEvent.Type
GearShifter = structs.CarState.GearShifter
TransmissionType = structs.CarParams.TransmissionType

STALK_BUTTONS = ("AP_CANCEL_COMMAND", "AP_ENABLE_COMMAND", "AP_INCREASE_SPEED_COMMAND", "AP_DECREASE_SPEED_COMMAND",
                 "AP_INCREASE_DISTANCE_COMMAND", "AP_REDUCE_DISTANCE_COMMAND")


def get_stock_data(parser, name):
  # Never synthesize a stock frame from zero-initialized signals or a truncated payload.
  data = parser.vl_raw.get(name)
  return data if data is not None and len(data) == parser.dbc.name_to_msg[name].size else None


class CarState(CarStateBase):
  def __init__(self, CP):
    super().__init__(CP)
    self.steer_and_ap_stalk_msg: bytes | None = None
    self.eps_stock_data: bytes | None = None
    self.camera_stock_data: bytes | None = None
    self.longitudinal_stock_data: bytes | None = None
    self.hud_stock_data: bytes | None = None

    self.is_activation_lever_pulled = False
    self.prev_activation_lever_pulled = False
    self.lever_pull_valid = False
    self.cancel_requested = False
    self.main_on = False
    self.acc_waiting = False
    self.acc_driving = False
    self.stalk_idle = False
    self.steer_fault_temporary_counter = 0

  def update(self, can_parsers) -> structs.CarState:
    cp = can_parsers[Bus.main]
    cp_cam = can_parsers[Bus.cam]
    ret = structs.CarState()

    self.steer_and_ap_stalk_msg = get_stock_data(cp, "STEER_AND_AP_STALK")
    self.eps_stock_data = get_stock_data(cp, "RX_STEER_RELATED")
    self.camera_stock_data = get_stock_data(cp_cam, "STEER_CMD")
    self.longitudinal_stock_data = get_stock_data(cp_cam, "ACC_CMD")
    self.hud_stock_data = get_stock_data(cp_cam, "LATERAL_STATE")

    self.parse_wheel_speeds(ret,
      cp.vl["WHEEL_SPEEDS"]["FRONT_LEFT_WHEEL_SPEED"],
      cp.vl["WHEEL_SPEEDS"]["FRONT_RIGHT_WHEEL_SPEED"],
      cp.vl["WHEEL_SPEEDS"]["REAR_LEFT_WHEEL_SPEED"],
      cp.vl["WHEEL_SPEEDS"]["REAR_RIGHT_WHEEL_SPEED"]
    )

    ret.accFaulted = bool(cp_cam.vl["ACC"]["CRUISE_STATE_2"] == 0)
    self.acc_waiting = cp_cam.vl["ACC"]["CRUISE_STATE_2"] == 2
    self.acc_driving = cp_cam.vl["ACC"]["CRUISE_STATE_2"] == 3
    ret.cruiseState.driving = self.acc_driving
    ret.camLeadDistance = cp_cam.vl["UNDEFINED_INTERESTING_SIGNALS"]["LEAD_DISTANCE"]
    self.stalk_idle = not any(cp.vl["STEER_AND_AP_STALK"][button] for button in STALK_BUTTONS)
    ret.cruiseState.speed = cp_cam.vl["ACC"]["ACC_SPEED_SELECTION"]  * CV.KPH_TO_MS
    ret.cruiseState.speedCluster = ret.cruiseState.speed
    if not self.CP.openpilotLongitudinalControl:
      ret.cruiseState.speed = -1

    ret.standstill = abs(ret.vEgoRaw) < 1e-3
    ret.gasPressed = cp.vl["CAR_OVERALL_SIGNALS2"]["GAS_POSITION"] > 0
    ret.brakePressed = cp.vl["BRAKE2"]["PEDAL_BRAKE_PRESSED"] != 0
    ret.parkingBrake = cp.vl["CAR_OVERALL_SIGNALS"]["DRIVE_MODE"] == 0

    ret.gearShifter = GearShifter.drive if int(cp.vl["CAR_OVERALL_SIGNALS"]["DRIVE_MODE"]) == 1 else \
                      GearShifter.neutral if int(cp.vl["CAR_OVERALL_SIGNALS"]["DRIVE_MODE"]) == 2 else \
                      GearShifter.reverse if int(cp.vl["CAR_OVERALL_SIGNALS"]["DRIVE_MODE"]) == 3 else \
                      GearShifter.park

    ret.steeringAngleDeg = cp.vl["STEER_AND_AP_STALK"]["STEERING_ANGLE"] * (-1 if cp.vl["STEER_AND_AP_STALK"]["STEERING_DIRECTION"] else 1)
    ret.steeringRateDeg = cp.vl["STEER_AND_AP_STALK"]["STEERING_RATE"] * (-1 if (cp.vl["STEER_AND_AP_STALK"]["RATE_DIRECTION"] > 0) else 1)

    # Since loopback was throwing a CanError even though the logic expected cp_loopback.vl_vall > 0, I moved the detection to interface.py against lat_active.
    ret.steerFaultTemporary = False # (bool(cp_loopback.vl["STEER_CMD"]["STEER_REQUEST"]) and bool(cp.vl["RX_STEER_RELATED"]["A_RX_STEER_REQUESTED"] != 1))
    self.steer_fault_temporary_counter = (self.steer_fault_temporary_counter + 1) if (cp.vl["RX_STEER_RELATED"]["EPS_FAULT_PERMANENT"] == 1) else 0
    ret.steerFaultTemporary |= self.steer_fault_temporary_counter > 100
    ret.steerFaultPermanent = False #self.steer_fault_permanent_counter > 500

    ret.steeringTorque = cp.vl["RX_STEER_RELATED"]["B_RX_DRIVER_TORQUE"]
    ret.steeringTorqueEps = cp.vl["RX_STEER_RELATED"]["B_RX_EPS_TORQUE"]
    ret.steeringPressed = abs(ret.steeringTorque) > 50

    ret.doorOpen = any([cp.vl["DOOR_DRIVER"]["DOOR_REAR_RIGHT_OPEN"],
                        cp.vl["DOOR_DRIVER"]["DOOR_FRONT_RIGHT_OPEN"],
                        cp.vl["DOOR_DRIVER"]["DOOR_REAR_LEFT_OPEN"],
                        cp.vl["DOOR_DRIVER"]["DOOR_DRIVER_OPEN"]])
    ret.seatbeltUnlatched = bool(cp.vl["SEATBELT"]["SEAT_BELT_DRIVER_STATE"])
    ret.leftBlinker, ret.rightBlinker = self.update_blinker_from_lamp(50, cp.vl["LIGHTS"]["LEFT_TURN_SIGNAL"],
                                                                      cp.vl["LIGHTS"]["RIGHT_TURN_SIGNAL"])
    ret.leftBlindspot = bool(cp.vl["RADAR_BEHIND"]["BSM_LEFT"] > 0)
    ret.rightBlindspot = bool(cp.vl["RADAR_BEHIND"]["BSM_RIGHT"] > 0)

    # with always-on lateral the brake only pauses longitudinal (handled in selfdrived), it doesn't disengage
    always_on_lateral = bool(self.CP.safetyConfigs[-1].safetyParam & GwmSafetyFlags.ALWAYS_ON_LATERAL)
    # the last frame too: the panda reads the stalk mid-update, it may not have seen the brake released yet
    brake_pressed = ret.brakePressed or self.out.brakePressed
    disengage = cp.vl["STEER_AND_AP_STALK"]["AP_CANCEL_COMMAND"] or (brake_pressed and not always_on_lateral)
    if disengage:
      self.main_on = False
    self.is_activation_lever_pulled = bool(cp.vl["STEER_AND_AP_STALK"]["AP_ENABLE_COMMAND"])
    lever_pressed = self.is_activation_lever_pulled and not self.prev_activation_lever_pulled
    lever_released = not self.is_activation_lever_pulled and self.prev_activation_lever_pulled
    # The panda engages on the pull and drops it on cancel or brake, this engages on the release. A pull that
    # fails the panda's rule can't count here, or openpilot shows engaged with every torque command blocked.
    self.lever_pull_valid = (self.lever_pull_valid or lever_pressed) and not disengage
    # openpilot dropped out on its own (fault, gear, monitoring) and nothing but cancel or brake clears main_on
    if lever_pressed and self.cancel_requested:
      self.main_on = False
    if lever_released and self.lever_pull_valid:
      if self.main_on and always_on_lateral:
        # lever pulled while engaged: resume longitudinal after a brake pause
        ret.buttonEvents = [structs.CarState.ButtonEvent(pressed=False, type=ButtonType.resumeCruise)]
      self.main_on = True
    self.prev_activation_lever_pulled = self.is_activation_lever_pulled

    ret.cruiseState.available = self.main_on
    ret.cruiseState.enabled = self.main_on

    return ret

  @staticmethod
  def get_can_parsers(CP):
    # Compute bus offset from number of safetyConfigs so multipanda setups
    # (internal + external pandas) map DBCs to the correct physical bus.
    can_base = CanBusBase(CP, None)
    main_bus = can_base.offset
    adas_bus = can_base.offset + 1
    cam_bus = can_base.offset + 2

    return {
      Bus.main: CANParser(DBC[CP.carFingerprint][Bus.pt], [("STEER_AND_AP_STALK", 0), ("RX_STEER_RELATED", 0)], main_bus),
      Bus.adas: CANParser(DBC[CP.carFingerprint][Bus.pt], [], adas_bus),
      Bus.cam: CANParser(DBC[CP.carFingerprint][Bus.pt], [("STEER_CMD", 0), ("ACC_CMD", 0), ("LATERAL_STATE", 0),
                                                           ("UNDEFINED_INTERESTING_SIGNALS", float("nan"))], cam_bus),
    }
