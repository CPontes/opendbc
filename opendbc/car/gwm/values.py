from dataclasses import dataclass, field
from enum import IntFlag

from opendbc.car.structs import CarParams
from opendbc.car import Bus, CarSpecs, DbcDict, PlatformConfig, Platforms, uds
from opendbc.car.docs_definitions import CarDocs, CarHarness, CarParts
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, p16

Ecu = CarParams.Ecu


class CarControllerParams:
  STEER_STEP = 2
  STEER_MAX = 253
  ACCEL_MAX = 2
  ACCEL_MIN = -3.5

  def __init__(self, CP: CarParams):
    self.STEER_DELTA_UP = 4
    self.STEER_DELTA_DOWN = 6
    self.STEER_ERROR_MAX = 80


class GwmFlags(IntFlag):
  # Hexa: a light push against openpilot fades its torque out instead of cutting it. Not a panda flag: the panda takes
  # any fade down with the request on
  SOFT_HANDOFF = 1
  # Hexa: the driver only counts as against openpilot when against both the request and the torque going out. In a
  # reversal the request crosses zero while the applied torque still points the old way: a driver helping it isn't
  # against openpilot, and cutting it then dropped the help at once. Not a panda flag: the panda limits the torque by the
  # EPS, not by the driver
  OPPOSE_BOTH = 2
  # Hexa: back from a touch of the gas at a stop of ours, with the car never moving, the stock hold at once and no frame of gas.
  # Not a panda flag: the panda takes the hold with openpilot in control and the pedal off
  GAS_TOUCH_HOLD = 4
  # Hexa: at a stop of ours, a touch of the driver's gas (short and light) doesn't take the hold away: our hold stays with no brake
  # while the pedal is down, and openpilot comes back at the stock hold, as with GAS_TOUCH_HOLD. Longer or firmer, as before. Not a
  # panda flag: the panda takes the hold with no brake from a stopped car
  GAS_TAP_HOLD = 8


class GwmSafetyFlags(IntFlag):
  LONG_CONTROL = 1
  ALWAYS_ON_LATERAL = 2  # brake pauses longitudinal only, steering stays engaged until cancel
  AUTO_RESUME = 4  # openpilot may pull the stalk once per stop
  ACC_MINUS = 8  # openpilot may press the minus button of the ACC
  ACC_PLUS = 16  # and the plus, to the road's limit: with the ACC driving, below the panda's ceiling (CRUISE-01)


@dataclass
class GWMCarDocs(CarDocs):
  package: str = "Adaptive Cruise Control (ACC) & Lane Assist"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.gwm]))


@dataclass
class GWMPlatformConfig(PlatformConfig):
  dbc_dict: DbcDict = field(default_factory=lambda: {
    Bus.pt: 'gwm_haval_h6_mk3_generated',
  })


class CAR(Platforms):
  GWM_HAVAL_H6 = GWMPlatformConfig(
    [
      GWMCarDocs("GWM Haval H6 PHEV 2023-24"),
      GWMCarDocs("GWM Haval H6 GT PHEV 2023-24"),
    ],
    CarSpecs(wheelbase=2.738, steerRatio=17.416),
  )


GREATWALLMOTORS_VERSION_REQUEST_MULTI = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER]) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_SPARE_PART_NUMBER) + \
  p16(uds.DATA_IDENTIFIER_TYPE.VEHICLE_MANUFACTURER_ECU_SOFTWARE_VERSION_NUMBER) + \
  p16(uds.DATA_IDENTIFIER_TYPE.APPLICATION_DATA_IDENTIFICATION)
GREATWALLMOTORS_VERSION_RESPONSE = bytes([uds.SERVICE_TYPE.READ_DATA_BY_IDENTIFIER + 0x40])

GREATWALLMOTORS_RX_OFFSET = 0x6a

FW_QUERY_CONFIG = FwQueryConfig(
  fw_version_regex=br"\xf1\x87[\x00-\xff]+\xf1\x89[\x00-\xff]+",
  requests=[request for bus, obd_multiplexing in [(1, True), (1, False), (0, False)] for request in [
    Request(
      [GREATWALLMOTORS_VERSION_REQUEST_MULTI],
      [GREATWALLMOTORS_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.engine],
      rx_offset=GREATWALLMOTORS_RX_OFFSET,
      bus=bus,
      obd_multiplexing=obd_multiplexing,
    ),
    Request(
      [GREATWALLMOTORS_VERSION_REQUEST_MULTI],
      [GREATWALLMOTORS_VERSION_RESPONSE],
      whitelist_ecus=[Ecu.engine],
      bus=bus,
      obd_multiplexing=obd_multiplexing,
    ),
  ]],
)

DBC = CAR.create_dbc_map()
