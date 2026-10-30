from dataclasses import dataclass, field
from enum import IntFlag

from opendbc.car.structs import CarParams
from opendbc.car import Bus, CarSpecs, DbcDict, PlatformConfig, Platforms
from opendbc.car.docs_definitions import CarDocs, CarHarness, CarParts
from opendbc.car.fw_query_definitions import FwQueryConfig, Request, StdQueries

Ecu = CarParams.Ecu


class CarControllerParams:
  STEER_STEP = 2
  STEER_MAX = 253
  ACCEL_MAX = 2
  ACCEL_MIN = -3.5

  def __init__(self, CP: CarParams):
    self.STEER_DELTA_UP = 4
    self.STEER_DELTA_DOWN = 6
    self.STEER_ERROR_MAX = 70


class GwmSafetyFlags(IntFlag):
  LONG_CONTROL = 1


@dataclass
class GWMCarDocs(CarDocs):
  package: str = "Adaptive Cruise Control (ACC) & Lane Assist"
  car_parts: CarParts = field(default_factory=CarParts.common([CarHarness.custom]))


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


# Placeholder, FW Query will be added in separate PR
FW_QUERY_CONFIG = FwQueryConfig(
  fw_version_regex=br"\d{9}",
  requests=[
    Request(
      [StdQueries.TESTER_PRESENT_REQUEST, StdQueries.UDS_VERSION_REQUEST],
      [StdQueries.TESTER_PRESENT_RESPONSE, StdQueries.UDS_VERSION_RESPONSE],
      bus=0,
    ),
  ],
)

DBC = CAR.create_dbc_map()
