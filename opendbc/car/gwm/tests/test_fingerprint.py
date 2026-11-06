"""The car is recognized by its firmware alone, with nothing forced, and an unknown firmware picks no car at all."""
from opendbc.car.fw_versions import match_fw_to_car
from opendbc.car.gwm.values import CAR
from opendbc.car.structs import CarParams

# the engine ECU as a 2024 H6 GT PHEV answered, read the same in every route we logged (both brands' queries reach it)
ENGINE_FW = b'\xf1\x873612100XEC56000\xf1\x89S013A01XKN17002'


def car_fw(fw, brands=("volkswagen", "gwm")):
  return [CarParams.CarFw(ecu=CarParams.Ecu.engine, fwVersion=fw, address=0x7e0, subAddress=0, brand=b, obdMultiplexing=True) for b in brands]


def test_the_logged_firmware_picks_the_haval_exactly():
  exact, matches = match_fw_to_car(car_fw(ENGINE_FW), "", allow_fuzzy=False, log=False)
  assert exact and matches == {CAR.GWM_HAVAL_H6}


def test_an_unknown_firmware_picks_no_car():
  _, matches = match_fw_to_car(car_fw(b'\xf1\x873612100XEC99999\xf1\x89S013A01XKN99999'), "", log=False)
  assert matches == set()
