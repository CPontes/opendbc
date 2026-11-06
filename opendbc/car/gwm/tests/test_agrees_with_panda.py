"""CarState and the panda keep separate engagement state from the same frames. When CarState says engaged and the panda
doesn't, every torque command is blocked with openpilot showing engaged: it happened in 2 of 114 pulls in the logs."""
import random

import pytest

from opendbc.can.packer import CANPacker
from opendbc.car import Bus
from opendbc.car.gwm.interface import CarInterface
from opendbc.car.gwm.values import CAR, DBC
from opendbc.car.gwm.gwmcan import checksum
from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py

STALK = "STEER_AND_AP_STALK"


class Both:
  def __init__(self, param):
    CP = CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6)
    CP.safetyConfigs[-1].safetyParam |= param
    self.CI = CarInterface(CP)
    self.CI.update([])
    self.packer = CANPacker(DBC[CAR.GWM_HAVAL_H6][Bus.pt])
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, param)
    self.safety.init_tests()
    self.counter = 0
    self.brake_counter = 0
    self.nanos = 0

  def step(self, lever, cancel, brake, brake_first, acc=None):
    self.counter = (self.counter + 1) % 15
    _, dat, _ = self.packer.make_can_msg(STALK, 0, {"AP_ENABLE_COMMAND": lever, "AP_CANCEL_COMMAND": cancel, "COUNTER": self.counter})
    dat = bytes([checksum(dat[1:8], 0x2D)]) + dat[1:]
    frames = [(0xA1, dat, 0)]
    if brake is not None:
      self.brake_counter = (self.brake_counter + 1) % 15
      _, bdat, _ = self.packer.make_can_msg("BRAKE2", 0, {"PEDAL_BRAKE_PRESSED": brake, "COUNTER": self.brake_counter})
      frames.append((0x120, bytes([checksum(bdat[1:8], 0xEE)]) + bdat[1:], 0))
    if acc is not None:
      frames.append(self.packer.make_can_msg("ACC", 2, {"CRUISE_STATE_2": acc}))
    # the car's frames land on the panda one at a time, CarState sees them as one batch
    for addr, data, bus in (reversed(frames) if brake_first else frames):
      assert self.safety.safety_rx_hook(libsafety_py.make_CANPacket(addr, bus, data))
    self.nanos += 10_000_000
    return self.CI.update([(self.nanos, frames)]).cruiseState.enabled, self.safety.get_controls_allowed()


@pytest.mark.parametrize("seed", range(20))
def test_never_engaged_with_the_panda_refusing(seed):
  random_drive(0, seed)


def random_drive(param, seed):
  rng = random.Random(seed)
  both = Both(param)
  lever = cancel = brake = 0
  acc = 3
  engagements = dropped = 0
  out = prev = False
  for i in range(3000):
    lever ^= rng.random() < 0.15
    cancel ^= rng.random() < (0.3 if cancel else 0.02)
    # the stalk comes at 100 Hz, the brake at 50 Hz
    if i % 2 == 0:
      brake ^= rng.random() < 0.2
    # openpilot drops out on its own: the heartbeat takes controls away and it asks to cancel
    if prev and not out and rng.random() < 0.05:
      out = True
      dropped += 1
      both.safety.set_controls_allowed(False)
    both.CI.CS.cancel_requested = out and prev
    # the ACC at 10 Hz, now and then off for a while: dropped out, or a pull it never took
    if i % 10 == 0 and rng.random() < 0.1:
      acc = 1 if acc == 3 else 3
    enabled, allowed = both.step(int(lever), int(cancel), None if i % 2 else int(brake), rng.random() < 0.5,
                                 acc if i % 10 == 0 else None)
    out = out and enabled
    assert allowed or not enabled or out
    engagements += enabled and not prev
    prev = enabled
  # the sequence really went through engagements
  assert engagements > 5 and dropped > 0
