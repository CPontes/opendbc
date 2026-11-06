"""openpilot longitudinal: whatever the plan asks and the driver does, every ACC command (0x143) carcontroller builds is one the
panda lets out. A refused frame leaves the PCM with no ACC command at all. The panda engages from the frames it reads (lever,
brake, gas pedal), and openpilot follows the same rules."""
import random
from collections import Counter
from types import SimpleNamespace

import pytest

from opendbc.can.packer import CANPacker
from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.gwm.gwmcan import checksum
from opendbc.car.gwm.interface import CarInterface
from opendbc.car.gwm.values import CAR, DBC, GwmSafetyFlags
from opendbc.car.structs import CarParams
from opendbc.safety.tests.libsafety import libsafety_py

LongCtrlState = structs.CarControl.Actuators.LongControlState
LONG = GwmSafetyFlags.LONG_CONTROL.value
ALWAYS_ON = GwmSafetyFlags.ALWAYS_ON_LATERAL.value


class Bench:
  def __init__(self, param):
    CP = CarInterface.get_params(CAR.GWM_HAVAL_H6, gen_empty_fingerprint(), [], True, False, False)
    CP.safetyConfigs[-1].safetyParam = param
    self.cc = CarInterface(CP).CC
    self.packer = CANPacker(DBC[CAR.GWM_HAVAL_H6][Bus.pt])
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, param)
    self.safety.init_tests()
    self.counters = Counter()
    self.t = 100_000_000

  def rx(self, name, values, crc_at=None, xor=None, counter=None):
    if counter is not None:
      self.counters[name] = (self.counters[name] + 1) % 15
      values = {**values, counter: self.counters[name]}
    addr, dat, bus = self.packer.make_can_msg(name, 0, values)
    dat = bytearray(dat)
    if crc_at is not None:
      dat[crc_at] = checksum(dat[crc_at + 1:crc_at + 8], xor)
    assert self.safety.safety_rx_hook(libsafety_py.make_CANPacket(addr, 0, bytes(dat))), name

  def tx(self, addr, bus, dat):
    return self.safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, bytes(dat)))


@pytest.mark.parametrize("param", [LONG, LONG | ALWAYS_ON])
@pytest.mark.parametrize("seed", range(8))
def test_every_acc_frame_of_the_controller_passes_the_panda(param, seed):
  rng = random.Random(seed)
  b = Bench(param)
  always_on = bool(param & ALWAYS_ON)
  dbc = b.packer.dbc
  CS = SimpleNamespace(out=structs.CarState(), camera_stock_data=None, eps_stock_data=None, hud_stock_data=None,
                       longitudinal_stock_data=bytes(dbc.name_to_msg["ACC_CMD"].size), steer_and_ap_stalk_msg=None,
                       acc_waiting=False, acc_driving=False, stalk_idle=True, rolling_back=False)
  CS.out.gearShifter = structs.CarState.GearShifter.drive
  speed, target, accel = 10., 10., 0.
  lever = brake = gas = False
  enabled = paused = False
  acc_left = 0  # frames the stock ACC still drives: it lets go about 0.1 s after the pedal
  modes = Counter()
  for i in range(6000):
    b.t += 10_000
    b.safety.set_timer(b.t)
    # a drive with stops, at up to 3.5 m/s^2 like the car; the driver now and then on the lever, the brake or the gas
    if rng.random() < 0.005:
      target = 0. if rng.random() < 0.5 else rng.uniform(0., 25.)
    speed = max(0., speed + min(max(target - speed, -0.035), 0.02))
    accel = float(min(max(accel + rng.uniform(-0.4, 0.4), -3.5), 2.))
    pull = (not lever) and rng.random() < 0.004
    lever = pull or (lever and rng.random() < 0.8)
    brake = (brake and rng.random() < 0.97) or ((not brake) and rng.random() < 0.003)
    gas = (gas and rng.random() < 0.95) or ((not gas) and not brake and rng.random() < 0.002)

    wheels = {f"{w}_WHEEL_SPEED": speed * 3.6 for w in ("FRONT_LEFT", "FRONT_RIGHT", "REAR_LEFT", "REAR_RIGHT")}
    b.rx("WHEEL_SPEEDS", wheels, crc_at=0, xor=0x7F, counter="COUNTER")
    if i % 2 == 0:
      b.rx("BRAKE2", {"PEDAL_BRAKE_PRESSED": int(brake)}, crc_at=0, xor=0xEE, counter="COUNTER")
      b.rx("CAR_OVERALL_SIGNALS2", {"GAS_POSITION": 10 if gas else 0}, crc_at=8, xor=0x95, counter="COUNTER2")
    b.rx("STEER_AND_AP_STALK", {"AP_ENABLE_COMMAND": int(lever)}, crc_at=0, xor=0x2D, counter="COUNTER")

    # openpilot, by the same rules: a pull engages; the brake ends it, or with always-on lateral pauses longitudinal to the next pull
    if pull and not brake:
      enabled, paused = True, False
    if brake and enabled:
      if always_on:
        paused = True
      else:
        enabled = False
    stock_mode = rng.choice((0, 4, 8, 12, 13, 20, 25))  # whatever the camera's own ACC is doing
    CS.longitudinal_stock_data = b.packer.make_can_msg("ACC_CMD", 2, {"BRAKE_OR_GAS_REQ": stock_mode})[1]
    CS.out.vEgo = CS.out.vEgoRaw = speed
    CS.out.standstill = speed < 1e-3
    CS.out.brakePressed, CS.out.gasPressed = brake, gas
    acc_left = 10 if (enabled and not paused and not brake) else max(acc_left - 1, 0)
    CS.acc_driving = acc_left > 0 and not CS.out.standstill
    CS.rolling_back = speed < 1. and rng.random() < 0.02

    CC = structs.CarControl()
    CC.enabled = enabled
    CC.longActive = enabled and not paused and not gas and not brake
    CC.actuators.accel = accel
    CC.actuators.longControlState = LongCtrlState.stopping if speed < 0.3 and accel < 0 else LongCtrlState.pid
    for addr, dat, bus in b.cc.update(CC.as_reader(), CS, 0)[1]:
      if addr == 0x143:
        modes[dat[9] & 0x1F] += 1
        modes["handoff"] += brake and (dat[9] & 0x1F) in (13, 25) and 181 - dat[13] > 0
        assert b.tx(addr, bus, dat), (i, speed, accel, enabled, paused, brake, gas, dat[9] & 0x1F, 181 - dat[13])
  # the drives went through standby, gas, brake and hold, the pedal (4) and the handoff to it: our brake kept under the pedal
  assert all(modes[m] > 10 for m in (4, 8, 12, 13, 25, "handoff")), modes
