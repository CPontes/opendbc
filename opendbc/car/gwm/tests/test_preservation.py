import random
import math
import unittest
from types import SimpleNamespace

from opendbc.can.packer import CANPacker
from opendbc.can.parser import CANParser
from opendbc.car import structs
from opendbc.car.gwm import gwmcan
from opendbc.car.gwm.interface import CarInterface
from opendbc.car.gwm.values import CAR

DBC = 'gwm_haval_h6_mk3_generated'


class TestGwmPreservation(unittest.TestCase):
  def setUp(self):
    self.packer = CANPacker(DBC)
    self.bus = SimpleNamespace(main=0, camera=2)

  def assert_unchanged(self, original, output, allowed):
    self.assertEqual(len(output), len(original))
    for i, (before, after) in enumerate(zip(original, output, strict=True)):
      self.assertEqual((before ^ after) & ~allowed.get(i, 0), 0, f'Unexpected change in byte {i}')

  def test_steering_preserves_every_unowned_bit(self):
    rng = random.Random(17)
    for steer in [-253, -1, 0, 1, 253]:
      for request in [False, True]:
        for _ in range(20):
          original = rng.randbytes(64)
          addr, output, bus = gwmcan.create_steer_command(self.packer, self.bus, original, steer, request)
          self.assertEqual((addr, bus), (0x12B, 0))
          self.assert_unchanged(original, output, {8:0xff, 9:0xff, 10:0xfc, 12:0xff, 13:0xe0, 15:0x2f})
          self.assertEqual(output[16:24], original[16:24])
          self.assertEqual(output[15] & 0xf, ((original[15] & 0xf) + 1) % 15)
          self.assertEqual(bool(output[15] & 0x20), request)
          raw_torque = ((output[12] & 0x7f) << 3) | (output[13] >> 5)
          self.assertEqual((raw_torque - 1024 if raw_torque & 512 else raw_torque) + 1, steer)
          self.assertEqual(output[8], gwmcan.checksum(output[9:16], 0x9b))
          self.assertEqual((output[9] >> 2) & 0x1f, gwmcan.gwm_basic_chksum_for_0x12B(output))

  def test_uint64_low_bits_survive_parser_state_and_helper(self):
    ci = CarInterface(CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6))
    for value in [2**53+1, 2**63+1, 2**64-1]:
      original = bytearray(range(64))
      original[16:24] = value.to_bytes(8, 'big')
      ci.update([[1_000_000, [(0x12b, bytes(original), 2)]]])
      self.assertEqual(ci.CS.camera_stock_data, original)
      output = gwmcan.create_steer_command(self.packer, self.bus, ci.CS.camera_stock_data, 0, False)[1]
      self.assertEqual(int.from_bytes(output[16:24], 'big'), value)

  def test_eps_fault_flags_counters_and_unknown_bits(self):
    rng = random.Random(18)
    for flags in range(16):
      for torque in [-1024, -253, 0, 253, 1023]:
        original = bytearray(rng.randbytes(64))
        original[15] = (flags << 4) | 11
        addr, output, bus = gwmcan.create_wheel_touch(self.packer, self.bus, bytes(original), torque)
        self.assertEqual((addr, bus), (0x147, 2))
        self.assert_unchanged(original, output, {8:0xff, 9:0x7f, 10:0xf0})
        self.assertEqual(output[15], original[15])
        raw_torque = ((output[9] & 0x7f) << 4) | (output[10] >> 4)
        self.assertEqual(raw_torque - 2048 if raw_torque & 1024 else raw_torque, torque)
        self.assertEqual(output[:8], original[:8])
        self.assertEqual(output[8], gwmcan.checksum(output[9:16], 0x61))

  def test_acc_preserves_opaque_block_and_counters(self):
    rng = random.Random(19)
    for active in [False, True]:
      for standstill in [False, True]:
        for accel in [-3.5, -0.01, 0, 0.5, 2]:
          braking = accel < 0
          original = rng.randbytes(64)
          addr, output, bus = gwmcan.create_longitudinal_command(self.packer, self.bus, original, accel, braking, active, standstill)
          self.assertEqual((addr, bus), (0x143, 0))
          self.assert_unchanged(original, output, {8:0xff, 9:0x1f, 12:0x20, 13:0xff, 24:0xff, 27:0xff, 28:0xff, 29:0x80})
          self.assertEqual(output[16:24], original[16:24])
          expected_brake = math.floor(accel / 3.5 * 66 - 41 + 0.5) if active and braking else 0
          expected_gas = {0: 60, 0.5: 725, 2: 3000}.get(accel, 0) if active and not braking else 0
          self.assertEqual(output[13] - 181, expected_brake)
          self.assertEqual(((output[27] & 31) << 8 | output[28]) - 192, expected_gas)
          self.assertEqual(output[8], gwmcan.checksum(output[9:16], 0xef))
          self.assertEqual(output[24], gwmcan.checksum(output[25:32], 0x87))

  def test_hud_and_buttons_preserve_unknown_bits(self):
    rng = random.Random(20)
    for active in [False, True]:
      original = rng.randbytes(64)
      output = gwmcan.create_hud_command(self.packer, self.bus, original, active)[1]
      self.assert_unchanged(original, output, {16:0xff, 17:0x38 if active else 0})
      self.assertEqual((output[17] >> 3) & 7, 5 if active else (original[17] >> 3) & 7)
      self.assertEqual(output[16], gwmcan.checksum(output[17:24], 0x66))
      for counter in range(15):
        original = bytearray(rng.randbytes(8))
        original[7] = (original[7] & 0xf0) | counter
        output = gwmcan.create_cancel_command(self.packer, self.bus, bytes(original))[1]
        self.assert_unchanged(original, output, {0:0xff, 5:0xf0, 6:0xc, 7:0xf})
        self.assertEqual(output[7] & 0xf, (counter + 1) % 15)
        self.assertEqual(output[5] & 0xf0, 0x40)
        self.assertEqual(output[6] & 0xc, 0)
        self.assertEqual(output[0], gwmcan.checksum(output[1:8], 0x2d))

  def test_no_transmission_without_stock_frames(self):
    cp = CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6)
    cp.openpilotLongitudinalControl = True
    ci = CarInterface(cp)
    cc = structs.CarControl()
    cc.enabled = cc.latActive = cc.longActive = cc.cruiseControl.cancel = True
    cc.actuators.torque = 1
    for _ in range(20):
      actuators, frames = ci.CC.update(cc.as_reader(), ci.CS, 0)
      self.assertEqual(frames, [])
      self.assertEqual(actuators.torqueOutputCan, 0)

  def test_raw_frames_reach_all_controller_outputs(self):
    cp = CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6)
    cp.openpilotLongitudinalControl = True
    ci = CarInterface(cp)
    rng = random.Random(21)
    specs = [(0x12b, 64, 2), (0x147, 64, 0), (0x143, 64, 2), (0x23d, 64, 2), (0xa1, 8, 0)]
    original = {addr:rng.randbytes(size) for addr, size, _ in specs}
    ci.update([[1_000_000, [(addr, original[addr], bus) for addr, _, bus in specs]]])
    ci.CS.acc_driving = True
    ci.CS.out.standstill = False
    cc = structs.CarControl()
    cc.cruiseControl.cancel = True
    _, frames = ci.CC.update(cc.as_reader(), ci.CS, 0)
    self.assertEqual({addr for addr, _, _ in frames}, set(original))
    for addr, output, _ in frames:
      if addr in [0x12b, 0x143]:
        self.assertEqual(output[16:24], original[addr][16:24])
      if addr == 0x147:
        self.assertEqual(output[15], original[addr][15])

  def test_truncated_stock_frame_is_not_forwarded(self):
    ci = CarInterface(CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6))
    ci.update([[1_000_000, [(0x12b, bytes(64), 2)]]])
    self.assertIsNotNone(ci.CS.camera_stock_data)
    ci.update([[2_000_000, [(0x12b, bytes(16), 2)]]])
    self.assertIsNone(ci.CS.camera_stock_data)
    _, frames = ci.CC.update(structs.CarControl().as_reader(), ci.CS, 0)
    self.assertNotIn(0x12b, [addr for addr, _, _ in frames])
    ci.update([[3_000_000, [(0x12b, bytes(64), 2)]]])
    self.assertIsNotNone(ci.CS.camera_stock_data)

  def test_raw_parser_keeps_last_accepted_frame(self):
    parser = CANParser('fca_giorgio', [('EPS_1', 0)], 0)
    self.assertNotIn('EPS_1', parser.vl_raw)
    first = bytes.fromhex('175197cc00df')
    good = bytearray.fromhex('175197c901a3')
    bad = bytes.fromhex('175197c901a2')
    parser.update([[1_000_000, [(0xde, first, 0)]]])
    parser.update([[2_000_000, [(0xde, good, 0), (0xde, bad, 0), (0xde, first, 1)]]])
    self.assertEqual(parser.vl_raw['EPS_1'], good)
    self.assertIs(parser.vl_raw['EPS_1'], parser.vl_raw[0xde])
    good[0] = 0
    self.assertEqual(parser.vl_raw['EPS_1'], bytes.fromhex('175197c901a3'))
    parser.update([])
    self.assertEqual(parser.vl_raw[0xde], bytes.fromhex('175197c901a3'))


if __name__ == '__main__':
  unittest.main()
