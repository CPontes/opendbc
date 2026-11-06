#!/usr/bin/env python3
import unittest
from types import SimpleNamespace

from opendbc.can.dbc import DBC
from opendbc.car.structs import CarParams
import opendbc.safety.tests.common as common
from opendbc.safety.tests.libsafety import libsafety_py
from opendbc.safety.tests.common import CANPackerSafety
from opendbc.car.gwm import gwmcan
from opendbc.car.gwm.gwmcan import checksum as _checksum


opendbc = "gwm_haval_h6_mk3_generated"
dbc = DBC(opendbc)

STALK_BUTTONS = ("AP_ENABLE_COMMAND", "AP_INCREASE_SPEED_COMMAND", "AP_DECREASE_SPEED_COMMAND",
                 "AP_INCREASE_DISTANCE_COMMAND", "AP_REDUCE_DISTANCE_COMMAND")


def get_signal_range(signal):
  if signal.is_signed:
    # For signed: -(2^(bits-1)) a (2^(bits-1)-1)
    max_val = (2 ** (signal.size - 1)) - 1
    min_val = -(2 ** (signal.size - 1))
  else:
    # For unsigned: 0 a (2^bits - 1)
    max_val = (2 ** signal.size) - 1
    min_val = 0

  # Apply factor and offset
  physical_max = max_val * signal.factor + signal.offset
  physical_min = min_val * signal.factor + signal.offset

  return physical_min, physical_max


def checksum(msg):
  addr, dat, bus = msg
  ret = bytearray(dat)

  if addr == 0xA1: # STEER_AND_AP_STALK
    ret[0] = _checksum(ret[1:], 0x2D)
  elif addr == 0x13B: # WHEEL_SPEEDS
    ret[0] = _checksum(ret[1:8], 0x7F)
  elif addr == 0x60: # CAR_OVERALL_SIGNALS2: the gas pedal is in the second block
    ret[8] = _checksum(ret[9:16], 0x95)
  elif addr == 0x120: # BRAKE2
    ret[0] = _checksum(ret[1:8], 0xEE)
  elif addr == 0x147: # RX_STEER_RELATED: the EPS torque is in the second block
    ret[8] = _checksum(ret[9:16], 0x61)

  return addr, ret, bus


class TestGwmSafety(common.CarSafetyTest, common.MotorTorqueSteeringSafetyTest):
  # stock ACC: openpilot steers, cancels and tells the camera the driver is holding the wheel. The camera's ACC command
  # (0x143) is forwarded untouched
  TX_MSGS = [[0x12B, 0], [0x23D, 0], [0x147, 2], [0xA1, 2]]  # steering, HUD, wheel touch, cancel
  RELAY_MALFUNCTION_ADDRS = {0: (0x12B, 0x23D), 2: (0x147,)}
  FWD_BLACKLISTED_ADDRS = {0: [0x147], 2: [0x12B, 0x23D]}

  MAX_RATE_UP = 4
  MAX_RATE_DOWN = 6
  MAX_TORQUE_LOOKUP = [0], [253]
  MAX_RT_DELTA = 100
  MAX_TORQUE_ERROR = 80
  TORQUE_MEAS_TOLERANCE = 1

  def setUp(self):
    self.packer = CANPackerSafety(opendbc)
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, 0)
    self.safety.init_tests()

  def _user_gas_msg(self, gas):
    # COUNTER2 is the gas block's counter; the packer only counts a signal named COUNTER
    self.gas_counter = (getattr(self, "gas_counter", -1) + 1) % 15
    values = {"GAS_POSITION": gas, "COUNTER2": self.gas_counter}
    return self.packer.make_can_msg_safety("CAR_OVERALL_SIGNALS2", 0, values, fix_checksum=checksum)

  def _user_brake_msg(self, brake):
    self.brake_counter = (getattr(self, "brake_counter", -1) + 1) % 15
    values = {"PEDAL_BRAKE_PRESSED": brake, "COUNTER": self.brake_counter}
    return self.packer.make_can_msg_safety("BRAKE2", 0, values, fix_checksum=checksum)

  def _speed_msg(self, speed):
    values = {f"{pos}_WHEEL_SPEED": speed * 1.0 for pos in ["FRONT_LEFT", "FRONT_RIGHT", "REAR_LEFT", "REAR_RIGHT"]}
    return self.packer.make_can_msg_safety("WHEEL_SPEEDS", 0, values, fix_checksum=checksum)

  def _pcm_status_msg(self, enable):
    values = {"AP_ENABLE_COMMAND": enable, "AP_CANCEL_COMMAND": not enable}
    return self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 0, values, fix_checksum=checksum)

  def test_main_cancel_button(self):
    self.safety.set_controls_allowed(True)
    self._rx(self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 0, {"AP_CANCEL_COMMAND": 1}, fix_checksum=checksum))
    self.assertFalse(self.safety.get_controls_allowed())

  def _stalk_tx(self, **values):
    return self._tx(self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 2, values, fix_checksum=checksum))

  def test_stalk_only_goes_out_to_cancel(self):
    for controls_allowed in (False, True):
      self.safety.set_controls_allowed(controls_allowed)
      self.assertTrue(self._stalk_tx(AP_CANCEL_COMMAND=1))
      self.assertFalse(self._stalk_tx())
      for button in STALK_BUTTONS:
        self.assertFalse(self._stalk_tx(**{button: 1}))
        self.assertFalse(self._stalk_tx(AP_CANCEL_COMMAND=1, **{button: 1}))

  def test_original_stalk_held_only_right_after_our_cancel(self):
    self.safety.set_timer(100000)
    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0xA1))
    self.assertTrue(self._stalk_tx(AP_CANCEL_COMMAND=1))
    self.assertEqual(-1, self.safety.safety_fwd_hook(0, 0xA1))
    self.safety.set_timer(100000 + 14999)
    self.assertEqual(-1, self.safety.safety_fwd_hook(0, 0xA1))
    self.safety.set_timer(100000 + 15000)
    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0xA1))

  def test_a_refused_stalk_frame_holds_nothing(self):
    self.safety.set_timer(100000)
    self.assertFalse(self._stalk_tx(AP_ENABLE_COMMAND=1))
    self.assertFalse(self._stalk_tx(AP_CANCEL_COMMAND=1, AP_ENABLE_COMMAND=1))
    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0xA1))

  def test_a_driver_button_reaches_the_camera_during_our_cancel(self):
    self.safety.set_timer(100000)
    self.assertTrue(self._stalk_tx(AP_CANCEL_COMMAND=1))
    msg = self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 0, {"AP_ENABLE_COMMAND": 1, "COUNTER": 3}, fix_checksum=checksum)
    self._rx(msg)
    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0xA1))

  def test_a_new_session_forgets_our_frame(self):
    self.safety.set_timer(100000)
    self.assertTrue(self._stalk_tx(AP_CANCEL_COMMAND=1))
    self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, self.safety.get_current_safety_param())
    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0xA1))

  def test_cancel_frame_of_the_car_port_passes(self):
    # with the driver on every button, the frame carcontroller builds still has to go out
    stock = self.packer.make_can_msg("STEER_AND_AP_STALK", 0, dict.fromkeys(STALK_BUTTONS, 1))[1]
    addr, dat, bus = gwmcan.create_cancel_command(self.packer, SimpleNamespace(camera=2), stock)
    self.assertTrue(self._tx(libsafety_py.make_CANPacket(addr, bus, dat)))

  def test_new_pull_after_openpilot_dropped_out(self):
    def stalk(enable):
      self._rx(self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 0, {"AP_ENABLE_COMMAND": enable}, fix_checksum=checksum))

    stalk(1)
    stalk(0)
    self.assertTrue(self.safety.get_controls_allowed())
    # no cancel and no brake: only the heartbeat took controls away, so nothing cleared the lever state
    self.safety.set_controls_allowed(False)
    stalk(0)
    self.assertFalse(self.safety.get_controls_allowed())
    stalk(1)
    self.assertTrue(self.safety.get_controls_allowed())

  def test_counters_wrap_after_14(self):
    # measured on the bus: 0xA1 and 0x13B go 13, 14, 0. Looping over the wrap only stays valid if 14 is the last value
    for name in ("STEER_AND_AP_STALK", "WHEEL_SPEEDS"):
      for counter in [13, 14, 0] * 10:
        self.assertTrue(self._rx(self.packer.make_can_msg_safety(name, 0, {"COUNTER": counter}, fix_checksum=checksum)))

  def test_gas_and_brake_checked_like_the_car_sends_them(self):
    # measured on the bus: 45321 brake and 90650 gas frames, the CRC matches all of them and the counters go 0 to 14
    for counter in [13, 14, 0] * 10:
      self.assertTrue(self._rx(self.packer.make_can_msg_safety("BRAKE2", 0, {"COUNTER": counter}, fix_checksum=checksum)))
      self.assertTrue(self._rx(self.packer.make_can_msg_safety("CAR_OVERALL_SIGNALS2", 0, {"COUNTER2": counter}, fix_checksum=checksum)))
    for name, crc_byte in (("BRAKE2", 0), ("CAR_OVERALL_SIGNALS2", 8)):
      msg = self.packer.make_can_msg_safety(name, 0, {}, fix_checksum=checksum)
      msg[0].data[crc_byte] ^= 0xFF
      self.assertFalse(self._rx(msg), name)

  def test_gas_and_brake_from_real_frames(self):
    # frames as logged on a 2024 H6 GT PHEV, CRC and counter untouched: the panda takes them, and reads the gas pedal
    gas_on = bytes.fromhex("5548180033803802d0023412034c8782e090641c395c4122ff000000000000004dffff3e1728000228787c0000000232ff00045700000800ff00000000000000")
    brake = bytes.fromhex("027269411b00000c" + "00" * 56)
    self.assertTrue(self._rx(common.libsafety_py.make_CANPacket(0x120, 0, brake)))
    self.assertTrue(self._rx(common.libsafety_py.make_CANPacket(0x60, 0, gas_on)))
    self.assertTrue(self.safety.get_gas_pressed_prev())

  def _torque_meas_msg(self, torque):
    # 11-bit signed signal clip to not produce errors on test
    torque_signal = dbc.name_to_msg["RX_STEER_RELATED"].sigs["B_RX_EPS_TORQUE"]
    min_torque, max_torque = get_signal_range(torque_signal)
    torque = max(min(torque, max_torque), min_torque)

    self.eps_counter = (getattr(self, "eps_counter", -1) + 1) % 15
    values = {"B_RX_EPS_TORQUE": torque, "B_COUNTER": self.eps_counter}
    return self.packer.make_can_msg_safety("RX_STEER_RELATED", 0, values, fix_checksum=checksum)

  def _torque_cmd_msg(self, torque, steer_req=1):
    # 10-bit signed signal clip to not produce errors on test
    torque_signal = dbc.name_to_msg["STEER_CMD"].sigs["TORQUE_CMD"]
    min_torque, max_torque = get_signal_range(torque_signal)
    torque = max(min(torque, max_torque), min_torque)

    # the other torque fields as the car port fills them (gwmcan.create_steer_command)
    values = {"STEER_REQUEST": steer_req, "TORQUE_CMD": torque, "TORQUE_REFLECTED": -torque, "INVERT_DIRECTION": int(torque > 0 and steer_req)}
    return self.packer.make_can_msg_safety("STEER_CMD", 0, values)

  def _acc_msg(self, state):
    return self.packer.make_can_msg_safety("ACC", 2, {"CRUISE_STATE_2": state})

  def _engage(self):
    for enable in (1, 0):
      self._rx(self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 0, {"AP_ENABLE_COMMAND": enable}, fix_checksum=checksum))

  def _stalk_idle(self):
    self._rx(self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 0, {}, fix_checksum=checksum))

  def test_the_acc_off_for_half_a_second_ends_the_engagement(self):
    # CRUISE_STATE_2 is 1 with the ACC off. Measured: 2 of 4213 frames engaged, right after the pull
    self._engage()
    self.assertTrue(self.safety.get_controls_allowed())
    for _ in range(4):
      self._rx(self._acc_msg(1))
      self._stalk_idle()
      self.assertTrue(self.safety.get_controls_allowed())
    self._rx(self._acc_msg(1))
    self._stalk_idle()
    self.assertFalse(self.safety.get_controls_allowed())
    # a new pull engages again
    self._engage()
    self.assertTrue(self.safety.get_controls_allowed())

  def test_a_new_pull_gives_the_acc_its_time_again(self):
    self._engage()
    for _ in range(4):
      self._rx(self._acc_msg(1))
    self._engage()
    for _ in range(4):
      self._rx(self._acc_msg(1))
    self._stalk_idle()
    self.assertTrue(self.safety.get_controls_allowed())
    self._rx(self._acc_msg(1))
    self._stalk_idle()
    self.assertFalse(self.safety.get_controls_allowed())

  def test_the_acc_driving_keeps_the_engagement(self):
    self._engage()
    for i in range(100):
      self._rx(self._acc_msg(1 if i % 4 == 0 else 3))  # brief ACC-off frames never add up
      self._stalk_idle()
    self.assertTrue(self.safety.get_controls_allowed())
    for state in (0, 2, 3, 5):
      for _ in range(10):
        self._rx(self._acc_msg(state))
      self._stalk_idle()
      self.assertTrue(self.safety.get_controls_allowed(), state)

  def test_the_acc_off_before_the_pull_does_not_block_it(self):
    for _ in range(20):
      self._rx(self._acc_msg(1))
    self._engage()
    self.assertTrue(self.safety.get_controls_allowed())
    for _ in range(4):
      self._rx(self._acc_msg(1))
    self._stalk_idle()
    self.assertTrue(self.safety.get_controls_allowed())

  def test_eps_torque_checked_like_the_car_sends_it(self):
    # measured on the bus: 24110 frames, CRC (xor 0x61) and counter 0 to 14 on every one
    for counter in [13, 14, 0] * 10:
      self.assertTrue(self._rx(self.packer.make_can_msg_safety("RX_STEER_RELATED", 0, {"B_COUNTER": counter}, fix_checksum=checksum)))
    msg = self.packer.make_can_msg_safety("RX_STEER_RELATED", 0, {}, fix_checksum=checksum)
    msg[0].data[8] ^= 0xFF
    self.assertFalse(self._rx(msg))

  def test_steer_fields_follow_the_torque(self):
    # TORQUE_REFLECTED is the command's negative; INVERT_DIRECTION is set only for a requested positive torque
    self.safety.set_controls_allowed(True)
    self.safety.set_torque_meas(0, 0)
    for torque in (-3, 0, 3):
      good = {"STEER_REQUEST": 1, "TORQUE_CMD": torque, "TORQUE_REFLECTED": -torque, "INVERT_DIRECTION": int(torque > 0)}
      bad = [{"TORQUE_REFLECTED": -torque + 1}, {"TORQUE_REFLECTED": -torque - 1}, {"INVERT_DIRECTION": int(torque <= 0)}]
      for change in [{}] + bad:
        self._set_prev_torque(torque)
        msg = self.packer.make_can_msg_safety("STEER_CMD", 0, {**good, **change})
        self.assertEqual(change == {}, self._tx(msg), (torque, change))
    self._set_prev_torque(0)
    no_request = {"STEER_REQUEST": 0, "TORQUE_CMD": 0, "TORQUE_REFLECTED": 0}
    self.assertTrue(self._tx(self.packer.make_can_msg_safety("STEER_CMD", 0, no_request)))
    self.assertFalse(self._tx(self.packer.make_can_msg_safety("STEER_CMD", 0, {**no_request, "INVERT_DIRECTION": 1})))

  def test_steer_frame_of_the_car_port_passes(self):
    stock = bytes(64)
    self.safety.set_controls_allowed(True)
    for torque in (-4, 0, 4):
      self._set_prev_torque(torque)
      self.safety.set_torque_meas(torque, torque)
      addr, dat, bus = gwmcan.create_steer_command(self.packer, SimpleNamespace(main=0), stock, torque, True)
      self.assertTrue(self._tx(libsafety_py.make_CANPacket(addr, bus, dat)), torque)

  def test_rx_hook(self):
    # speed
    self.assertTrue(self._rx(self._speed_msg(0)))
    # invalidate checksum
    msg = self._speed_msg(0)
    msg[0].data[0] = 0xFF
    self.assertFalse(self._rx(msg))

    # cruise
    self.assertTrue(self._rx(self._pcm_status_msg(0)))
    # invalidate checksum
    msg = self._pcm_status_msg(0)
    msg[0].data[0] = 0xFF
    self.assertFalse(self._rx(msg))


if __name__ == "__main__":
  unittest.main()
