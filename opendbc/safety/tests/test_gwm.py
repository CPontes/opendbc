#!/usr/bin/env python3
import unittest
from types import SimpleNamespace

from opendbc.can.dbc import DBC
from opendbc.car.gwm.values import GwmSafetyFlags
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


class GwmPacker(CANPackerSafety):
  """Frames as the car sends them: the gas, brake and EPS frames with their CRC and a running counter, and our ACC command
  with the mode that goes with its fields (the panda checks it)."""
  COUNTED = {"CAR_OVERALL_SIGNALS2": "COUNTER2", "BRAKE2": "COUNTER", "RX_STEER_RELATED": "B_COUNTER"}

  def __init__(self, dbc_name):
    super().__init__(dbc_name)
    self.counters = {}

  def make_can_msg_safety(self, name_or_addr, bus, values, fix_checksum=None):
    if name_or_addr in self.COUNTED and bus == 0:
      signal = self.COUNTED[name_or_addr]
      if signal not in values:
        self.counters[name_or_addr] = (self.counters.get(name_or_addr, -1) + 1) % 15
        values = {**values, signal: self.counters[name_or_addr]}
      fix_checksum = fix_checksum or checksum
    if name_or_addr == "ACC_CMD" and "BRAKE_OR_GAS_REQ" not in values:
      mode = 13 if values.get("BRAKE_CMD", 0) else 12 if values.get("GAS_CMD", 0) else 8
      values = {**values, "BRAKE_OR_GAS_REQ": mode}
    return super().make_can_msg_safety(name_or_addr, bus, values, fix_checksum)


class TestGwmSafety(common.CarSafetyTest, common.MotorTorqueSteeringSafetyTest, common.LongitudinalGasBrakeSafetyTest):
  TX_MSGS = [[0x12B, 0], [0x143, 0], [0x147, 2], [0xA1, 2]] # Steer, long, wheel touch, cancel
  RELAY_MALFUNCTION_ADDRS = {0: (0x12B, 0x143, 0x23D), 2: (0x147,)}
  FWD_BLACKLISTED_ADDRS = {0: [0x147], 2: [0x12B, 0x143, 0x23D]}

  MAX_RATE_UP = 4
  MAX_RATE_DOWN = 6
  MAX_TORQUE_LOOKUP = [0], [253]
  MAX_RT_DELTA = 100
  MAX_TORQUE_ERROR = 80
  TORQUE_MEAS_TOLERANCE = 1

  MIN_GAS = -192
  MAX_GAS = 4577
  MIN_POSSIBLE_GAS = -192
  MAX_POSSIBLE_GAS = 4608  # reasonably excessive limits, not signal max. The sparse test values stay whole numbers
  INACTIVE_GAS = 0

  MAX_BRAKE = 107
  MAX_POSSIBLE_BRAKE = 180

  def setUp(self):
    self.packer = GwmPacker(opendbc)
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, GwmSafetyFlags.LONG_CONTROL)
    self.safety.init_tests()

  def _user_gas_msg(self, gas):
    values = {"GAS_POSITION": gas}
    return self.packer.make_can_msg_safety("CAR_OVERALL_SIGNALS2", 0, values)

  def _user_brake_msg(self, brake):
    values = {"PEDAL_BRAKE_PRESSED": brake}
    return self.packer.make_can_msg_safety("BRAKE2", 0, values)

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

  def test_stalk_without_the_option_only_goes_out_to_cancel(self):
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

  def _torque_meas_msg(self, torque):
    # 11-bit signed signal clip to not produce errors on test
    torque_signal = dbc.name_to_msg["RX_STEER_RELATED"].sigs["B_RX_EPS_TORQUE"]
    min_torque, max_torque = get_signal_range(torque_signal)
    torque = max(min(torque, max_torque), min_torque)

    values = {"B_RX_EPS_TORQUE": torque}
    return self.packer.make_can_msg_safety("RX_STEER_RELATED", 0, values)

  def _torque_cmd_msg(self, torque, steer_req=1):
    # 10-bit signed signal clip to not produce errors on test
    torque_signal = dbc.name_to_msg["STEER_CMD"].sigs["TORQUE_CMD"]
    min_torque, max_torque = get_signal_range(torque_signal)
    torque = max(min(torque, max_torque), min_torque)

    values = {"STEER_REQUEST": steer_req, "TORQUE_CMD": torque}
    return self.packer.make_can_msg_safety("STEER_CMD", 0, values)

  def _send_brake_msg(self, brake):
    values = {"BRAKE_CMD": -brake, "GAS_CMD": 0}
    return self.packer.make_can_msg_safety("ACC_CMD", 0, values)

  def _send_gas_msg(self, gas):
    values = {"GAS_CMD": gas, "BRAKE_CMD": 0}
    return self.packer.make_can_msg_safety("ACC_CMD", 0, values)

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


class GwmStalkTest(common.SafetyTestBase):
  SAFETY_PARAM = 0

  def setUp(self):
    self.packer = GwmPacker(opendbc)
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, self.SAFETY_PARAM)
    self.safety.init_tests()
    self.stalk_counter = 0

  def _rx_stalk(self, **buttons):
    self.stalk_counter = (self.stalk_counter + 1) % 15
    values = {"COUNTER": self.stalk_counter} | buttons
    return self._rx(self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 0, values, fix_checksum=checksum))

  def _resume(self, **buttons):
    return self._tx(self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 2, {"AP_ENABLE_COMMAND": 1, **buttons}, fix_checksum=checksum))

  def _speed(self, speed):
    values = {f"{pos}_WHEEL_SPEED": speed * 3.6 for pos in ["FRONT_LEFT", "FRONT_RIGHT", "REAR_LEFT", "REAR_RIGHT"]}
    return self._rx(self.packer.make_can_msg_safety("WHEEL_SPEEDS", 0, values, fix_checksum=checksum))

  def _gas(self, gas):
    return self._rx(self.packer.make_can_msg_safety("CAR_OVERALL_SIGNALS2", 0, {"GAS_POSITION": gas}))

  def _brake(self, pressed):
    return self._rx(self.packer.make_can_msg_safety("BRAKE2", 0, {"PEDAL_BRAKE_PRESSED": pressed}))

  def _engage(self):
    self._speed(0)
    self._rx_stalk(AP_ENABLE_COMMAND=1)
    self._rx_stalk()
    self.assertTrue(self.safety.get_controls_allowed())

class TestGwmAutoResume(GwmStalkTest):
  SAFETY_PARAM = GwmSafetyFlags.AUTO_RESUME.value

  def test_resume_frame_of_the_car_port_passes(self):
    self._engage()
    stock = self.packer.make_can_msg("STEER_AND_AP_STALK", 0, dict.fromkeys(STALK_BUTTONS, 1))[1]
    addr, dat, bus = gwmcan.create_resume_command(self.packer, SimpleNamespace(camera=2), stock)
    self.assertTrue(self._tx(libsafety_py.make_CANPacket(addr, bus, dat)))

  def test_refused_without_the_option(self):
    self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, 0)
    self.safety.init_tests()
    self._engage()
    self.assertFalse(self._resume())
    self.assertEqual(0, self.safety.safety_fwd_hook(2, 0xA1))

  def test_stopped_and_engaged_passes(self):
    self._engage()
    self.assertTrue(self._resume())

  def test_refused_while_disengaged(self):
    self._speed(0)
    self.assertFalse(self._resume())

  def test_refused_while_moving(self):
    self._engage()
    self._speed(5)
    self.assertFalse(self._resume())

  def test_refused_with_gas(self):
    self._engage()
    self._gas(10)
    self.assertFalse(self._resume())
    self._gas(0)
    self.assertTrue(self._resume())

  def test_refused_with_the_brake(self):
    self._engage()
    self._brake(1)
    self.assertFalse(self._resume())

  def test_brake_during_the_stop_needs_a_new_pull(self):
    self._engage()
    self._brake(1)
    self._rx_stalk()
    self._brake(0)
    self._rx_stalk()
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertFalse(self._resume())

  def test_refused_with_a_driver_button(self):
    self._engage()
    for button in STALK_BUTTONS + ("AP_CANCEL_COMMAND",):
      self._rx_stalk(**{button: 1})
      if button != "AP_CANCEL_COMMAND":
        self.assertFalse(self._resume(), button)
      self._rx_stalk()
    self.assertFalse(self._resume())  # cancel ended it

  def test_only_the_enable_button(self):
    self._engage()
    for button in STALK_BUTTONS[1:]:
      self.assertFalse(self._resume(**{button: 1}), button)
    self.assertFalse(self._resume(AP_CANCEL_COMMAND=1))

  def test_a_budget_per_stop(self):
    self._engage()
    for _ in range(20):
      self.assertTrue(self._resume())
    self.assertFalse(self._resume())
    self._speed(1)
    self._speed(0)
    self.assertTrue(self._resume())

  def test_cancel_still_passes_after_the_budget(self):
    self._engage()
    for _ in range(25):
      self._resume()
    self.assertTrue(self._tx(self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 2, {"AP_CANCEL_COMMAND": 1}, fix_checksum=checksum)))

  def test_original_stalk_held_only_right_after_our_frame(self):
    self._engage()
    self.safety.set_timer(100000)
    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0xA1))
    self.assertTrue(self._resume())
    self.assertEqual(-1, self.safety.safety_fwd_hook(0, 0xA1))
    self.safety.set_timer(100000 + 14999)
    self.assertEqual(-1, self.safety.safety_fwd_hook(0, 0xA1))
    self.safety.set_timer(100000 + 15000)
    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0xA1))

  def test_hold_covers_this_message_and_direction_only(self):
    self._engage()
    self.safety.set_timer(100000)
    self.assertTrue(self._resume())
    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0x13B))
    self.assertEqual(0, self.safety.safety_fwd_hook(2, 0xA1))

  def test_refused_frame_holds_nothing(self):
    self._engage()
    self._gas(10)
    self.safety.set_timer(100000)
    self.assertFalse(self._resume())
    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0xA1))

  def test_a_driver_button_always_reaches_the_camera(self):
    self._engage()
    self.safety.set_timer(100000)
    self.assertTrue(self._resume())
    self._rx_stalk(AP_CANCEL_COMMAND=1)
    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0xA1))

  def test_speed_button_reaches_the_camera(self):
    self._engage()
    self.safety.set_timer(100000)
    self.assertTrue(self._resume())
    self._rx_stalk(AP_INCREASE_SPEED_COMMAND=1)
    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0xA1))

  def test_nothing_held_before_the_first_frame(self):
    self._engage()
    self.safety.set_timer(0)
    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0xA1))


class TestGwmAccMinus(GwmStalkTest):
  SAFETY_PARAM = GwmSafetyFlags.ACC_MINUS.value

  def _minus(self, **values):
    return self._tx(self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 2, {"AP_DECREASE_SPEED_COMMAND": 1, **values}, fix_checksum=checksum))

  def _driving(self):
    self._engage()
    self._speed(15)
    self.safety.set_timer(100000000)

  def test_passes_engaged_and_moving(self):
    self._driving()
    self.assertTrue(self._minus())

  def test_frame_of_the_car_port_passes(self):
    self._driving()
    stock = self.packer.make_can_msg("STEER_AND_AP_STALK", 0, dict.fromkeys(STALK_BUTTONS, 1))[1]
    addr, dat, bus = gwmcan.create_minus_command(self.packer, SimpleNamespace(camera=2), stock)
    self.assertTrue(self._tx(libsafety_py.make_CANPacket(addr, bus, dat)))

  def test_refused_without_the_option(self):
    self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, 0)
    self.safety.init_tests()
    self._driving()
    self.assertFalse(self._minus())

  def test_refused_stopped(self):
    self._engage()
    self.safety.set_timer(100000000)
    self.assertFalse(self._minus())

  def test_refused_while_disengaged(self):
    self._driving()
    self.safety.set_controls_allowed(False)
    self.assertFalse(self._minus())

  def test_never_the_plus_or_any_other_button(self):
    self._driving()
    for button in STALK_BUTTONS:
      if button != "AP_DECREASE_SPEED_COMMAND":
        self.assertFalse(self._tx(self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 2, {button: 1}, fix_checksum=checksum)), button)
        self.assertFalse(self._minus(**{button: 1}), button)

  def test_refused_with_gas_or_brake(self):
    self._driving()
    self._gas(10)
    self.assertFalse(self._minus())
    self._gas(0)
    self.assertTrue(self._minus())
    self._brake(1)
    self.assertFalse(self._minus())

  def test_refused_with_a_driver_button(self):
    self._driving()
    self._rx_stalk(AP_INCREASE_SPEED_COMMAND=1)
    self.assertFalse(self._minus())
    self._rx_stalk()
    self.assertTrue(self._minus())

  def test_a_budget_per_minute(self):
    self._driving()
    for _ in range(78):
      self.assertTrue(self._minus())
    self.assertFalse(self._minus())
    self.safety.set_timer(100000000 + 59999999)
    self.assertFalse(self._minus())
    self.safety.set_timer(100000000 + 60000000)
    self.assertTrue(self._minus())

  def test_the_budget_counts_any_minute_not_a_fixed_window(self):
    self._driving()
    self.assertTrue(self._minus())
    self.safety.set_timer(100000000 + 59000000)
    for _ in range(77):
      self.assertTrue(self._minus())
    self.assertFalse(self._minus())
    self.safety.set_timer(100000000 + 60000000)  # a minute after the first, one second after the last
    self.assertFalse(self._minus())
    self.safety.set_timer(100000000 + 59000000 + 59999999)
    self.assertFalse(self._minus())
    self.safety.set_timer(100000000 + 59000000 + 60000000)
    self.assertTrue(self._minus())

  def test_refused_frames_do_not_keep_the_budget_away(self):
    self._driving()
    for _ in range(78):
      self.assertTrue(self._minus())
    for i in range(10):
      self.safety.set_timer(100000000 + i * 5000000)
      self.assertFalse(self._minus())
    self.safety.set_timer(100000000 + 60000000)
    self.assertTrue(self._minus())

  def test_a_new_session_clears_the_budget(self):
    self._driving()
    for _ in range(78):
      self._minus()
    self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, self.SAFETY_PARAM)
    self.safety.init_tests()
    self._driving()
    self.assertTrue(self._minus())

  def test_cancel_still_passes_after_the_budget_of_the_minus(self):
    self._driving()
    for _ in range(80):
      self._minus()
    self.assertTrue(self._tx(self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 2, {"AP_CANCEL_COMMAND": 1}, fix_checksum=checksum)))

  def test_original_stalk_held_right_after_our_minus(self):
    self._driving()
    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0xA1))
    self.assertTrue(self._minus())
    self.assertEqual(-1, self.safety.safety_fwd_hook(0, 0xA1))
    self.safety.set_timer(100000000 + 15000)
    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0xA1))


class TestGwmAccPlus(GwmStalkTest):
  """The plus of the ACC, to the road's limit (CRUISE-01): the panda checks the ACC's own state and set speed."""
  SAFETY_PARAM = GwmSafetyFlags.ACC_PLUS.value

  def _plus(self, **values):
    return self._tx(self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 2, {"AP_INCREASE_SPEED_COMMAND": 1, **values}, fix_checksum=checksum))

  def _acc(self, state=3, kph=60):
    return self._rx(self.packer.make_can_msg_safety("ACC", 2, {"CRUISE_STATE_2": state, "ACC_SPEED_SELECTION": kph}))

  def _driving(self, kph=60):
    self._engage()
    self._speed(15)
    self._acc(3, kph)
    self.safety.set_timer(100000000)

  def test_passes_with_the_acc_driving_below_the_ceiling(self):
    self._driving(60)
    self.assertTrue(self._plus())
    self._acc(3, 105)
    self.assertTrue(self._plus())

  def test_frame_of_the_car_port_passes(self):
    self._driving()
    stock = self.packer.make_can_msg("STEER_AND_AP_STALK", 0, dict.fromkeys(STALK_BUTTONS, 1))[1]
    addr, dat, bus = gwmcan.create_plus_command(self.packer, SimpleNamespace(camera=2), stock)
    self.assertTrue(self._tx(libsafety_py.make_CANPacket(addr, bus, dat)))

  def test_never_at_or_above_the_ceiling(self):
    for kph in (110, 111, 120, 255):
      self._driving(kph)
      self.assertFalse(self._plus(), kph)

  def test_only_with_the_acc_driving(self):
    for state in (0, 1, 2, 4, 7):
      self._driving()
      self._acc(state, 60)
      self.assertFalse(self._plus(), state)

  def test_nothing_before_the_camera_said_how_the_acc_is(self):
    self._engage()
    self._speed(15)
    self.safety.set_timer(100000000)
    self.assertFalse(self._plus())

  def test_refused_without_the_option_and_the_minus_flag_is_not_enough(self):
    for param in (0, GwmSafetyFlags.ACC_MINUS.value):
      self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, param)
      self.safety.init_tests()
      self._driving()
      self.assertFalse(self._plus(), param)

  def test_the_plus_flag_lets_no_minus_through(self):
    self._driving()
    self.assertFalse(self._tx(self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 2, {"AP_DECREASE_SPEED_COMMAND": 1}, fix_checksum=checksum)))

  def test_refused_stopped_disengaged_or_with_the_driver(self):
    self._engage()
    self._acc(3, 60)
    self.safety.set_timer(100000000)
    self.assertFalse(self._plus())  # stopped
    self._driving()
    self.safety.set_controls_allowed(False)
    self.assertFalse(self._plus())
    self._driving()
    self._gas(10)
    self.assertFalse(self._plus())
    self._gas(0)
    self._rx_stalk(AP_DECREASE_SPEED_COMMAND=1)
    self.assertFalse(self._plus())
    self._rx_stalk()
    self.assertTrue(self._plus())
    self._brake(1)
    self.assertFalse(self._plus())

  def test_only_the_plus_button(self):
    self._driving()
    for button in STALK_BUTTONS:
      if button != "AP_INCREASE_SPEED_COMMAND":
        self.assertFalse(self._plus(**{button: 1}), button)

  def test_a_budget_per_minute(self):
    self._driving()
    for _ in range(104):
      self.assertTrue(self._plus())
    self.assertFalse(self._plus())
    self.safety.set_timer(100000000 + 59999999)
    self.assertFalse(self._plus())
    self.safety.set_timer(100000000 + 60000000)
    self.assertTrue(self._plus())

  def test_a_new_session_clears_it(self):
    self._driving()
    for _ in range(104):
      self._plus()
    self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, self.SAFETY_PARAM)
    self.safety.init_tests()
    self.assertFalse(self._plus())  # nothing from the camera yet in this session
    self._driving()
    self.assertTrue(self._plus())

  def test_cancel_still_passes_after_the_budget(self):
    self._driving()
    for _ in range(110):
      self._plus()
    self.assertTrue(self._tx(self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 2, {"AP_CANCEL_COMMAND": 1}, fix_checksum=checksum)))

  def test_original_stalk_held_right_after_our_plus(self):
    self._driving()
    self.assertTrue(self._plus())
    self.assertEqual(-1, self.safety.safety_fwd_hook(0, 0xA1))
    self.safety.set_timer(100000000 + 15000)
    self.assertEqual(2, self.safety.safety_fwd_hook(0, 0xA1))


class TestGwmBrakeHandoff(GwmStalkTest):
  """The pedal takes a moment to build pressure, and the stock ACC keeps braking for about 0.1 s until it drops. Ours may too."""
  SAFETY_PARAM = GwmSafetyFlags.LONG_CONTROL
  T0 = 100000000

  def setUp(self):
    super().setUp()
    self.safety.set_timer(self.T0)
    self._speed(10)
    self._rx_stalk(AP_ENABLE_COMMAND=1)
    self._rx_stalk()

  def _long(self, brake=0, gas=0):
    return self._tx(self.packer.make_can_msg_safety("ACC_CMD", 0, {"BRAKE_CMD": -brake, "GAS_CMD": gas}))

  def _pedal_while_braking(self, brake=70):
    self.assertTrue(self._long(brake))
    self._brake(1)
    self.assertFalse(self.safety.get_controls_allowed())

  def test_the_brake_stays_a_moment_after_the_pedal(self):
    self._pedal_while_braking(70)
    self.assertTrue(self._long(70))
    self.assertTrue(self._long(60))

  def test_it_only_comes_down(self):
    self._pedal_while_braking(70)
    self.assertFalse(self._long(71))
    self.assertTrue(self._long(60))
    self.assertFalse(self._long(61))
    self.assertTrue(self._long(60))

  def test_it_ends_with_the_window(self):
    self._pedal_while_braking(70)
    self.safety.set_timer(self.T0 + 199999)
    self.assertTrue(self._long(70))
    self.safety.set_timer(self.T0 + 200000)
    self.assertFalse(self._long(70))
    self.assertTrue(self._long(0))

  def test_never_gas(self):
    self._pedal_while_braking(70)
    self.assertFalse(self._long(0, 1))
    self.assertFalse(self._long(70, 1))
    self.assertFalse(self._long(70, -1))

  def test_the_brake_field_at_its_zero_counts(self):
    self.assertTrue(self._long(41))
    self._brake(1)
    self.assertTrue(self._long(41))
    self.assertFalse(self._long(42))

  def test_not_below_the_zero_of_the_brake_field(self):
    self.assertTrue(self._long(40))
    self._brake(1)
    self.assertFalse(self._long(40))

  def test_not_out_of_the_gas(self):
    self.assertTrue(self._long(0, 500))
    self._brake(1)
    self.assertFalse(self._long(70))

  def test_a_refused_frame_is_no_brake_before_it(self):
    self.assertTrue(self._long(70))
    self.assertFalse(self._long(108))
    self._brake(1)
    self.assertFalse(self._long(70))

  def test_letting_go_of_the_pedal_ends_it(self):
    self._pedal_while_braking(70)
    self._brake(0)
    self.assertFalse(self._long(70))

  def test_the_gas_pedal_ends_it(self):
    self._pedal_while_braking(70)
    self._gas(100)
    self.assertFalse(self._long(70))

  def test_a_second_press_starts_nothing(self):
    self._pedal_while_braking(70)
    self._brake(0)
    self._brake(1)
    self.assertFalse(self._long(70))

  def test_not_when_openpilot_was_not_in_control(self):
    self.assertTrue(self._long(70))
    self._rx_stalk(AP_CANCEL_COMMAND=1)
    self.assertFalse(self.safety.get_controls_allowed())
    self._brake(1)
    self.assertFalse(self._long(70))

  def test_with_the_gas_held_nothing_starts(self):
    self.assertTrue(self._long(70))
    self._gas(100)
    self._brake(1)
    self._gas(0)
    self.assertFalse(self._long(70))

  def test_a_new_session_forgets_it(self):
    self._pedal_while_braking(70)
    self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, self.SAFETY_PARAM)
    self.safety.init_tests()
    self.assertFalse(self._long(70))

  def test_a_pedal_held_down_starts_nothing(self):
    self._brake(1)
    self.safety.set_controls_allowed(1)
    self.assertTrue(self._long(70))
    self._brake(1)
    self.safety.set_controls_allowed(0)
    self.assertFalse(self._long(70))

  def test_the_window_never_comes_back(self):
    self._pedal_while_braking(70)
    self.safety.set_timer(self.T0 + 200000)
    self.assertFalse(self._long(70))
    self.safety.set_timer(self.T0 + 100)  # the timer wraps after 71 min
    self.assertFalse(self._long(70))

  def test_a_touch_of_the_gas_ends_it_for_good(self):
    self._pedal_while_braking(70)
    self._gas(100)
    self.assertFalse(self._long(70))
    self._gas(0)
    self.assertFalse(self._long(70))

  def test_a_negative_brake_field_is_refused(self):
    self._pedal_while_braking(70)
    self.assertFalse(self._long(-1))
    self.assertTrue(self._long(70))

  def test_a_new_session_forgets_the_last_brake(self):
    self.assertTrue(self._long(70))
    self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, self.SAFETY_PARAM)
    self.safety.init_tests()
    self.safety.set_timer(self.T0)
    self._speed(10)
    self._rx_stalk(AP_ENABLE_COMMAND=1)
    self._rx_stalk()
    self._brake(1)
    self.assertFalse(self._long(70))


class TestGwmBrakeHandoffAlwaysOn(TestGwmBrakeHandoff):
  SAFETY_PARAM = GwmSafetyFlags.LONG_CONTROL | GwmSafetyFlags.ALWAYS_ON_LATERAL

  def _pedal_while_braking(self, brake=70):
    self.assertTrue(self._long(brake))
    self._brake(1)

  def test_a_pedal_held_down_starts_nothing(self):
    self._brake(1)
    self.assertFalse(self._long(70))
    self._brake(1)
    self.assertFalse(self._long(70))

  def test_a_second_press_starts_nothing(self):
    self._pedal_while_braking(70)
    self._brake(0)
    self._brake(1)
    self.assertFalse(self._long(70))

  def test_not_when_openpilot_was_not_in_control(self):
    self.assertTrue(self._long(70))
    self._rx_stalk(AP_CANCEL_COMMAND=1)
    self._brake(1)
    self.assertFalse(self._long(70))


class TestGwmAlwaysOnLateral(common.SafetyTestBase):
  """Brake pauses longitudinal only, steering stays engaged until cancel; lever resumes longitudinal."""

  def setUp(self):
    self.packer = GwmPacker(opendbc)
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, GwmSafetyFlags.LONG_CONTROL | GwmSafetyFlags.ALWAYS_ON_LATERAL)
    self.safety.init_tests()
    self.stalk_counter = 0

  def _stalk(self, enable=0, cancel=0):
    self.stalk_counter = (self.stalk_counter + 1) % 15
    values = {"AP_ENABLE_COMMAND": enable, "AP_CANCEL_COMMAND": cancel, "COUNTER": self.stalk_counter}
    return self._rx(self.packer.make_can_msg_safety("STEER_AND_AP_STALK", 0, values, fix_checksum=checksum))

  def _pull_lever(self):
    self._stalk(enable=1)
    self._stalk(enable=0)

  def _brake(self, pressed):
    return self._rx(self.packer.make_can_msg_safety("BRAKE2", 0, {"PEDAL_BRAKE_PRESSED": pressed}))

  def _speed(self, speed):
    values = {f"{pos}_WHEEL_SPEED": speed * 3.6 for pos in ["FRONT_LEFT", "FRONT_RIGHT", "REAR_LEFT", "REAR_RIGHT"]}
    return self._rx(self.packer.make_can_msg_safety("WHEEL_SPEEDS", 0, values, fix_checksum=checksum))

  def _steer_ok(self):
    msg = self.packer.make_can_msg_safety("STEER_CMD", 0, {"STEER_REQUEST": 1, "TORQUE_CMD": 0})
    return self._tx(msg)

  def _gas_ok(self, gas):
    return self._tx(self.packer.make_can_msg_safety("ACC_CMD", 0, {"GAS_CMD": gas, "BRAKE_CMD": 0}))

  def _brake_cmd_ok(self, brake):
    return self._tx(self.packer.make_can_msg_safety("ACC_CMD", 0, {"GAS_CMD": 0, "BRAKE_CMD": -brake}))

  def test_brake_pauses_longitudinal_only(self):
    self._speed(10)
    self._pull_lever()
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertTrue(self._gas_ok(1000))
    self.assertTrue(self._brake_cmd_ok(50))

    # brake while moving: steering stays, longitudinal limited to inactive commands once the handoff is over
    self.safety.set_timer(10000000)
    self._brake(1)
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertTrue(self._steer_ok())
    self.assertFalse(self._gas_ok(1000))
    self.safety.set_timer(10200000)
    self.assertFalse(self._brake_cmd_ok(50))
    self.assertTrue(self._gas_ok(0))

    # releasing the brake does not resume longitudinal
    self._brake(0)
    self.assertFalse(self._gas_ok(1000))
    self.assertTrue(self._steer_ok())

    # lever pull resumes it
    self._pull_lever()
    self.assertTrue(self._gas_ok(1000))

  def test_brake_at_standstill_pauses_longitudinal(self):
    self._speed(0)
    self._pull_lever()
    self._brake(1)
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertFalse(self._gas_ok(1000))

  def test_cancel_disengages_everything(self):
    self._speed(10)
    self._pull_lever()
    self._stalk(cancel=1)
    self.assertFalse(self.safety.get_controls_allowed())
    self.assertFalse(self._gas_ok(1000))

  def test_no_longitudinal_before_lever(self):
    self.safety.set_controls_allowed(True)
    self.assertFalse(self._gas_ok(1000))
    self.assertTrue(self._gas_ok(0))

  def test_flag_off_brake_disengages(self):
    self.safety.set_safety_hooks(CarParams.SafetyModel.gwm, GwmSafetyFlags.LONG_CONTROL)
    self.safety.init_tests()
    self._speed(10)
    self._pull_lever()
    self.assertTrue(self.safety.get_controls_allowed())
    self._brake(1)
    self.assertFalse(self.safety.get_controls_allowed())

# *** SAFE-02: the mode of our ACC command, CRC and counter on gas, brake and 0x147, and the ACC off ending the engagement ***

INACTIVE_MODES = (0, 4, 8, 20)
ACTIVE_MODES = (12, 13, 25)


class TestGwmAccMode(GwmStalkTest):
  """BRAKE_OR_GAS_REQ decides what the PCM does, whatever the gas and brake fields say: the hold mode brakes with a zero brake."""
  SAFETY_PARAM = GwmSafetyFlags.LONG_CONTROL

  def _acc(self, mode, gas=0, brake=0):
    return self._tx(self.packer.make_can_msg_safety("ACC_CMD", 0, {"BRAKE_OR_GAS_REQ": mode, "GAS_CMD": gas, "BRAKE_CMD": -brake}))

  def test_disengaged_only_the_modes_that_dont_actuate(self):
    self._speed(5)
    for mode in range(32):
      self.assertEqual(mode in INACTIVE_MODES, self._acc(mode), mode)

  def test_engaged_the_gas_brake_and_hold_too(self):
    self._engage()
    for mode in range(32):
      self.assertEqual(mode in INACTIVE_MODES + ACTIVE_MODES, self._acc(mode), mode)

  def test_with_the_gas_pedal_only_the_modes_that_dont_actuate(self):
    self._engage()
    self._gas(10)
    for mode in range(32):
      self.assertEqual(mode in INACTIVE_MODES, self._acc(mode), mode)  # stopped too: the gas pedal ends the hold

  def test_gas_only_in_the_gas_mode_brake_only_in_brake_and_hold(self):
    self._engage()
    for mode in INACTIVE_MODES + ACTIVE_MODES:
      self.assertEqual(mode == 12, self._acc(mode, gas=60), mode)
      self.assertEqual(mode == 12, self._acc(mode, gas=-192), mode)
      self.assertEqual(mode in (13, 25), self._acc(mode, brake=41), mode)
      self.assertFalse(self._acc(mode, gas=60, brake=41), mode)

  def test_after_cancel_no_hold_moving(self):
    self._engage()
    self._speed(1)
    self.assertTrue(self._acc(25, brake=80))
    self._rx_stalk(AP_CANCEL_COMMAND=1)
    self.assertFalse(self._acc(25))
    self.assertFalse(self._acc(13))
    self.assertTrue(self._acc(8))

  def test_the_hold_of_a_stopped_car_stays_disengaged(self):
    # the camera's own hold goes on as the canal always sent it: let go of, a car stopped on a climb rolls back
    self._speed(0)
    self.assertTrue(self._acc(25))
    self.assertFalse(self._acc(25, brake=41))  # with no brake of ours
    self._gas(10)
    self.assertFalse(self._acc(25))  # nor with the gas pedal
    self._gas(0)
    self._speed(0.5)
    self.assertFalse(self._acc(25))  # nor moving

  def test_the_frame_of_the_car_port_passes_whatever_the_camera_does(self):
    # disengaged: the camera's gas, brake or hold goes out as standby, its other modes as they are
    # stopped and with no gas pedal, the camera's hold stays as it is
    for stopped in (False, True):
      self._speed(0 if stopped else 5)
      for pedal, gas_pedal, expected in ((False, False, 8), (True, False, 4), (False, True, 20)):
        if gas_pedal:
          self._gas(10)
        for mode in INACTIVE_MODES + ACTIVE_MODES:
          stock = self.packer.make_can_msg("ACC_CMD", 2, {"BRAKE_OR_GAS_REQ": mode, "BRAKE_CMD": -60, "GAS_CMD": 500})[1]
          addr, dat, bus = gwmcan.create_longitudinal_command(self.packer, SimpleNamespace(main=0), stock, 0., False, False, False,
                                                               pedal=pedal, gas_pedal=gas_pedal, stopped=stopped)
          kept = mode in INACTIVE_MODES or (mode == 25 and stopped and not gas_pedal)
          self.assertEqual(dat[9] & 0x1F, mode if kept else expected, (mode, stopped, pedal, gas_pedal))
          self.assertTrue(self._tx(libsafety_py.make_CANPacket(addr, bus, dat)), (mode, stopped, pedal, gas_pedal))
        self._gas(0)


class TestGwmAccModeAlwaysOn(TestGwmAccMode):
  SAFETY_PARAM = GwmSafetyFlags.LONG_CONTROL | GwmSafetyFlags.ALWAYS_ON_LATERAL

  def test_paused_by_the_brake_only_the_modes_that_dont_actuate(self):
    # always-on lateral: the brake pauses longitudinal and steering stays. The hold must not come back with it
    self._engage()
    self._speed(5)
    self._brake(1)
    self._brake(0)
    self.assertTrue(self.safety.get_controls_allowed())
    for mode in range(32):  # moving: no hold
      self.assertEqual(mode in INACTIVE_MODES, self._acc(mode), mode)


class TestGwmHandoffMode(TestGwmBrakeHandoff):
  """The handoff to the pedal keeps a brake that only comes down: in the brake mode, or the hold if it was holding. Never gas."""

  def _hold(self, brake=70):
    return self._tx(self.packer.make_can_msg_safety("ACC_CMD", 0, {"BRAKE_OR_GAS_REQ": 25, "BRAKE_CMD": -brake, "GAS_CMD": 0}))

  def test_from_the_brake_no_hold(self):
    self._pedal_while_braking(70)
    self.assertFalse(self._hold(60))
    self.assertTrue(self._long(60))

  def test_from_the_hold_the_hold_or_the_brake(self):
    self.assertTrue(self._hold(80))
    self._brake(1)
    self.assertTrue(self._hold(80))
    self.assertTrue(self._long(70))
    self.assertTrue(self._hold(60))

  def test_no_gas_mode_with_no_gas(self):
    self._pedal_while_braking(70)
    self.assertFalse(self._tx(self.packer.make_can_msg_safety("ACC_CMD", 0, {"BRAKE_OR_GAS_REQ": 12, "BRAKE_CMD": 0, "GAS_CMD": 0})))

  def test_a_refused_frame_is_no_hold_before_it(self):
    self.assertTrue(self._hold(80))
    self.assertFalse(self._hold(108))  # over the limit: what the panda saw last is standby
    self._brake(1)
    self.assertFalse(self._hold(60))


class TestGwmHandoffModeAlwaysOn(TestGwmHandoffMode, TestGwmBrakeHandoffAlwaysOn):
  pass


class TestGwmChecksums(GwmStalkTest):
  """Gas (0x60), brake (0x120) and EPS torque (0x147): CRC and counter on every frame of the logs (crc_gwm.py), now checked."""

  def test_good_frames_and_wrapping_counters_pass(self):
    for counter in [13, 14, 0] * 10:
      self.assertTrue(self._rx(self.packer.make_can_msg_safety("BRAKE2", 0, {"COUNTER": counter})))
      self.assertTrue(self._rx(self.packer.make_can_msg_safety("CAR_OVERALL_SIGNALS2", 0, {"COUNTER2": counter})))
      self.assertTrue(self._rx(self.packer.make_can_msg_safety("RX_STEER_RELATED", 0, {"B_COUNTER": counter})))

  def test_a_bad_crc_is_refused(self):
    for name, crc_byte in (("BRAKE2", 0), ("CAR_OVERALL_SIGNALS2", 8), ("RX_STEER_RELATED", 8)):
      msg = self.packer.make_can_msg_safety(name, 0, {})
      msg[0].data[crc_byte] ^= 0xFF
      self.assertFalse(self._rx(msg), name)

  def test_real_frames(self):
    # as logged on a 2024 H6 GT PHEV, CRC and counter untouched: the panda takes them, and reads the gas pedal
    gas_on = bytes.fromhex("5548180033803802d0023412034c8782e090641c395c4122ff000000000000004dffff3e1728000228787c0000000232ff00045700000800ff00000000000000")
    brake = bytes.fromhex("027269411b00000c" + "00" * 56)
    self.assertTrue(self._rx(libsafety_py.make_CANPacket(0x120, 0, brake)))
    self.assertTrue(self._rx(libsafety_py.make_CANPacket(0x60, 0, gas_on)))
    self.assertTrue(self.safety.get_gas_pressed_prev())


class TestGwmAccOff(GwmStalkTest):
  """The ACC off (CRUISE_STATE_2 = 1) for 5 of its frames (0.5 s) ends the engagement: it dropped out, or never took the pull."""
  SAFETY_PARAM = GwmSafetyFlags.LONG_CONTROL | GwmSafetyFlags.AUTO_RESUME | GwmSafetyFlags.ACC_MINUS | GwmSafetyFlags.ACC_PLUS

  def _acc_state(self, state):
    self._rx(self.packer.make_can_msg_safety("ACC", 2, {"CRUISE_STATE_2": state, "ACC_SPEED_SELECTION": 60}))

  def _off_frames(self, n):
    for _ in range(n):
      self._acc_state(1)
    self._rx_stalk()

  def test_five_frames_end_it(self):
    self._engage()
    self._off_frames(4)
    self.assertTrue(self.safety.get_controls_allowed())
    self._off_frames(1)
    self.assertFalse(self.safety.get_controls_allowed())
    self._engage()  # a new pull engages again

  def test_a_new_pull_gives_the_acc_its_time_again(self):
    self._engage()
    self._off_frames(4)
    self._engage()
    self._off_frames(4)
    self.assertTrue(self.safety.get_controls_allowed())

  def test_off_before_the_pull_does_not_count(self):
    self._off_frames(20)
    self._engage()
    self._off_frames(4)
    self.assertTrue(self.safety.get_controls_allowed())

  def test_the_acc_states_of_a_stop_and_a_hold_keep_it(self):
    # 2 waiting at a stop or holding, 3 driving, 5 the driver's gas: never 1 in a stop
    self._engage()
    for state in (0, 2, 3, 5):
      for _ in range(20):
        self._acc_state(state)
      self._rx_stalk()
      self.assertTrue(self.safety.get_controls_allowed(), state)

  def test_short_off_frames_never_add_up(self):
    self._engage()
    for i in range(100):
      self._acc_state(1 if i % 4 == 0 else 3)
      self._rx_stalk()
    self.assertTrue(self.safety.get_controls_allowed())


class TestGwmAccOffAlwaysOn(GwmStalkTest):
  """Always-on lateral steers without the ACC by design: the ACC off ends nothing."""
  SAFETY_PARAM = GwmSafetyFlags.LONG_CONTROL | GwmSafetyFlags.ALWAYS_ON_LATERAL

  def test_the_acc_off_keeps_steering_and_longitudinal(self):
    self._engage()
    for _ in range(50):
      self._rx(self.packer.make_can_msg_safety("ACC", 2, {"CRUISE_STATE_2": 1}))
    self._rx_stalk()
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertTrue(self._tx(self.packer.make_can_msg_safety("ACC_CMD", 0, {"BRAKE_OR_GAS_REQ": 13, "BRAKE_CMD": -60, "GAS_CMD": 0})))


if __name__ == "__main__":
  unittest.main()
