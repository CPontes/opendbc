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

  return addr, ret, bus


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

  MIN_GAS = -10
  MAX_GAS = 4577
  MIN_POSSIBLE_GAS = -11
  MAX_POSSIBLE_GAS = 4600  # reasonably excessive limits, not signal max
  INACTIVE_GAS = 0

  MAX_BRAKE = 107
  MAX_POSSIBLE_BRAKE = 180

  def setUp(self):
    self.packer = CANPackerSafety(opendbc)
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


class TestGwmAutoResume(common.SafetyTestBase):
  SAFETY_PARAM = GwmSafetyFlags.AUTO_RESUME.value

  def setUp(self):
    self.packer = CANPackerSafety(opendbc)
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


class TestGwmAlwaysOnLateral(common.SafetyTestBase):
  """Brake pauses longitudinal only, steering stays engaged until cancel; lever resumes longitudinal."""

  def setUp(self):
    self.packer = CANPackerSafety(opendbc)
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

    # brake while moving: steering stays, longitudinal limited to inactive commands
    self._brake(1)
    self.assertTrue(self.safety.get_controls_allowed())
    self.assertTrue(self._steer_ok())
    self.assertFalse(self._gas_ok(1000))
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

if __name__ == "__main__":
  unittest.main()
