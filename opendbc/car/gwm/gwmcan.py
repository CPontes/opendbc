import numpy as np
from opendbc.car import CanBusBase
from opendbc.can.packer import set_value
from opendbc.car.gwm.values import CarControllerParams

STALK_MAX_COUNTER = 14  # 0xA1 counts 0 to 14, 15 never shows up on the bus
STEER_MAX_COUNTER = 14  # the camera's 0x12B too: 15 was ours, on 1 of every 15 frames

# The gas the Haval's own ACC asks for each acceleration: 212 min of its commands, with the slope taken out.
# Below zero it's less engine torque, down to coasting: the stock ACC spends 14% of its time there
ACC_GAS_ACCEL = [-0.14, -0.1, -0.05, 0., 0.1, 0.2, 0.35, 0.5, 0.75, 1., 1.25, 1.5, 1.75, 2.]  # m/s^2
ACC_GAS = [-192, -100, -25, 60, 215, 330, 545, 725, 1105, 1440, 1900, 2215, 2555, 3000]

# BRAKE_OR_GAS_REQ, as the camera sends it. The panda lets out only the ones that don't actuate while openpilot isn't in control
MODE_START = 0  # a frame at start
MODE_ACC_OFF = 4
MODE_STANDBY = 8
MODE_GAS = 12
MODE_BRAKE = 13
MODE_GAS_OVERRIDE = 20  # the driver on the gas with the ACC on
MODE_HOLD = 25  # the end of a stop and the stop itself, on the friction brake
# The three bits above BRAKE_OR_GAS_REQ in byte 9, unnamed in the DBC: the camera sets them to 2 only in the gas modes (12 and 20),
# for an instant as it leaves a stop, and to 0 in every other mode (routes 0c, 16 and 41). It is its request to drive off: passed
# on with our hold, the car let the brake go with the car still stopped, from ~500 of pressure to 0 in a second, and it rolled
# back on a climb (route 41, 37.5 min; 18 of 25 pressure losses in routes 36 to 41). Our brake and hold never carry it (LONG-05)
DRIVE_OFF_BYTE = 9
DRIVE_OFF_MASK = 0xE0
INACTIVE_MODES = (MODE_START, MODE_ACC_OFF, MODE_STANDBY, MODE_GAS_OVERRIDE)


class CanBus(CanBusBase):
  def __init__(self, CP=None, fingerprint=None) -> None:
    super().__init__(CP, fingerprint)

  @property
  def main(self) -> int:
    return self.offset

  @property
  def radar(self) -> int:
    return self.offset + 1

  @property
  def camera(self) -> int:
    return self.offset + 2


def _update_fields(packer, name, stock_data: bytes, values):
  """Pack only owned fields, leaving every other bit of the original frame intact."""
  msg = packer.dbc.name_to_msg[name]
  if len(stock_data) != msg.size:
    raise ValueError(f"{name}: expected {msg.size} bytes, got {len(stock_data)}")
  encoded = packer.make_can_msg(name, 0, values)[1]
  mask = bytearray(msg.size)
  for field in values:
    signal = msg.sigs[field]
    set_value(mask, signal, (1 << signal.size) - 1)
  return bytes((original & ~bits) | (new & bits) for original, new, bits in zip(stock_data, encoded, mask, strict=True))


def _get_value(packer, name, stock_data, field):
  signal = packer.dbc.name_to_msg[name].sigs[field]
  value = signal.get_raw_value(stock_data)
  if signal.is_signed and value & (1 << (signal.size - 1)):
    value -= 1 << signal.size
  return value * signal.factor + signal.offset


def create_steer_command(packer, CAN: CanBus, camera_stock_data: bytes, steer: float, steer_req: bool):
  steer = int(steer)
  values = {
    "STEER_REQUEST": int(steer_req),
    "SET_ME_X01": 1,
    "TORQUE_CMD": steer,
    "TORQUE_REFLECTED": -steer,
    "INVERT_DIRECTION": int(steer > 0 and steer_req),
    "COUNTER": (_get_value(packer, "STEER_CMD", camera_stock_data, "COUNTER") + 1) % (STEER_MAX_COUNTER + 1),
  }
  dat = _update_fields(packer, "STEER_CMD", camera_stock_data, values)
  dat = _update_fields(packer, "STEER_CMD", dat, {"BASIC_CHECKSUM": gwm_basic_chksum_for_0x12B(dat)})
  dat = _update_fields(packer, "STEER_CMD", dat, {"CRC_X9B": checksum(dat[9:16], 0x9B)})
  return 0x12B, dat, CAN.main


def create_longitudinal_command(packer, CAN, longitudinal_stock_data: bytes, accel, braking, active, standstill, pedal=False, hold=False,
                                gas_pedal=False, stopped=False):
  values = {s: _get_value(packer, "ACC_CMD", longitudinal_stock_data, s) for s in
            ["BRAKE_OR_GAS_REQ", "STANDSTILL_1", "STANDSTILL_2", "STANDSTILL_3"]}
  brake_cmd = accel_cmd = 0
  if braking and active:
    # the camera always sends these while braking, not only stopped. With 4 and 1 the car used the friction brakes
    # for everything: 10 times the hydraulic pressure of the stock ACC at the same deceleration, and no regen
    # the hold is the same frame with its own mode, as the camera sends it at the end of a stop
    values.update(BRAKE_OR_GAS_REQ=MODE_HOLD if hold else MODE_BRAKE, STANDSTILL_1=int(standstill or hold), STANDSTILL_2=3, STANDSTILL_3=0)
    brake_cmd = (-min(accel, 0.) / CarControllerParams.ACCEL_MIN * (107 - 41)) - 41
  elif active:
    values.update(BRAKE_OR_GAS_REQ=MODE_GAS, STANDSTILL_1=0, STANDSTILL_2=4, STANDSTILL_3=1)
    accel_cmd = np.interp(accel, ACC_GAS_ACCEL, ACC_GAS)
  elif stopped and values["BRAKE_OR_GAS_REQ"] == MODE_HOLD:
    # the camera's own hold of a stopped car stays, as the canal always did: let go of, a car stopped on a climb would roll back.
    # With the gas pedal too: the camera lets go of it by itself a moment after the pedal (0.04 to 0.61 s in 107 starts)
    pass
  elif pedal and values["BRAKE_OR_GAS_REQ"] not in INACTIVE_MODES:
    # the camera is a moment behind the pedal: never a frame of gas between our brake and the driver's
    values.update(BRAKE_OR_GAS_REQ=MODE_ACC_OFF, STANDSTILL_1=0, STANDSTILL_2=3, STANDSTILL_3=0)
  elif values["BRAKE_OR_GAS_REQ"] not in INACTIVE_MODES:
    # openpilot not in control: the camera's gas, brake or hold never goes out (it sends them for a few frames around
    # each engage). What the camera sends with the driver on the gas, or standing by
    values.update(BRAKE_OR_GAS_REQ=MODE_GAS_OVERRIDE if gas_pedal else MODE_STANDBY)
  values.update(BRAKE_CMD=brake_cmd, GAS_CMD=accel_cmd)
  dat = _update_fields(packer, "ACC_CMD", longitudinal_stock_data, values)
  if values["BRAKE_OR_GAS_REQ"] in (MODE_BRAKE, MODE_HOLD):
    dat = bytearray(dat)
    dat[DRIVE_OFF_BYTE] &= ~DRIVE_OFF_MASK & 0xFF
    dat = bytes(dat)
  dat = _update_fields(packer, "ACC_CMD", dat, {
    "CRC_BRAKE_0xEF": checksum(dat[9:16], 0xEF),
    "CRC_ACC_0x87": checksum(dat[25:32], 0x87),
  })
  return 0x143, dat, CAN.main


def create_wheel_touch(packer, CAN: CanBus, eps_stock_data: bytes, ea_simulated_torque: float):
  # The first EPS block, its CRC, fault flags, counters and unknown bits pass through unchanged.
  dat = _update_fields(packer, "RX_STEER_RELATED", eps_stock_data, {"B_RX_DRIVER_TORQUE": ea_simulated_torque})
  dat = _update_fields(packer, "RX_STEER_RELATED", dat, {"B_CRC_X61": checksum(dat[9:16], 0x61)})
  return 0x147, dat, CAN.camera


def _create_stalk_command(packer, CAN: CanBus, stock_data: bytes, button: str):
  # This button and no other, whatever the driver is pressing: the panda blocks anything else.
  values = {
    "AP_CANCEL_COMMAND": 0,
    "AP_ENABLE_COMMAND": 0,
    "AP_INCREASE_SPEED_COMMAND": 0,
    "AP_DECREASE_SPEED_COMMAND": 0,
    "AP_INCREASE_DISTANCE_COMMAND": 0,
    "AP_REDUCE_DISTANCE_COMMAND": 0,
    "COUNTER": (_get_value(packer, "STEER_AND_AP_STALK", stock_data, "COUNTER") + 1) % (STALK_MAX_COUNTER + 1),
  }
  dat = _update_fields(packer, "STEER_AND_AP_STALK", stock_data, values | {button: 1})
  dat = _update_fields(packer, "STEER_AND_AP_STALK", dat, {"CRC_X2D": checksum(dat[1:8], 0x2D)})
  return 0xA1, dat, CAN.camera


def create_cancel_command(packer, CAN: CanBus, stock_data: bytes):
  return _create_stalk_command(packer, CAN, stock_data, "AP_CANCEL_COMMAND")


def create_resume_command(packer, CAN: CanBus, stock_data: bytes):
  return _create_stalk_command(packer, CAN, stock_data, "AP_ENABLE_COMMAND")


def create_minus_command(packer, CAN: CanBus, stock_data: bytes):
  return _create_stalk_command(packer, CAN, stock_data, "AP_DECREASE_SPEED_COMMAND")


def create_plus_command(packer, CAN: CanBus, stock_data: bytes):
  return _create_stalk_command(packer, CAN, stock_data, "AP_INCREASE_SPEED_COMMAND")


def create_hud_command(packer, CAN: CanBus, hud_stock_data: bytes, steer_required):
  # With no requested change, still refresh the same CRC as the original implementation.
  values = {"LKAS_STATE": 5} if steer_required else {}
  dat = _update_fields(packer, "LATERAL_STATE", hud_stock_data, values)
  dat = _update_fields(packer, "LATERAL_STATE", dat, {"CRC_X66": checksum(dat[17:24], 0x66)})
  return 0x23D, dat, CAN.main


def checksum(data, xor_output):
  crc = 0
  poly = 0x1D
  for byte in data:
    crc ^= byte
    for _ in range(8):
      crc = ((crc << 1) ^ poly) if (crc & 0x80) else (crc << 1)
      crc &= 0xFF
  return crc ^ xor_output


def gwm_basic_chksum_for_0x12B(d: bytearray) -> int:
  invert_direction = d[12] >> 7 & 0x1
  counter = d[15] & 0xF
  steer_requested = d[15] >> 5 & 0x1
  return (28 - (steer_requested * 8) - counter - invert_direction) & 0x1F
