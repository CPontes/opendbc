from opendbc.car import CanBusBase
from opendbc.can.packer import set_value

STALK_MAX_COUNTER = 14  # 0xA1 counts 0 to 14, 15 never shows up on the bus
STEER_MAX_COUNTER = 14  # the camera's 0x12B too: 15 was ours, on 1 of every 15 frames

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
