#pragma once

#include "opendbc/safety/declarations.h"

#define GWM_ADAS_ACTIVATION      0xA1U // RX from STEER_AND_AP_STALK
#define GWM_GAS                  0x60U // RX from CAR_OVERALL_SIGNALS
#define GWM_BRAKE               0x120U // RX from BRAKE2
#define GWM_SPEED               0x13BU // RX from WHEEL_SPEEDS
#define GWM_RX_STEER_RELATED    0x147U // RX from EPS to CAMERA
#define GWM_STEER_CMD           0x12BU // TX from OP to EPS
#define GWM_CRUISE              0x2ABU // RX from the camera: the ACC's state
#define GWM_ACC_CMD             0x143U // RX from the camera to the PCM, forwarded untouched
#define GWM_BLIND_SPOT          0x16FU
#define GWM_HUD                 0x23DU

// CAN bus
#define GWM_MAIN_BUS 0U
#define GWM_CAMERA_BUS  2U

#define GWM_STALK_HOLD_US     15000U  // the original stalk frame stays out this long after each one of ours
#define GWM_ACC_OFF               1U  // CRUISE_STATE_2 with the ACC off
#define GWM_ACC_OFF_FRAMES        5U  // 0x2AB frames (10 Hz) with the ACC off while engaged: it dropped out, so do we

static bool gwm_stalk_button = false;  // the driver is on a stalk button
static uint32_t gwm_acc_off_frames = 0U;

// our stalk frame goes to the camera alone: the original stays out right after each one
static bool gwm_stalk_tx_seen = false;
static uint32_t gwm_stalk_tx_ts = 0U;

// Each protected block is 8 bytes: the CRC in its first byte, the counter in the low nibble of its last.
// The gas pedal is in the second block of 0x60 and the EPS torque in the second block of 0x147; the others are in the
// first block of their message.
static int gwm_block(const CANPacket_t *msg) {
  return ((msg->addr == GWM_GAS) || (msg->addr == GWM_RX_STEER_RELATED)) ? 8 : 0;
}

static bool gwm_checked(const CANPacket_t *msg) {
  return (msg->addr == GWM_SPEED) || (msg->addr == GWM_ADAS_ACTIVATION) || (msg->addr == GWM_GAS) || (msg->addr == GWM_BRAKE) ||
         (msg->addr == GWM_RX_STEER_RELATED);
}

static uint8_t gwm_get_counter(const CANPacket_t *msg) {
  uint8_t cnt = 0;
  if (gwm_checked(msg)) {
    cnt = msg->data[gwm_block(msg) + 7] & 0xFU;
  } else {
  }
  return cnt;
}

static uint32_t gwm_get_checksum(const CANPacket_t *msg) {
  uint8_t chksum = 0;
  if (gwm_checked(msg)) {
    chksum = msg->data[gwm_block(msg)] & 0xFFU;
  } else {
  }
  return chksum;
}

static uint32_t gwm_compute_checksum(const CANPacket_t *msg) {
  uint8_t chksum = 0;
  uint8_t crc = 0x00;
  const uint8_t poly = 0x1D;
  uint8_t xor_out = 0x00;
  int block = gwm_block(msg);
  int len = 8;
  for (int i = 1; i < len; i++) {
    uint8_t byte = msg->data[block + i];
    crc ^= byte;
    for (int bit = 0; bit < 8; bit++) {
      if ((crc & 0x80U) != 0U) {
        crc = (crc << 1) ^ poly;
      } else {
        crc <<= 1;
      }
      crc &= 0xFFU;
    }
  }
  if (msg->addr == GWM_ADAS_ACTIVATION) {
    xor_out = 0x2DU;
  } else if (msg->addr == GWM_SPEED) {
    xor_out = 0x7FU;
  } else if (msg->addr == GWM_GAS) {
    xor_out = 0x95U;
  } else if (msg->addr == GWM_BRAKE) {
    xor_out = 0xEEU;
  } else if (msg->addr == GWM_RX_STEER_RELATED) {
    xor_out = 0x61U;
  } else {
  }
  chksum = crc ^ xor_out;
  return chksum;
}

static void gwm_rx_hook(const CANPacket_t *msg) {
  // the ACC off for a while with openpilot engaged: the ACC dropped out, or never took the pull. Only counted while engaged:
  // the ACC is still off for a moment when the pull engages
  if ((msg->bus == GWM_CAMERA_BUS) && (msg->addr == GWM_CRUISE)) {
    bool acc_off = (((uint32_t)msg->data[18] >> 3) & 0x7U) == GWM_ACC_OFF;
    gwm_acc_off_frames = (acc_main_on && acc_off) ? (gwm_acc_off_frames + 1U) : 0U;
    if (gwm_acc_off_frames >= GWM_ACC_OFF_FRAMES) {
      acc_main_on = false;
      gwm_acc_off_frames = 0U;
    }
  }

  if (msg->bus == GWM_MAIN_BUS) {
    // GAS_POSITION
    if (msg->addr == GWM_GAS) {
      gas_pressed = msg->data[9] > 0U;
    }

    if (msg->addr == GWM_SPEED) {
      uint32_t fl = ((msg->data[1] << 8) | msg->data[2]) & 0x1FFFU;
      uint32_t fr = ((msg->data[3] << 8) | msg->data[4]) & 0x1FFFU;
      uint32_t rl = ((msg->data[41] << 8) | msg->data[42]) & 0x1FFFU;
      uint32_t rr = ((msg->data[43] << 8) | msg->data[44]) & 0x1FFFU;
      float speed = (float)((fr + rr + rl + fl) / 4.0f * 0.05924739f * KPH_TO_MS);
      vehicle_moving = speed > 0.0f;
      UPDATE_VEHICLE_SPEED(speed);
    }

    if (msg->addr == GWM_BRAKE) {
      brake_pressed = GET_BIT(msg, 11U);
    }

    if (msg->addr == GWM_RX_STEER_RELATED) {
      int torque_meas_new = ((msg->data[13] & 0x7U) << 8) | (msg->data[14]);
      torque_meas_new = to_signed(torque_meas_new, 11) + 548;
      update_sample(&torque_meas, torque_meas_new);

      // increase torque_meas by 1 to be conservative on rounding
      torque_meas.min--;
      torque_meas.max++;
    }

    // state machine to enter and exit controls for button enabling
    if (msg->addr == GWM_ADAS_ACTIVATION) {
      gwm_stalk_button = ((msg->data[5] & 0xF0U) != 0U) || ((msg->data[6] & 0x0CU) != 0U);
      bool cruise_button = GET_BIT(msg, 47U);
      // enter controls on the rising edge of set or resume
      if (cruise_button && !cruise_button_prev) {
        acc_main_on = true;
        // every pull is a new engage: openpilot may have dropped out with no cancel or brake to clear acc_main_on
        cruise_engaged_prev = false;
        // and gives the ACC its time again
        gwm_acc_off_frames = 0U;
      }
      // exit controls once cancel is pressed
      bool cancel_button = GET_BIT(msg, 46U);
      if (cancel_button || brake_pressed) {
        acc_main_on = false;
      }
      pcm_cruise_check(acc_main_on);
      cruise_button_prev =  cruise_button ? 1 : 0;
    }
  }
}

static bool gwm_tx_hook(const CANPacket_t *msg) {
  const TorqueSteeringLimits GWM_TORQUE_STEERING_LIMITS = {
    .max_torque = 253,
    .max_rate_up = 4,
    .max_rate_down = 6,
    .max_torque_error = 80,
    .max_rt_delta = 100,
    .type = TorqueMotorLimited,
  };

  bool tx = true;
  bool violation = false;

  if (msg->bus == GWM_MAIN_BUS) {
    if (msg->addr == GWM_STEER_CMD) {
      int desired_torque = (((msg->data[12] & 0x7FU) << 3) | ((msg->data[13] & 0xE0U) >> 5));
      desired_torque = to_signed(desired_torque, 10) + 1;
      bool steer_req = GET_BIT(msg, 125U);
      violation |= steer_torque_cmd_checks(desired_torque, steer_req, GWM_TORQUE_STEERING_LIMITS);
      // the other torque fields follow the command, as the camera sends them: TORQUE_REFLECTED is its negative (8 bits,
      // offset -1) and INVERT_DIRECTION is set only for a positive torque that is requested
      uint32_t reflected = (((uint32_t)msg->data[9] & 0x3U) << 6) | ((uint32_t)msg->data[10] >> 2);
      int reflected_torque = 1 - desired_torque;
      uint32_t reflected_expected = ((uint32_t)reflected_torque) & 0xFFU;
      bool invert = GET_BIT(msg, 103U);
      violation |= (reflected != reflected_expected) || (invert != ((desired_torque > 0) && steer_req));
    }
  }

  // the stalk only goes out to cancel: no other button
  if (msg->addr == GWM_ADAS_ACTIVATION) {
    bool other_button = (msg->data[6] & 0x0CU) != 0U;
    bool cancel = ((msg->data[5] & 0xF0U) == 0x40U) && !other_button;
    violation |= !cancel;
    if (cancel) {
      gwm_stalk_tx_seen = true;
      gwm_stalk_tx_ts = microsecond_timer_get();
    }
  }

  if (violation) {
    tx = false;
  }
  return tx;
}

static bool gwm_fwd_hook(int bus_num, int addr) {
  // Our frame alternating with the original stalk frame was never seen working: the camera gets ours alone.
  // It lets go by itself if we stop sending, and a driver's button always goes through.
  bool block_msg = false;
  if ((bus_num == (int)GWM_MAIN_BUS) && (addr == (int)GWM_ADAS_ACTIVATION) && gwm_stalk_tx_seen && !gwm_stalk_button) {
    block_msg = safety_get_ts_elapsed(microsecond_timer_get(), gwm_stalk_tx_ts) < GWM_STALK_HOLD_US;
  }
  return block_msg;
}

static safety_config gwm_init(uint16_t param) {
  static const CanMsg GWM_TX_MSGS[] = {
    {GWM_ADAS_ACTIVATION, GWM_CAMERA_BUS, 8, .check_relay = false}, // Cancel command
    {GWM_RX_STEER_RELATED, GWM_CAMERA_BUS, 64, .check_relay = true}, // EPS steering feedback to camera
    {GWM_STEER_CMD, GWM_MAIN_BUS, 64, .check_relay = true}, // Steering command
    {GWM_HUD, GWM_MAIN_BUS, 64, .check_relay = true}, // HUD and dashboard
  };

  static const RxCheck gwm_rx_checks[] = {
    {.msg = {{GWM_ADAS_ACTIVATION, GWM_MAIN_BUS, 8, 100U, .max_counter = 14U, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // cruise state, steering angle, steer rate
    {.msg = {{GWM_SPEED, GWM_MAIN_BUS, 64, 50U, .max_counter = 14U, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // speed
    {.msg = {{GWM_GAS, GWM_MAIN_BUS, 64, 50U, .max_counter = 14U, .ignore_quality_flag = true}, { 0 }, { 0 }}},   // gas pedal
    {.msg = {{GWM_BRAKE, GWM_MAIN_BUS, 64, 50U, .max_counter = 14U, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // brake2
    {.msg = {{GWM_RX_STEER_RELATED, GWM_MAIN_BUS, 64, 50U, .max_counter = 14U, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // eps feedback to camera
    {.msg = {{GWM_STEER_CMD, GWM_CAMERA_BUS, 64, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // copy stock steering cmd
    {.msg = {{GWM_CRUISE, GWM_CAMERA_BUS, 64, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // CRUISE_STATE, ACC
    {.msg = {{GWM_ACC_CMD, GWM_CAMERA_BUS, 64, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // the camera's ACC command, forwarded
    {.msg = {{GWM_BLIND_SPOT, GWM_MAIN_BUS, 64, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // Blind spot monitor
    {.msg = {{GWM_HUD, GWM_CAMERA_BUS, 64, 20U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // HUD and dashboard
  };

  gwm_stalk_button = false;
  gwm_stalk_tx_seen = false;
  gwm_stalk_tx_ts = 0U;

  gwm_acc_off_frames = 0U;
  SAFETY_UNUSED(param);
  return BUILD_SAFETY_CFG(gwm_rx_checks, GWM_TX_MSGS);
}

const safety_hooks gwm_hooks = {
  .init = gwm_init,
  .rx = gwm_rx_hook,
  .tx = gwm_tx_hook,
  .fwd = gwm_fwd_hook,
  .get_counter = gwm_get_counter,
  .get_checksum = gwm_get_checksum,
  .compute_checksum = gwm_compute_checksum,
};
