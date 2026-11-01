#pragma once

#include "opendbc/safety/declarations.h"

#define GWM_ADAS_ACTIVATION      0xA1U // RX from STEER_AND_AP_STALK
#define GWM_GAS                  0x60U // RX from CAR_OVERALL_SIGNALS
#define GWM_BRAKE               0x120U // RX from BRAKE2
#define GWM_SPEED               0x13BU // RX from WHEEL_SPEEDS
#define GWM_RX_STEER_RELATED    0x147U // RX from EPS to CAMERA
#define GWM_STEER_CMD           0x12BU // TX from OP to EPS
#define GWM_CRUISE              0x2ABU
#define GWM_LONG_CONTROL        0x143U // TX from OP to PCM
#define GWM_BLIND_SPOT          0x16FU
#define GWM_HUD                 0x23DU

// CAN bus
#define GWM_MAIN_BUS 0U
#define GWM_CAMERA_BUS  2U

#define GWM_STALK_HOLD_US     15000U  // the original stalk frame stays out this long after each one of ours
#define GWM_RESUME_MAX_FRAMES    20U  // per stop. A real pull is 10 to 16 frames
#define GWM_MINUS_MAX_FRAMES     78U  // 6 presses, 30 km/h down. It can't take the car off a road by itself
#define GWM_MINUS_QUIET_US   60000000U  // the budget comes back only after this long with none: any minute holds 6 presses at most

// always-on lateral: the brake pauses openpilot longitudinal only, steering stays
// engaged until the cancel button. Longitudinal resumes on the next lever pull.
static bool gwm_always_on_lateral = false;
static bool gwm_long_allowed = false;
static bool gwm_brake_pressed_prev = false;

// auto resume: one pull of the stalk per stop, with openpilot engaged and no pedal or button in use
static bool gwm_auto_resume = false;
static bool gwm_stalk_button = false;
static uint32_t gwm_resume_frames = 0U;

// the minus button of the ACC: never the plus, and a budget of its own
static bool gwm_acc_minus = false;
static uint32_t gwm_minus_frames = 0U;
static uint32_t gwm_minus_last_ts = 0U;

// our stalk frame goes to the camera alone: the original stays out right after each one
static bool gwm_stalk_tx_seen = false;
static uint32_t gwm_stalk_tx_ts = 0U;

static uint8_t gwm_get_counter(const CANPacket_t *msg) {
  uint8_t cnt = 0;
  if ((msg->addr == GWM_SPEED) || (msg->addr == GWM_ADAS_ACTIVATION)) {
    cnt = msg->data[7] & 0xFU;
  } else {
  }
  return cnt;
}

static uint32_t gwm_get_checksum(const CANPacket_t *msg) {
  uint8_t chksum = 0;
  if ((msg->addr == GWM_SPEED) || (msg->addr == GWM_ADAS_ACTIVATION)) {
    chksum = msg->data[0] & 0xFFU;
  } else {
  }
  return chksum;
}

static uint32_t gwm_compute_checksum(const CANPacket_t *msg) {
  uint8_t chksum = 0;
  uint8_t crc = 0x00;
  const uint8_t poly = 0x1D;
  uint8_t xor_out = 0x00;
  int len = 8;
  for (int i = 1; i < len; i++) {
    uint8_t byte = msg->data[i];
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
  } else {
  }
  chksum = crc ^ xor_out;
  return chksum;
}

static void gwm_rx_hook(const CANPacket_t *msg) {
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
      float speed = (float)((fr + rr + rl + fl) / 4.0f * 0.05924739 * KPH_TO_MS);
      vehicle_moving = speed > 0.0f;
      UPDATE_VEHICLE_SPEED(speed);
      if (vehicle_moving) {
        gwm_resume_frames = 0U;
      }
    }

    if (msg->addr == GWM_BRAKE) {
      bool pressed = GET_BIT(msg, 11U);
      if (gwm_always_on_lateral) {
        // keep the generic brake disengage from dropping lateral, pause longitudinal instead
        if (pressed && (!gwm_brake_pressed_prev || vehicle_moving)) {
          gwm_long_allowed = false;
        }
        gwm_brake_pressed_prev = pressed;
      } else {
        brake_pressed = pressed;
      }
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
        gwm_long_allowed = true;
        // every pull is a new engage: openpilot may have dropped out with no cancel or brake to clear acc_main_on
        cruise_engaged_prev = false;
      }
      // exit controls once cancel is pressed
      bool cancel_button = GET_BIT(msg, 46U);
      if (cancel_button || brake_pressed) {
        acc_main_on = false;
        gwm_long_allowed = false;
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

  const LongitudinalLimits GWM_LONG_LIMITS = {
    .max_gas = 4577,
    .min_gas = -10,
    .inactive_gas = 0,
    .max_brake = 107,
  };

  bool tx = true;
  bool violation = false;

  if (msg->bus == GWM_MAIN_BUS) {
    if (msg->addr == GWM_STEER_CMD) {
      int desired_torque = (((msg->data[12] & 0x7FU) << 3) | ((msg->data[13] & 0xE0U) >> 5));
      desired_torque = to_signed(desired_torque, 10) + 1;
      bool steer_req = GET_BIT(msg, 125U);
      violation |= steer_torque_cmd_checks(desired_torque, steer_req, GWM_TORQUE_STEERING_LIMITS);
    }

    if (msg->addr == GWM_LONG_CONTROL) {
      int brake_raw = msg->data[13];
      brake_raw = 181 - brake_raw;
      violation |= longitudinal_brake_checks(brake_raw, GWM_LONG_LIMITS);

      int gas_raw = ((msg->data[27] & 0x1FU) << 8) | (msg->data[28]);
      gas_raw = gas_raw - 192;
      violation |= longitudinal_gas_checks(gas_raw, GWM_LONG_LIMITS);

      // longitudinal paused by the brake in always-on lateral: only inactive commands
      bool long_paused = gwm_always_on_lateral && !gwm_long_allowed;
      violation |= long_paused && ((brake_raw != 0) || (gas_raw != GWM_LONG_LIMITS.inactive_gas));
    }
  }

  // the stalk only goes out to cancel or to resume from a stop: no speed or distance button
  if (msg->addr == GWM_ADAS_ACTIVATION) {
    bool other_button = (msg->data[6] & 0x0CU) != 0U;
    bool cancel = ((msg->data[5] & 0xF0U) == 0x40U) && !other_button;
    bool resume = ((msg->data[5] & 0xF0U) == 0x80U) && !other_button;
    bool driver_clear = !brake_pressed && !gas_pressed && !gwm_stalk_button;
    bool resume_allowed = resume && gwm_auto_resume && controls_allowed && gwm_long_allowed && !vehicle_moving && driver_clear &&
                          (gwm_resume_frames < GWM_RESUME_MAX_FRAMES);
    bool minus = ((msg->data[5] & 0xF0U) == 0U) && ((msg->data[6] & 0x0CU) == 0x04U);
    if (minus && (safety_get_ts_elapsed(microsecond_timer_get(), gwm_minus_last_ts) >= GWM_MINUS_QUIET_US)) {
      gwm_minus_frames = 0U;
    }
    bool minus_allowed = minus && gwm_acc_minus && controls_allowed && gwm_long_allowed && vehicle_moving && driver_clear &&
                         (gwm_minus_frames < GWM_MINUS_MAX_FRAMES);
    violation |= !cancel && !resume_allowed && !minus_allowed;
    if (resume_allowed) {
      gwm_resume_frames += 1U;
    }
    if (minus_allowed) {
      gwm_minus_frames += 1U;
      gwm_minus_last_ts = microsecond_timer_get();
    }
    if (cancel || resume_allowed || minus_allowed) {
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
    {GWM_ADAS_ACTIVATION, GWM_CAMERA_BUS, 8, .check_relay = false}, // Cancel and resume commands
    {GWM_RX_STEER_RELATED, GWM_CAMERA_BUS, 64, .check_relay = true}, // EPS steering feedback to camera
    {GWM_STEER_CMD, GWM_MAIN_BUS, 64, .check_relay = true}, // Steering command
    {GWM_HUD, GWM_MAIN_BUS, 64, .check_relay = true}, // HUD and dashboard
  };

  static const CanMsg GWM_LONG_TX_MSGS[] = {
    {GWM_ADAS_ACTIVATION, GWM_CAMERA_BUS, 8, .check_relay = false}, // Cancel and resume commands
    {GWM_RX_STEER_RELATED, GWM_CAMERA_BUS, 64, .check_relay = true}, // EPS steering feedback to camera
    {GWM_STEER_CMD, GWM_MAIN_BUS, 64, .check_relay = true}, // Steering command
    {GWM_LONG_CONTROL, GWM_MAIN_BUS, 64, .check_relay = true}, // Longitudinal control message from camera
    {GWM_HUD, GWM_MAIN_BUS, 64, .check_relay = true}, // HUD and dashboard
  };

  static const RxCheck gwm_rx_checks[] = {
    {.msg = {{GWM_ADAS_ACTIVATION, GWM_MAIN_BUS, 8, 100U, .max_counter = 14U, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // cruise state, steering angle, steer rate
    {.msg = {{GWM_SPEED, GWM_MAIN_BUS, 64, 50U, .max_counter = 14U, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // speed
    {.msg = {{GWM_GAS, GWM_MAIN_BUS, 64, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}},   // gas pedal
    {.msg = {{GWM_BRAKE, GWM_MAIN_BUS, 64, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // brake2
    {.msg = {{GWM_RX_STEER_RELATED, GWM_MAIN_BUS, 64, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // eps feedback to camera
    {.msg = {{GWM_STEER_CMD, GWM_CAMERA_BUS, 64, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // copy stock steering cmd
    {.msg = {{GWM_CRUISE, GWM_CAMERA_BUS, 64, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // CRUISE_STATE, ACC
    {.msg = {{GWM_LONG_CONTROL, GWM_CAMERA_BUS, 64, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // Longitudinal control message from camera
    {.msg = {{GWM_BLIND_SPOT, GWM_MAIN_BUS, 64, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // Blind spot monitor
    {.msg = {{GWM_HUD, GWM_CAMERA_BUS, 64, 20U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // HUD and dashboard
  };

  const uint32_t FLAG_GWM_ALWAYS_ON_LATERAL = 2U;
  const uint32_t FLAG_GWM_AUTO_RESUME = 4U;
  const uint32_t FLAG_GWM_ACC_MINUS = 8U;
  gwm_always_on_lateral = GET_FLAG(param, FLAG_GWM_ALWAYS_ON_LATERAL);
  gwm_auto_resume = GET_FLAG(param, FLAG_GWM_AUTO_RESUME);
  gwm_acc_minus = GET_FLAG(param, FLAG_GWM_ACC_MINUS);
  gwm_long_allowed = false;
  gwm_brake_pressed_prev = false;
  gwm_stalk_button = false;
  gwm_resume_frames = 0U;
  gwm_minus_frames = 0U;
  gwm_minus_last_ts = 0U;
  gwm_stalk_tx_seen = false;
  gwm_stalk_tx_ts = 0U;

  bool gwm_longitudinal = false;
#ifdef ALLOW_DEBUG
  const uint32_t FLAG_GWM_LONG_CONTROL = 1U;
  gwm_longitudinal = GET_FLAG(param, FLAG_GWM_LONG_CONTROL);
#endif

  // FIXME: cppcheck thinks that gwm_longitudinal is always false. This is not true
  // if ALLOW_DEBUG is defined but cppcheck is run without ALLOW_DEBUG
  // cppcheck-suppress knownConditionTrueFalse
  return gwm_longitudinal ? BUILD_SAFETY_CFG(gwm_rx_checks, GWM_LONG_TX_MSGS) : \
                            BUILD_SAFETY_CFG(gwm_rx_checks, GWM_TX_MSGS);
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
