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
#define GWM_PLUS_MAX_FRAMES     104U  // 8 presses, 40 km/h up. Only to the road's limit, and never past GWM_PLUS_MAX_KPH
#define GWM_PLUS_QUIET_US    60000000U  // as the minus: any minute holds 8 presses at most
#define GWM_PLUS_MAX_KPH         110U  // the ACC's set speed: at or above this, no plus from openpilot (Brazil's highest limit)
#define GWM_ACC_DRIVING            3U  // CRUISE_STATE_2 with the ACC in control

#define GWM_ACC_OFF                1U  // CRUISE_STATE_2 with the ACC off
#define GWM_ACC_OFF_FRAMES         5U  // 0x2AB frames (10 Hz) with the ACC off while engaged: it dropped out, so do we

// BRAKE_OR_GAS_REQ of 0x143 [log]: 0 for a frame at start, 4 with the ACC off, 8 standing by, 12 on the gas, 13 on the brake,
// 20 with the driver on the gas, 25 holding the car at a stop. The camera sends 12, 13 and 25 for a few frames with openpilot
// disengaged too: the mode decides what the PCM does, so with openpilot not in control only the first four go out
#define GWM_MODE_START             0U
#define GWM_MODE_ACC_OFF           4U
#define GWM_MODE_STANDBY           8U
#define GWM_MODE_GAS              12U
#define GWM_MODE_BRAKE            13U
#define GWM_MODE_GAS_OVERRIDE     20U
#define GWM_MODE_HOLD             25U

#define GWM_BRAKE_ZERO              41  // the brake field with no braking in it
#define GWM_BRAKE_HANDOFF_US    200000U  // a pedal takes about this to build pressure, and the stock ACC brakes on for 0.1 s

// always-on lateral: the brake pauses openpilot longitudinal only, steering stays
// engaged until the cancel button. Longitudinal resumes on the next lever pull.
static bool gwm_always_on_lateral = false;
static bool gwm_long_allowed = false;
static bool gwm_brake_pressed_prev = false;

// brake handoff: when the pedal takes over from a brake of ours, that brake may stay for a moment. Only coming down, never
// with gas, once per press. Letting go of it at once takes the braking away before the pedal has built any
static int gwm_brake_last = 0;  // what the last frame that went out asked for
static bool gwm_handoff = false;
static int gwm_handoff_brake = 0;
static uint32_t gwm_handoff_ts = 0U;
static uint32_t gwm_mode_last = GWM_MODE_START;  // the mode of the last frame that went out
static uint32_t gwm_handoff_mode = GWM_MODE_START;  // and the one when the pedal came: the handoff keeps it, or eases to the brake

// the ACC off with openpilot engaged: it dropped out, or never took the pull (the stock ACC driving)
static uint32_t gwm_acc_off_frames = 0U;

// auto resume: one pull of the stalk per stop, with openpilot engaged and no pedal or button in use
static bool gwm_auto_resume = false;
static bool gwm_stalk_button = false;
static uint32_t gwm_resume_frames = 0U;

// the minus button of the ACC, with a budget of its own
static bool gwm_acc_minus = false;
static uint32_t gwm_minus_frames = 0U;
static uint32_t gwm_minus_last_ts = 0U;

// the plus button of the ACC, to the road's limit (CRUISE-01): a budget of its own, only with the ACC driving below a ceiling
static bool gwm_acc_plus = false;
static uint32_t gwm_plus_frames = 0U;
static uint32_t gwm_plus_last_ts = 0U;
static bool gwm_acc_seen = false;  // a 0x2AB from the camera since the session began
static uint32_t gwm_set_speed = 0U;  // the ACC's set speed, km/h (ACC_SPEED_SELECTION)
static uint32_t gwm_acc_state = 0U;  // CRUISE_STATE_2

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
  // the ACC's own state and set speed, as the camera sends them: the plus needs both
  if ((msg->bus == GWM_CAMERA_BUS) && (msg->addr == GWM_CRUISE)) {
    gwm_set_speed = msg->data[22];
    gwm_acc_state = ((uint32_t)msg->data[18] >> 3) & 0x7U;
    gwm_acc_seen = true;
    // the ACC off for 5 frames with openpilot engaged ends it, as the car port does. Not with always-on lateral: it steers
    // without the ACC by design, and its longitudinal pauses on the brake
    bool acc_off = gwm_acc_state == GWM_ACC_OFF;
    gwm_acc_off_frames = (acc_main_on && acc_off && !gwm_always_on_lateral) ? (gwm_acc_off_frames + 1U) : 0U;
    if (gwm_acc_off_frames >= GWM_ACC_OFF_FRAMES) {
      acc_main_on = false;
      gwm_long_allowed = false;
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
      if (vehicle_moving) {
        gwm_resume_frames = 0U;
      }
    }

    if (msg->addr == GWM_BRAKE) {
      bool pressed = GET_BIT(msg, 11U);
      bool long_on = get_longitudinal_allowed() && (!gwm_always_on_lateral || gwm_long_allowed);
      if (pressed && !gwm_brake_pressed_prev && long_on && (gwm_brake_last >= GWM_BRAKE_ZERO)) {
        gwm_handoff = true;
        gwm_handoff_brake = gwm_brake_last;
        gwm_handoff_ts = microsecond_timer_get();
        gwm_handoff_mode = gwm_mode_last;
      }
      if (!pressed) {
        gwm_handoff = false;
      }
      if (gwm_always_on_lateral) {
        // keep the generic brake disengage from dropping lateral, pause longitudinal instead
        if (pressed && (!gwm_brake_pressed_prev || vehicle_moving)) {
          gwm_long_allowed = false;
        }
      } else {
        brake_pressed = pressed;
      }
      gwm_brake_pressed_prev = pressed;
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
        // and gives the ACC its time again
        gwm_acc_off_frames = 0U;
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
    .min_gas = -192,  // the stock ACC asks for less engine torque down to here
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
      int gas_raw = ((msg->data[27] & 0x1FU) << 8) | (msg->data[28]);
      gas_raw = gas_raw - 192;
      violation |= longitudinal_gas_checks(gas_raw, GWM_LONG_LIMITS);

      uint32_t mode = msg->data[9] & 0x1FU;
      bool inactive_mode = (mode == GWM_MODE_START) || (mode == GWM_MODE_ACC_OFF) || (mode == GWM_MODE_STANDBY) ||
                           (mode == GWM_MODE_GAS_OVERRIDE);
      bool brake_mode = (mode == GWM_MODE_BRAKE) || (mode == GWM_MODE_HOLD);
      // gas only in the gas mode, brake only in the brake and hold modes: never both
      violation |= (gas_raw != GWM_LONG_LIMITS.inactive_gas) && (mode != GWM_MODE_GAS);
      violation |= (brake_raw != 0) && !brake_mode;

      gwm_handoff = gwm_handoff && !gas_pressed && (safety_get_ts_elapsed(microsecond_timer_get(), gwm_handoff_ts) < GWM_BRAKE_HANDOFF_US);
      // the handoff keeps a brake that only comes down, in the brake mode, or in the hold if it was holding
      bool handoff_mode = (mode == GWM_MODE_BRAKE) || inactive_mode || ((mode == GWM_MODE_HOLD) && (gwm_handoff_mode == GWM_MODE_HOLD));
      bool handoff = gwm_handoff && handoff_mode && (gas_raw == GWM_LONG_LIMITS.inactive_gas) && (brake_raw >= 0) &&
                     (brake_raw <= gwm_handoff_brake);
      if (handoff) {
        gwm_handoff_brake = brake_raw;
      } else {
        violation |= longitudinal_brake_checks(brake_raw, GWM_LONG_LIMITS);

        // longitudinal paused by the brake in always-on lateral: only inactive commands
        bool long_paused = gwm_always_on_lateral && !gwm_long_allowed;
        violation |= long_paused && ((brake_raw != 0) || (gas_raw != GWM_LONG_LIMITS.inactive_gas));

        // without openpilot in control, only the modes that don't actuate: a hold with a zero brake still brakes. The one
        // exception is the camera's own hold of a stopped car, which goes on as it does today: let go of, a car stopped on a
        // climb would roll back. With the gas pedal too: the camera lets go of it by itself 0.04 to 0.61 s after the pedal
        // (107 starts from a stop, segurar_com_gas.py). Never moving, and never with a brake of ours
        bool long_on = get_longitudinal_allowed() && !long_paused;
        bool stopped_hold = (mode == GWM_MODE_HOLD) && !vehicle_moving;
        violation |= !inactive_mode && !stopped_hold && !(long_on && ((mode == GWM_MODE_GAS) || brake_mode));
      }
      gwm_brake_last = violation ? 0 : brake_raw;
      gwm_mode_last = violation ? GWM_MODE_STANDBY : mode;
    }
  }

  // the stalk only goes out to cancel, to resume from a stop, or for one step of the set speed: no distance button
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
    bool plus = ((msg->data[5] & 0xF0U) == 0U) && ((msg->data[6] & 0x0CU) == 0x08U);
    if (plus && (safety_get_ts_elapsed(microsecond_timer_get(), gwm_plus_last_ts) >= GWM_PLUS_QUIET_US)) {
      gwm_plus_frames = 0U;
    }
    bool acc_below_ceiling = gwm_acc_seen && (gwm_acc_state == GWM_ACC_DRIVING) && (gwm_set_speed < GWM_PLUS_MAX_KPH);
    bool plus_allowed = plus && gwm_acc_plus && controls_allowed && gwm_long_allowed && vehicle_moving && driver_clear &&
                        acc_below_ceiling && (gwm_plus_frames < GWM_PLUS_MAX_FRAMES);
    violation |= !cancel && !resume_allowed && !minus_allowed && !plus_allowed;
    if (resume_allowed) {
      gwm_resume_frames += 1U;
    }
    if (minus_allowed) {
      gwm_minus_frames += 1U;
      gwm_minus_last_ts = microsecond_timer_get();
    }
    if (plus_allowed) {
      gwm_plus_frames += 1U;
      gwm_plus_last_ts = microsecond_timer_get();
    }
    if (cancel || resume_allowed || minus_allowed || plus_allowed) {
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
    {.msg = {{GWM_GAS, GWM_MAIN_BUS, 64, 50U, .max_counter = 14U, .ignore_quality_flag = true}, { 0 }, { 0 }}},   // gas pedal
    {.msg = {{GWM_BRAKE, GWM_MAIN_BUS, 64, 50U, .max_counter = 14U, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // brake2
    {.msg = {{GWM_RX_STEER_RELATED, GWM_MAIN_BUS, 64, 50U, .max_counter = 14U, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // eps feedback to camera
    {.msg = {{GWM_STEER_CMD, GWM_CAMERA_BUS, 64, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // copy stock steering cmd
    {.msg = {{GWM_CRUISE, GWM_CAMERA_BUS, 64, 10U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // CRUISE_STATE, ACC
    {.msg = {{GWM_LONG_CONTROL, GWM_CAMERA_BUS, 64, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // Longitudinal control message from camera
    {.msg = {{GWM_BLIND_SPOT, GWM_MAIN_BUS, 64, 50U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // Blind spot monitor
    {.msg = {{GWM_HUD, GWM_CAMERA_BUS, 64, 20U, .ignore_checksum = true, .ignore_counter = true, .ignore_quality_flag = true}, { 0 }, { 0 }}}, // HUD and dashboard
  };

  const uint32_t FLAG_GWM_ALWAYS_ON_LATERAL = 2U;
  const uint32_t FLAG_GWM_AUTO_RESUME = 4U;
  const uint32_t FLAG_GWM_ACC_MINUS = 8U;
  const uint32_t FLAG_GWM_ACC_PLUS = 16U;
  gwm_always_on_lateral = GET_FLAG(param, FLAG_GWM_ALWAYS_ON_LATERAL);
  gwm_auto_resume = GET_FLAG(param, FLAG_GWM_AUTO_RESUME);
  gwm_acc_minus = GET_FLAG(param, FLAG_GWM_ACC_MINUS);
  gwm_acc_plus = GET_FLAG(param, FLAG_GWM_ACC_PLUS);
  gwm_long_allowed = false;
  gwm_brake_pressed_prev = false;
  gwm_brake_last = 0;
  gwm_handoff = false;
  gwm_handoff_brake = 0;
  gwm_handoff_ts = 0U;
  gwm_mode_last = GWM_MODE_START;
  gwm_handoff_mode = GWM_MODE_START;
  gwm_acc_off_frames = 0U;
  gwm_stalk_button = false;
  gwm_resume_frames = 0U;
  gwm_minus_frames = 0U;
  gwm_minus_last_ts = 0U;
  gwm_plus_frames = 0U;
  gwm_plus_last_ts = 0U;
  gwm_acc_seen = false;
  gwm_set_speed = 0U;
  gwm_acc_state = 0U;
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
