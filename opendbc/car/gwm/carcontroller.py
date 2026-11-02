from collections import deque
import math

import numpy as np
from opendbc.can.packer import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, structs
from opendbc.car.common.filter_simple import FirstOrderFilter
from opendbc.car.lateral import apply_meas_steer_torque_limits
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.gwm import gwmcan
from opendbc.car.gwm.values import CarControllerParams

LongCtrlState = structs.CarControl.Actuators.LongControlState

MAX_USER_TORQUE = 100  # 1.0 Nm
RESUME_FRAMES = 13  # at 100 Hz: the driver's pulls last 0.10 to 0.16 s
CANCEL_FRAMES = 13  # the driver's pushes last 0.13 s
CANCEL_WAIT = 100  # frames, before pushing again if the ACC is still driving
MINUS_FRAMES = 13  # the driver's presses last 0.09 to 0.18 s
MINUS_GAP = 35  # frames between presses: the driver's are 0.35 s apart, and the set speed reads back at 10 Hz
# Between these the last of gas or brake stays: a request around zero flipped them 76 times a minute. The Haval's own ACC
# goes to the brake around -0.23 m/s^2, comes back around -0.16 and changes 7 times a minute
BRAKE_ON = -0.25  # m/s^2
BRAKE_OFF = -0.1  # m/s^2
# An eased brake waits on the brake with nothing asked, unless gas is really wanted. Going straight to the gas left the car
# still slowing and asking for more, and the next brake came in under 3 s half of the time: on a mild descent, pulses of
# 0.8 s every 2 s the driver felt as pinching the pedal. The Haval's own ACC stays on the brake for 3 s, and comes back to it
# in under 3 s only 22% of the time
BRAKE_HOLD_TIME = 2.  # s
BRAKE_RELEASE = 0.1  # m/s^2
BRAKE_HOLD_MIN_SPEED = 5.  # m/s, under this it would keep the car from leaving a stop: up to 1.5 s without gas
# Under the same speed a request to slow down keeps the brake, however small. Letting go of it left the car's creep holding it
# at 1.5 km/h: it never got to the stop, and rolled onto the crosswalk until the driver braked
# Out of the brake the gas starts from coasting and climbs, as the stock ACC does (-192, then under 100 a second later). Going
# straight to +230 pushed the car on, and on a mild descent the next brake came in under 3 s half of the time. A request
# for more than GAS_RAMP_MAX is the car really wanted to go, and nothing holds it back
GAS_RAMP_RATE = 0.5  # m/s^3
GAS_RAMP_MAX = 0.4  # m/s^2
# Moderate braking builds slowly enough for regen to take it, as the stock ACC does: under 1 m/s^3 in 434 brake
# applications. Harder braking isn't limited
BRAKE_RAMP_RATE = 1.0  # m/s^3
BRAKE_RAMP_MIN = -1.0  # m/s^2
HANDOFF_TIME = 0.18  # s, under the 0.2 of the panda. The stock ACC brakes on for 0.1 s after the pedal, until it drops
HANDOFF_MIN_BRAKE = -0.06  # m/s^2, the brake field has to be above its zero or the panda doesn't open


class CarController(CarControllerBase):
  def __init__(self, dbc_names, CP):
    super().__init__(dbc_names, CP)
    self.params = CarControllerParams(self.CP)
    self.packer = CANPacker(dbc_names[Bus.main])
    self.apply_torque_last = 0
    self.lat_active = False  # what the EPS was last asked to do, not just what openpilot wants
    self.CAN = gwmcan.CanBus(CP)
    self.accel = 0.0
    self.braking = False
    self.brake_accel_last = 0.0
    self.brake_hold = 0.
    self.brake_sent = deque([0.] * 3, maxlen=3)  # the panda takes its level at the pedal, frames before openpilot lets go
    self.handoff_left = 0.  # s
    self.handoff_accel = 0.
    self.gas_ramp: float | None = None  # how far the gas has climbed since the brake, None when it isn't climbing
    self.pitch = FirstOrderFilter(0., 0.5, DT_CTRL)
    self.resume_prev = False
    self.resume_frames = 0
    self.cancel_frames = 0
    self.cancel_wait = 0
    self.minus_frames = 0
    self.minus_gap = 0

  def update(self, CC, CS, now_nanos):
    can_sends = []
    actuators = CC.actuators
    if len(CC.orientationNED) == 3:
      self.pitch.update(CC.orientationNED[1])
    # Only let go when the driver steers against openpilot. In tight curves the driver usually helps in the
    # same direction because openpilot is torque limited, and cutting the torque then left them alone mid-curve.
    driver_opposing = CS.out.steeringTorque * actuators.torque <= 0
    lat_active = CC.latActive and not (abs(CS.out.steeringTorque) >= MAX_USER_TORQUE and driver_opposing)

    # Increment counter so cancel is prioritized even without openpilot longitudinal
    # A push like the driver's, only with the ACC driving a moving car. At a stop it holds the car and only leaves
    # on a pull, and what a cancel does there was never seen. The panda keeps the original frame out meanwhile.
    cancel = CC.cruiseControl.cancel and CS.acc_driving and not CS.out.standstill and CS.steer_and_ap_stalk_msg is not None
    if not cancel:
      self.cancel_frames = 0
      self.cancel_wait = 0
    elif self.cancel_wait > 0:
      self.cancel_wait -= 1
    else:
      self.cancel_frames += 1
      can_sends.append(gwmcan.create_cancel_command(self.packer, self.CAN, CS.steer_and_ap_stalk_msg))
      if self.cancel_frames >= CANCEL_FRAMES:
        self.cancel_frames = 0
        self.cancel_wait = CANCEL_WAIT

    # One pull per request, like the driver's, only with the ACC waiting at a stop. The panda has the last word.
    resume = CC.cruiseControl.resume and not CC.cruiseControl.cancel
    if resume and not self.resume_prev and CS.acc_waiting:
      self.resume_frames = RESUME_FRAMES
    self.resume_prev = resume
    if not resume or not CS.stalk_idle or CS.steer_and_ap_stalk_msg is None:
      self.resume_frames = 0
    if self.resume_frames > 0:
      self.resume_frames -= 1
      can_sends.append(gwmcan.create_resume_command(self.packer, self.CAN, CS.steer_and_ap_stalk_msg))

    # One press at a time, only with the ACC driving a moving car and nobody on the stalk. Who asks reads the set speed back.
    minus = CC.cruiseControl.decrease and not CC.cruiseControl.cancel and CS.acc_driving and not CS.out.standstill and CS.stalk_idle
    if not minus or CS.steer_and_ap_stalk_msg is None:
      self.minus_frames = 0
      self.minus_gap = 0
    elif self.minus_gap > 0:
      self.minus_gap -= 1
    else:
      self.minus_frames += 1
      can_sends.append(gwmcan.create_minus_command(self.packer, self.CAN, CS.steer_and_ap_stalk_msg))
      if self.minus_frames >= MINUS_FRAMES:
        self.minus_frames = 0
        self.minus_gap = MINUS_GAP

    if self.frame % 2 == 0: # 50 Hz
      # Track the actual transmitted request; no original means no replacement request.
      # Between transmit cycles, keep the state of the last emitted command.
      self.lat_active = False
      # Steer command
      new_torque = int(round(actuators.torque * self.params.STEER_MAX))
      apply_torque = apply_meas_steer_torque_limits(new_torque, self.apply_torque_last, CS.out.steeringTorqueEps, self.params)
      # Prevent sending the same 'apply_torque = 1' torque repeatedly, as it can cause EPS faults.
      if abs(apply_torque) == 1:
        apply_torque = apply_torque * 2
      if not lat_active or CS.camera_stock_data is None:
        apply_torque = 0
      if CS.camera_stock_data is not None:
        can_sends.append(gwmcan.create_steer_command(
          self.packer,
          self.CAN,
          camera_stock_data=CS.camera_stock_data,
          steer=apply_torque,
          steer_req=lat_active,
        ))
        self.lat_active = lat_active
      self.apply_torque_last = apply_torque

      # Satisfy steer nudge requests
      ea_simulated_torque = float(np.clip(apply_torque * 2, -self.params.STEER_MAX, self.params.STEER_MAX))
      if abs(CS.out.steeringTorque) > abs(ea_simulated_torque):
        ea_simulated_torque = CS.out.steeringTorque
      if CS.eps_stock_data is not None:
        can_sends.append(gwmcan.create_wheel_touch(
          self.packer,
          self.CAN,
          eps_stock_data=CS.eps_stock_data,
          ea_simulated_torque=ea_simulated_torque,
        ))

      # Longitudinal control
      if self.CP.openpilotLongitudinalControl and CS.longitudinal_stock_data is not None:
        standstill = actuators.longControlState == LongCtrlState.stopping
        accel = actuators.accel
        slope = math.sin(self.pitch.x) * ACCELERATION_DUE_TO_GRAVITY
        # the slope, before the car has to speed up or slow down to show it. A climb never takes braking away
        if actuators.longControlState == LongCtrlState.pid and (slope < 0 or accel > 0):
          accel += slope
        self.accel = float(np.clip(accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
        brake_off = 0. if CS.out.vEgo <= BRAKE_HOLD_MIN_SPEED and actuators.accel < 0. else BRAKE_OFF
        brake_wanted = standstill or self.accel < (brake_off if self.braking else BRAKE_ON)
        self.brake_hold = BRAKE_HOLD_TIME if brake_wanted else max(self.brake_hold - DT_CTRL * self.params.STEER_STEP, 0.)
        brake_held = self.braking and self.brake_hold > 0. and self.accel < BRAKE_RELEASE and CS.out.vEgo > BRAKE_HOLD_MIN_SPEED
        was_braking = self.braking
        self.braking = CC.longActive and (brake_wanted or brake_held)
        if self.braking and not standstill:
          # the slope only chooses the brake: from there the car holds the deceleration by itself on any slope, and adding it again brakes twice
          self.accel = float(np.clip(min(actuators.accel, 0.), CarControllerParams.ACCEL_MIN, 0.))
        # the pedal takes a moment to build pressure: until the car drops the ACC our brake stays, as its own does
        pedal = CS.out.brakePressed and not CS.out.gasPressed
        level = max(self.brake_sent)  # the weakest of them: asking for more than the panda saw is refused, and no 0x143 goes out at all
        if pedal and CS.acc_driving and was_braking and not CC.longActive and level < HANDOFF_MIN_BRAKE:
          self.handoff_left, self.handoff_accel = HANDOFF_TIME, level
        elif not pedal or not CS.acc_driving or CC.longActive:
          self.handoff_left = 0.
        handing_off = self.handoff_left > 0.
        self.handoff_left = max(self.handoff_left - DT_CTRL * self.params.STEER_STEP, 0.)
        if self.braking and not standstill and self.accel > BRAKE_RAMP_MIN:
          self.accel = max(self.accel, self.brake_accel_last - BRAKE_RAMP_RATE * DT_CTRL * self.params.STEER_STEP)
        self.brake_accel_last = min(self.accel, 0.)
        if self.braking:
          self.gas_ramp = gwmcan.ACC_GAS_ACCEL[0]
        elif self.gas_ramp is not None:
          self.gas_ramp += GAS_RAMP_RATE * DT_CTRL * self.params.STEER_STEP
          if not CC.longActive or self.accel >= GAS_RAMP_MAX or self.gas_ramp >= self.accel or CS.out.vEgo <= BRAKE_HOLD_MIN_SPEED:
            self.gas_ramp = None
          else:
            self.accel = self.gas_ramp
        self.brake_sent.append(self.accel if self.braking else 0.)
        if handing_off:
          self.accel = self.handoff_accel
        can_sends.append(gwmcan.create_longitudinal_command(
          self.packer,
          self.CAN,
          longitudinal_stock_data=CS.longitudinal_stock_data,
          accel=self.accel,
          braking=self.braking or handing_off,
          active=CC.longActive or handing_off,
          standstill=standstill,
          pedal=pedal,
        ))

    if self.frame % 5 == 0 and CS.hud_stock_data is not None: # 20 Hz
      # HUD updates
      can_sends.append(gwmcan.create_hud_command(
        self.packer,
        self.CAN,
        hud_stock_data=CS.hud_stock_data,
        steer_required=CC.latActive,
      ))

    new_actuators = actuators.as_builder()
    new_actuators.torque = self.apply_torque_last / self.params.STEER_MAX
    new_actuators.torqueOutputCan = self.apply_torque_last
    new_actuators.accel = self.accel

    self.frame += 1
    return new_actuators, can_sends
