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
# Moderate braking builds slowly enough for regen to take it, as the stock ACC does: under 1 m/s^3 in 434 brake
# applications. Harder braking isn't limited
BRAKE_RAMP_RATE = 1.0  # m/s^3
BRAKE_RAMP_MIN = -1.0  # m/s^2


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
        self.braking = CC.longActive and (standstill or self.accel < (BRAKE_OFF if self.braking else BRAKE_ON))
        if self.braking and not standstill and self.accel > BRAKE_RAMP_MIN:
          self.accel = max(self.accel, self.brake_accel_last - BRAKE_RAMP_RATE * DT_CTRL * self.params.STEER_STEP)
        self.brake_accel_last = self.accel if self.braking else min(self.accel, 0.)
        can_sends.append(gwmcan.create_longitudinal_command(
          self.packer,
          self.CAN,
          longitudinal_stock_data=CS.longitudinal_stock_data,
          accel=self.accel,
          braking=self.braking,
          active=CC.longActive,
          standstill=standstill,
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
