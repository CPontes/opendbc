from collections import deque
import math

import numpy as np
from opendbc.can.packer import CANPacker
from opendbc.car import ACCELERATION_DUE_TO_GRAVITY, Bus, DT_CTRL, structs
from opendbc.car.common.filter_simple import FirstOrderFilter
from opendbc.car.lateral import apply_meas_steer_torque_limits
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.gwm import gwmcan
from opendbc.car.gwm.values import CarControllerParams, GwmFlags

LongCtrlState = structs.CarControl.Actuators.LongControlState
GearShifter = structs.CarState.GearShifter

MAX_USER_TORQUE = 100  # 1.0 Nm
# With the soft handoff, a push under this fades openpilot's torque at STEER_DELTA_DOWN (0.84 s from the top; the panda
# would take it faster) and it comes
# back from where it was. 70% of the pushes between 1.0 and 1.5 Nm lasted under 1 s, a correction: cut, the torque took
# 1.0 to 1.2 s to come back and the wheel fell back 3 to 18 degrees. 79 to 100% of the firmer ones lasted over 1 s,
# taking over: those still let go at once (16 routes, 988 cuts)
SOFT_HANDOFF_MAX_TORQUE = 150  # 1.5 Nm
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
# At the end of a stop the stock ACC goes to its hold mode, at 0.9 km/h and 0.7 s before the wheels stop (median of 90 stops).
# There the car keeps the friction brake it has and adds to it: 415 of pressure when the wheels stop, and it stays there
# whatever is asked, in none of 54 stops under 233. On 13 ours kept 289, but it lost it standing: under 50 within 2 s in 6 of
# 143 stops, with nobody asking to go. Arriving on regen on a 2 degree climb it had 59 and rolled back 17 cm (route 34, 30.28
# min). The stock request climbs to -2.12 in 2.7 s, and it leaves straight to the gas: 79 departures, none rolling back
HOLD_SPEED = 0.25  # m/s
HOLD_ACCEL = -2.12  # m/s^2
HOLD_RAMP_RATE = 0.6  # m/s^3
# Rolling back is a car at a crawl: 0.4 km/h in the one we saw. Faster, the 5 would be a bad reading, and the hold mode is
# never used moving. Once the wheels stop the hold stays a moment, for the pressure to build as at the end of a stock stop
# (0.74 s from the hold mode to the stop): let go at once, a gas that didn't hold the climb would roll it back again
ROLLBACK_MAX_SPEED = 1.  # m/s
ROLLBACK_HOLD_TIME = 1.  # s
# A stop that comes back before the car moves keeps the hold it had. Behind a car creeping at 1 to 2 km/h the planner went to the
# gas for a frame and back to stopping, and the hold started again from nothing: in the hold mode the pressure follows what is
# asked, and it fell from 860 to 0 in 0.3 s with the car still stopped, which then rolled back 13 cm on a climb (route 40, 2.8 min)


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
    self.brake_sent = deque([None] * 3, maxlen=3)  # the panda takes its level at the pedal, frames before openpilot lets go
    self.handoff_left = 0.  # s
    self.handoff_accel = 0.
    self.handoff_hold = False
    self.holding = False
    self.hold_kept: float | None = None  # the hold this stop reached, until the car moves on
    self.rollback_left = 0.  # s
    self.gas_ramp: float | None = None  # how far the gas has climbed since the brake, None when it isn't climbing
    self.pitch = FirstOrderFilter(0., 0.5, DT_CTRL)
    self.resume_prev = False
    self.resume_frames = 0
    self.cancel_frames = 0
    self.cancel_wait = 0
    self.minus_frames = 0
    self.minus_gap = 0
    self.plus_frames = 0
    self.plus_gap = 0
    self.stalk_quiet = MINUS_GAP  # frames since our last stalk frame: a press of one button never right after the other's

  def update(self, CC, CS, now_nanos):
    can_sends = []
    actuators = CC.actuators
    if len(CC.orientationNED) == 3:
      self.pitch.update(CC.orientationNED[1])
    # Only let go when the driver steers against openpilot. In tight curves the driver usually helps in the
    # same direction because openpilot is torque limited, and cutting the torque then left them alone mid-curve.
    driver_opposing = CS.out.steeringTorque * actuators.torque <= 0
    pushing = abs(CS.out.steeringTorque) >= MAX_USER_TORQUE and driver_opposing
    lat_active = CC.latActive and not pushing
    fading = CC.latActive and pushing and bool(self.CP.flags & GwmFlags.SOFT_HANDOFF) and \
             abs(CS.out.steeringTorque) < SOFT_HANDOFF_MAX_TORQUE

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
    self.stalk_quiet = 0 if any(addr == 0xA1 for addr, _, _ in can_sends) else self.stalk_quiet + 1
    minus = CC.cruiseControl.decrease and not CC.cruiseControl.cancel and CS.acc_driving and not CS.out.standstill and CS.stalk_idle
    if not minus or CS.steer_and_ap_stalk_msg is None:
      self.minus_frames = 0
      self.minus_gap = 0
    elif self.minus_gap > 0:
      self.minus_gap -= 1
    elif self.minus_frames == 0 and self.stalk_quiet <= MINUS_GAP:
      pass  # a frame of ours on the stalk just went out: a new press waits the gap
    else:
      self.minus_frames += 1
      can_sends.append(gwmcan.create_minus_command(self.packer, self.CAN, CS.steer_and_ap_stalk_msg))
      if self.minus_frames >= MINUS_FRAMES:
        self.minus_frames = 0
        self.minus_gap = MINUS_GAP

    # The plus, to the road's limit, the same way. Never with a minus asked: the minus wins
    self.stalk_quiet = 0 if any(addr == 0xA1 for addr, _, _ in can_sends) else self.stalk_quiet
    plus = CC.cruiseControl.increase and not CC.cruiseControl.decrease and not CC.cruiseControl.cancel and CS.acc_driving and \
           not CS.out.standstill and CS.stalk_idle
    if not plus or CS.steer_and_ap_stalk_msg is None:
      self.plus_frames = 0
      self.plus_gap = 0
    elif self.plus_gap > 0:
      self.plus_gap -= 1
    elif self.plus_frames == 0 and self.stalk_quiet <= MINUS_GAP:
      pass
    else:
      self.plus_frames += 1
      can_sends.append(gwmcan.create_plus_command(self.packer, self.CAN, CS.steer_and_ap_stalk_msg))
      if self.plus_frames >= MINUS_FRAMES:
        self.plus_frames = 0
        self.plus_gap = MINUS_GAP

    if self.frame % 2 == 0: # 50 Hz
      # Track the actual transmitted request; no original means no replacement request.
      # Between transmit cycles, keep the state of the last emitted command.
      self.lat_active = False
      # Steer command
      new_torque = 0 if fading else int(round(actuators.torque * self.params.STEER_MAX))
      apply_torque = apply_meas_steer_torque_limits(new_torque, self.apply_torque_last, CS.out.steeringTorqueEps, self.params)
      # Prevent sending the same 'apply_torque = 1' torque repeatedly, as it can cause EPS faults.
      if abs(apply_torque) == 1:
        apply_torque = 0 if fading else apply_torque * 2
      # fading out, the request stays until the torque is gone
      steer_req = lat_active or (fading and apply_torque != 0)
      if not steer_req or CS.camera_stock_data is None:
        apply_torque = 0
      if CS.camera_stock_data is not None:
        can_sends.append(gwmcan.create_steer_command(
          self.packer,
          self.CAN,
          camera_stock_data=CS.camera_stock_data,
          steer=apply_torque,
          steer_req=steer_req,
        ))
        self.lat_active = steer_req
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
        # rolling back in D the car is never left on the gas or loose: it holds, and openpilot drives on a moment after it stops
        in_drive = CS.out.gearShifter == GearShifter.drive and not CS.out.gasPressed and CC.longActive
        rolling_back = CS.rolling_back and in_drive and CS.out.vEgo < ROLLBACK_MAX_SPEED
        if rolling_back:
          self.rollback_left = ROLLBACK_HOLD_TIME
        elif not in_drive:
          self.rollback_left = 0.
        else:
          self.rollback_left = max(self.rollback_left - DT_CTRL * self.params.STEER_STEP, 0.)
        rolled_back = rolling_back or self.rollback_left > 0.
        brake_wanted = standstill or rolled_back or self.accel < (brake_off if self.braking else BRAKE_ON)
        self.brake_hold = BRAKE_HOLD_TIME if brake_wanted else max(self.brake_hold - DT_CTRL * self.params.STEER_STEP, 0.)
        brake_held = self.braking and self.brake_hold > 0. and self.accel < BRAKE_RELEASE and CS.out.vEgo > BRAKE_HOLD_MIN_SPEED
        was_braking = self.braking
        self.braking = CC.longActive and (brake_wanted or brake_held)
        if self.braking and not standstill:
          # the slope only chooses the brake: from there the car holds the deceleration by itself on any slope, and adding it again brakes twice
          self.accel = float(np.clip(min(actuators.accel, 0.), CarControllerParams.ACCEL_MIN, 0.))
        # the pedal takes a moment to build pressure: until the car drops the ACC our brake stays, as its own does
        pedal = CS.out.brakePressed and not CS.out.gasPressed
        # the weakest of them: asking for more than the panda saw is refused, and no 0x143 goes out at all
        if pedal and CS.acc_driving and was_braking and not CC.longActive and None not in self.brake_sent:
          self.handoff_left, self.handoff_accel, self.handoff_hold = HANDOFF_TIME, max(self.brake_sent), self.holding
        elif not pedal or not CS.acc_driving or CC.longActive:
          self.handoff_left = 0.
        handing_off = self.handoff_left > 0.
        self.handoff_left = max(self.handoff_left - DT_CTRL * self.params.STEER_STEP, 0.)
        if self.braking and not standstill and self.accel > BRAKE_RAMP_MIN:
          self.accel = max(self.accel, self.brake_accel_last - BRAKE_RAMP_RATE * DT_CTRL * self.params.STEER_STEP)
        # it starts stopping, and stays while there's braking near a stop: a frame out of stopping must not drop it to regen
        near_stop = CS.out.vEgo < HOLD_SPEED
        self.holding = self.braking and (rolled_back or (standstill and (self.holding or near_stop)) or (self.holding and near_stop))
        if self.holding:
          if rolled_back:
            hold = HOLD_ACCEL
          elif CS.out.brakePressed:
            # the panda takes its level at the pedal and refuses more: the hold stops climbing until openpilot lets go
            hold = self.brake_accel_last
          else:
            start = self.brake_accel_last if self.hold_kept is None else min(self.brake_accel_last, self.hold_kept)
            hold = max(start - HOLD_RAMP_RATE * DT_CTRL * self.params.STEER_STEP, HOLD_ACCEL)
          self.accel = min(self.accel, hold)
          self.hold_kept = self.accel if self.hold_kept is None else min(self.hold_kept, self.accel)
        elif not CC.longActive or CS.out.vEgo >= HOLD_SPEED:
          # out of D openpilot is never in control, so that ends it too
          self.hold_kept = None
        self.brake_accel_last = min(self.accel, 0.)
        if self.braking:
          self.gas_ramp = gwmcan.ACC_GAS_ACCEL[0]
        elif self.gas_ramp is not None:
          self.gas_ramp += GAS_RAMP_RATE * DT_CTRL * self.params.STEER_STEP
          if not CC.longActive or self.accel >= GAS_RAMP_MAX or self.gas_ramp >= self.accel or CS.out.vEgo <= BRAKE_HOLD_MIN_SPEED:
            self.gas_ramp = None
          else:
            self.accel = self.gas_ramp
        self.brake_sent.append(self.accel if self.braking else None)
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
          hold=self.holding or (handing_off and self.handoff_hold),
          gas_pedal=CS.out.gasPressed,
          stopped=CS.out.standstill,
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
