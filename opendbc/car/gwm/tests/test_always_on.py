"""Always-on lateral from real CAN frames: the brake pauses longitudinal, the lever resumes it, steering stays engaged."""
import pytest

from opendbc.car.gwm.tests import test_carstate as base
from opendbc.car.gwm.tests.test_agrees_with_panda import random_drive
from opendbc.car.gwm.tests.test_carstate import Car, STALK, resumes
from opendbc.car.gwm.values import GwmSafetyFlags

ALWAYS_ON = GwmSafetyFlags.ALWAYS_ON_LATERAL.value


class TestEngagementAlwaysOn(base.TestEngagement):
  safety_param = ALWAYS_ON


class TestBrakeAlwaysOn:
  def test_brake_keeps_engaged(self):
    car = Car(ALWAYS_ON)
    car.lever()
    ret = car.brake()
    assert ret.brakePressed and ret.cruiseState.enabled
    assert car.step(frames=100).cruiseState.enabled
    assert car.brake(0).cruiseState.enabled

  def test_lever_after_brake_resumes(self):
    car = Car(ALWAYS_ON)
    car.lever()
    car.brake()
    car.brake(0)
    assert not resumes(car.pull())
    ret = car.release()
    assert ret.cruiseState.enabled
    assert [be.pressed for be in resumes(ret)] == [False]
    assert not resumes(car.step())

  def test_lever_with_the_brake_held_also_resumes(self):
    car = Car(ALWAYS_ON)
    car.lever()
    car.brake()
    assert [be.pressed for be in resumes(car.lever())] == [False]

  def test_cancel_still_disengages_after_a_brake_pause(self):
    car = Car(ALWAYS_ON)
    car.lever()
    car.brake()
    assert not car.cancel().cruiseState.enabled

  def test_brake_during_the_pull_still_engages(self):
    # the panda ignores the brake here too
    car = Car(ALWAYS_ON)
    car.brake()
    car.pull()
    car.brake(0)
    assert car.release().cruiseState.enabled

  def test_resume_keeps_its_type_next_to_a_distance_press(self):
    car = Car(ALWAYS_ON, long=True)
    car.lever()
    car.CI.current_personality = 1
    car.step(ACC={"CAR_DISTANCE_SELECTION": 2}, **{STALK: {"AP_ENABLE_COMMAND": 1}})
    ret = car.release()
    assert sorted(str(be.type) for be in ret.buttonEvents) == ["gapAdjustCruise", "resumeCruise"]


@pytest.mark.parametrize("seed", range(20))
def test_never_engaged_with_the_panda_refusing(seed):
  random_drive(ALWAYS_ON, seed)
