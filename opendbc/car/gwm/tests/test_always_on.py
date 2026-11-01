"""Always-on lateral from real CAN frames: the brake pauses longitudinal, the lever resumes it, steering stays engaged."""
from opendbc.car.gwm.tests import test_carstate as base
from opendbc.car.gwm.tests.test_carstate import Car, resumes
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
