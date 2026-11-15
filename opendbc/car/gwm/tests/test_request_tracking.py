import unittest

from opendbc.car import structs
from opendbc.car.gwm.interface import CarInterface
from opendbc.car.gwm.values import CAR


class TestGwmRequestTracking(unittest.TestCase):
  def setUp(self):
    self.ci = CarInterface(CarInterface.get_non_essential_params(CAR.GWM_HAVAL_H6))
    self.control = structs.CarControl()
    self.control.enabled = self.control.latActive = True
    self.control.actuators.torque = 0.5

  def cycle(self, original=None):
    packets = [] if original is None else [[1_000_000 * (self.ci.CC.frame + 1), [(0x12b, original, 2)]]]
    state = self.ci.update(packets)
    _, frames = self.ci.apply(self.control.as_reader(), 0)
    return state, [data for address, data, bus in frames if address == 0x12b and bus == 0]

  def test_no_original_never_creates_a_request_or_false_fault(self):
    for _ in range(150):
      state, sent = self.cycle()
      self.assertEqual(sent, [])
      self.assertFalse(self.ci.CC.lat_active)
      self.assertFalse(state.steerFaultTemporary)
    self.assertEqual(self.ci.steer_fault_temporary_counter, 0)

  def test_unconfirmed_emitted_request_still_raises_fault(self):
    for _ in range(150):
      state, _ = self.cycle(bytes(64))
    self.assertTrue(self.ci.CC.lat_active)
    self.assertTrue(state.steerFaultTemporary)

  def test_truncation_clears_request_and_recovery_restores_it(self):
    _, sent = self.cycle(bytes(64))
    self.assertTrue(sent[0][15] & 0x20)
    self.assertTrue(self.ci.CC.lat_active)
    # No transmit on the odd controller cycle: retain the previous transmitted request.
    _, sent = self.cycle(bytes(16))
    self.assertEqual(sent, [])
    self.assertTrue(self.ci.CC.lat_active)
    for _ in range(150):
      state, sent = self.cycle(bytes(16))
      self.assertEqual(sent, [])
      self.assertFalse(self.ci.CC.lat_active)
    self.assertFalse(state.steerFaultTemporary)
    self.assertEqual(self.ci.steer_fault_temporary_counter, 0)
    _, first = self.cycle(bytes(64))
    _, second = self.cycle(bytes(64))
    sent = first + second  # recovery spans one transmit and one non-transmit cycle
    self.assertEqual(len(sent), 1)
    self.assertTrue(sent[0][15] & 0x20)
    self.assertTrue(self.ci.CC.lat_active)

  def test_tracks_emitted_request_at_50hz_including_deactivation(self):
    self.ci.CS.camera_stock_data = bytes(64)
    for active in [True, False, False, True, True, False, False]:
      self.control.latActive = active
      previous = self.ci.CC.lat_active
      frame = self.ci.CC.frame
      _, sent = self.ci.CC.update(self.control.as_reader(), self.ci.CS, 0)
      steer = [data for address, data, bus in sent if address == 0x12b and bus == 0]
      if frame % 2 == 0:
        self.assertEqual(len(steer), 1)
        self.assertEqual(self.ci.CC.lat_active, bool(steer[0][15] & 0x20))
        self.assertEqual(self.ci.CC.lat_active, active)
      else:
        self.assertEqual(steer, [])
        self.assertEqual(self.ci.CC.lat_active, previous)

  def test_driver_opposition_is_not_an_eps_fault(self):
    self.ci.CS.camera_stock_data = bytes(64)
    self.ci.CS.out.steeringTorque = -150
    _, sent = self.ci.CC.update(self.control.as_reader(), self.ci.CS, 0)
    steer = next(data for address, data, bus in sent if address == 0x12b and bus == 0)
    self.assertFalse(steer[15] & 0x20)
    self.assertFalse(self.ci.CC.lat_active)
    for _ in range(150):
      state = self.ci.update([])
    self.assertFalse(state.steerFaultTemporary)
