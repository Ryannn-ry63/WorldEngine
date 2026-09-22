import unittest
from unittest.mock import patch
import numpy as np
from worldengine.online.numerics import check_numeric_runtime
from worldengine.components.agents.controller.tracker.tracker_utils import (
    _fit_initial_velocity_and_acceleration_profile)


class NumericRuntimeTest(unittest.TestCase):
    def test_health_check_preserves_global_rng(self):
        before = np.random.get_state()
        self.assertEqual(check_numeric_runtime()['status'], 'PASS')
        after = np.random.get_state()
        self.assertEqual(before[0], after[0])
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])

    def test_rejects_finite_but_wrong_pseudoinverse(self):
        with patch('numpy.linalg.pinv', side_effect=lambda a: np.zeros_like(a)):
            with self.assertRaisesRegex(RuntimeError, 'numeric runtime self-check failed'):
                check_numeric_runtime()

    def test_constant_speed_reference_has_physical_solution(self):
        # Mirrors the straight action that exposed the H100 failure. This has
        # a known physical answer, independently of snapshot hash equality.
        theta, speed, dt, count = -1.6, 5.0, .1, 40
        displacement = np.tile([np.cos(theta), np.sin(theta)], (count, 1)) * speed * dt
        velocity, acceleration = _fit_initial_velocity_and_acceleration_profile(
            displacement, np.full(count, theta), dt, 1e-4)
        self.assertAlmostEqual(velocity, speed, places=3)
        np.testing.assert_allclose(acceleration, 0., atol=1e-3)


if __name__ == '__main__':
    unittest.main()
