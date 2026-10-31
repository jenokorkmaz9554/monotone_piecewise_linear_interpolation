"""Tests for monotone piecewise linear interpolation."""

import math
import unittest

from monotone_piecewise_linear_interpolation import MonotoneInterpolant, build_interpolant


class TestMonotoneInterpolant(unittest.TestCase):
    def test_linear_data_is_reproduced(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [0.0, 2.0, 4.0, 6.0]
        interp = MonotoneInterpolant(xs, ys)
        for x, expected in [(0.5, 1.0), (1.5, 3.0), (2.5, 5.0)]:
            self.assertAlmostEqual(interp(x), expected)

    def test_monotone_data_stays_monotone(self):
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        ys = [0.0, 1.0, 1.5, 1.6, 1.7]
        interp = MonotoneInterpolant(xs, ys)
        prev = interp(0.0)
        for i in range(1, 41):
            x = i * 0.1
            val = interp(x)
            self.assertLessEqual(val, interp(xs[-1]))
            self.assertGreaterEqual(val, prev - 1e-12)
            prev = val

    def test_flat_data_gives_flat_interpolant(self):
        xs = [0.0, 1.0, 2.0]
        ys = [3.0, 3.0, 3.0]
        interp = MonotoneInterpolant(xs, ys)
        self.assertAlmostEqual(interp(0.5), 3.0)
        self.assertAlmostEqual(interp(1.5), 3.0)

    def test_out_of_range_raises(self):
        interp = MonotoneInterpolant([0.0, 1.0], [0.0, 1.0])
        with self.assertRaises(ValueError):
            interp(-0.1)
        with self.assertRaises(ValueError):
            interp(1.1)

    def test_unequal_lengths_raise(self):
        with self.assertRaises(ValueError):
            MonotoneInterpolant([0.0, 1.0], [0.0])

    def test_too_few_points_raise(self):
        with self.assertRaises(ValueError):
            MonotoneInterpolant([0.0], [0.0])

    def test_non_increasing_xs_raise(self):
        with self.assertRaises(ValueError):
            MonotoneInterpolant([0.0, 0.0], [0.0, 1.0])
        with self.assertRaises(ValueError):
            MonotoneInterpolant([0.0, -1.0], [0.0, 1.0])

    def test_build_interpolant_returns_instance(self):
        interp = build_interpolant([0.0, 1.0], [0.0, 2.0])
        self.assertIsInstance(interp, MonotoneInterpolant)
        self.assertAlmostEqual(interp(0.5), 1.0)

    def test_slopes_property(self):
        interp = MonotoneInterpolant([0.0, 1.0, 2.0], [0.0, 1.0, 0.0])
        slopes = interp.slopes
        self.assertEqual(len(slopes), 2)
        # The limiter should cap the slopes at 3 times the smaller adjacent
        # secant, which for this data is 1.0.
        self.assertAlmostEqual(slopes[0], 1.0)
        self.assertAlmostEqual(slopes[1], -1.0)

    def test_exact_data_points(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [1.0, 3.0, 2.0, 5.0]
        interp = MonotoneInterpolant(xs, ys)
        for x, y in zip(xs, ys):
            self.assertAlmostEqual(interp(x), y)

    def test_no_overshoot_on_sharp_change(self):
        # Data with a sharp change in slope: the limiter should prevent
        # the interpolant from exceeding the data range on the steep side.
        xs = [0.0, 1.0, 2.0]
        ys = [0.0, 10.0, 11.0]
        interp = MonotoneInterpolant(xs, ys)
        for i in range(1, 20):
            x = i * 0.1
            val = interp(x)
            self.assertGreaterEqual(val, 0.0)
            self.assertLessEqual(val, 11.0)


if __name__ == "__main__":
    unittest.main()
