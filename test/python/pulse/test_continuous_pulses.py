# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name, unexpected-keyword-arg, no-value-for-parameter

"""Tests continuous pulse functions."""

import numpy as np

from qiskit.test import QiskitTestCase
import qiskit.pulse.pulses.continuous as continuous


class TestContinuousPulses(QiskitTestCase):
    """Test continuous pulses."""

    def test_constant(self):
        """Test constant pulse."""
        constant_arr = continuous.constant(np.linspace(0, 10, 50), amp=0.5j)
        np.testing.assert_equal(constant_arr, 0.5j)
        self.assertEqual(len(constant_arr), 50)

    def test_zero(self):
        """Test constant pulse."""
        zero_arr = continuous.zero(np.linspace(0, 10, 50))
        np.testing.assert_equal(zero_arr, 0.0)
        self.assertEqual(len(zero_arr), 50)

    def test_square(self):
        """Test square wave."""
        square_arr = continuous.square(np.linspace(0, 10, 100), amp=0.5, period=5)
        # with new phase
        square_arr_phased = continuous.square(np.linspace(0, 10, 100), amp=0.5,
                                              period=5, phase=np.pi/2)
        self.assertAlmostEqual(square_arr[0], 0.5)
        self.assertAlmostEqual(square_arr[1]-square_arr[0], 0.0)
        self.assertAlmostEqual(square_arr[25], -0.5)
        self.assertAlmostEqual(square_arr_phased[0], -0.5)
        # Assert bounded between -amp and amp
        self.assertTrue(np.all((-0.5 <= square_arr) & (square_arr <= 0.5)))
        self.assertEqual(len(square_arr), 100)

    def test_sawtooth(self):
        """Test sawtooth wave."""
        amp = 0.5
        period = 5
        times, dt = np.linspace(0, 10, 101, retstep=True)
        sawtooth_arr = continuous.sawtooth(times, amp=amp, period=period)
        # with new phase
        sawtooth_arr_phased = continuous.sawtooth(times, amp=amp,
                                                  period=period, phase=np.pi/2)
        self.assertAlmostEqual(sawtooth_arr[0], 0.0)
        # test slope
        self.assertAlmostEqual((sawtooth_arr[1]-sawtooth_arr[0])/dt, 2*amp/period)
        self.assertAlmostEqual(sawtooth_arr[24], 0.48)
        self.assertAlmostEqual(sawtooth_arr[50], 0.)
        self.assertAlmostEqual(sawtooth_arr[75], -amp)
        self.assertAlmostEqual(sawtooth_arr_phased[0], -amp)
        # Assert bounded between -amp and amp
        self.assertTrue(np.all((-amp <= sawtooth_arr) & (sawtooth_arr <= amp)))
        self.assertEqual(len(sawtooth_arr), 101)

    def test_triangle(self):
        """Test triangle wave."""
        amp = 0.5
        period = 5
        times, dt = np.linspace(0, 10, 101, retstep=True)
        triangle_arr = continuous.triangle(times, amp=amp, period=period)
        # with new phase
        triangle_arr_phased = continuous.triangle(times, amp=amp,
                                                  period=period, phase=np.pi/2)
        self.assertAlmostEqual(triangle_arr[0], 0.0)
        # test slope
        self.assertAlmostEqual((triangle_arr[1]-triangle_arr[0])/dt, 4*amp/period)
        self.assertAlmostEqual(triangle_arr[12], 0.48)
        self.assertAlmostEqual(triangle_arr[13], 0.48)
        self.assertAlmostEqual(triangle_arr[50], 0.)
        self.assertAlmostEqual(triangle_arr_phased[0], amp)
        # Assert bounded between -amp and amp
        self.assertTrue(np.all((-amp <= triangle_arr) & (triangle_arr <= amp)))
        self.assertEqual(len(triangle_arr), 101)

    def test_cos(self):
        """Test cosine wave."""
        amp = 0.5
        period = 5
        freq = 1/period
        times, dt = np.linspace(0, 10, 101, retstep=True)
        cos_arr = continuous.cos(np.linspace(0, 10, 101), amp=amp, freq=freq)
        # with new phase
        cos_arr_phased = continuous.cos(times, amp=amp,
                                        freq=freq, phase=np.pi/2)
        # Assert starts at 1
        self.assertAlmostEqual(cos_arr[0], amp)
        self.assertAlmostEqual(cos_arr[6], 0.3644, places=2)
        self.assertAlmostEqual(cos_arr[25], -amp)
        self.assertAlmostEqual(cos_arr[50], amp)
        self.assertAlmostEqual(cos_arr_phased[0], 0.0)
        # Assert bounded between -amp and amp
        self.assertTrue(np.all((-amp <= cos_arr) & (cos_arr <= amp)))
        self.assertEqual(len(cos_arr), 101)

    def test_sin(self):
        """Test sine wave."""
        amp = 0.5
        period = 5
        freq = 1/period
        times, dt = np.linspace(0, 10, 101, retstep=True)
        sin_arr = continuous.sin(times, amp=amp, freq=freq)
        # with new phase
        sin_arr_phased = continuous.sin(np.linspace(0, 10, 101), amp=0.5,
                                        freq=1/5, phase=np.pi/2)
        # Assert starts at 1
        self.assertAlmostEqual(sin_arr[0], 0.0)
        self.assertAlmostEqual(sin_arr[6], 0.34227, places=2)
        self.assertAlmostEqual(sin_arr[25], 0.0)
        self.assertAlmostEqual(sin_arr[13], 0.5, places=2)
        self.assertAlmostEqual(sin_arr_phased[0], 0.5)
        # Assert bounded between -amp and amp
        self.assertTrue(np.all((-0.5 <= sin_arr) & (sin_arr <= 0.5)))
        self.assertEqual(len(sin_arr), 101)

    def test_gaussian(self):
        """Test gaussian pulse."""
        amp = 0.5
        center = 10
        sigma = 2
        times, dt = np.linspace(0, 20, 1001, retstep=True)
        gaussian_arr = continuous.gaussian(times, amp, center, sigma)
        center_time = np.argmax(gaussian_arr)
        self.assertTrue(times[center_time], center)
        self.assertTrue(gaussian_arr[center_time], amp)
        self.assertAlmostEqual(np.sum(gaussian_arr*dt), amp*np.sqrt(2*np.pi*sigma**2), places=3)

    def test_gaussian_deriv(self):
        """Test gaussian derivative pulse."""
        amp = 0.5
        center = 10
        sigma = 2
        times, dt = np.linspace(0, 20, 1000, retstep=True)
        deriv_prefactor = -sigma**2/(times-center)

        gaussian_deriv_arr = continuous.gaussian_deriv(times, amp, center, sigma)
        gaussian_arr = gaussian_deriv_arr*deriv_prefactor

        self.assertAlmostEqual(continuous.gaussian_deriv(np.array([0]), amp, center, sigma)[0],
                               0, places=5)
        self.assertAlmostEqual(np.sum(gaussian_arr*dt), amp*np.sqrt(2*np.pi*sigma**2), places=3)

    def test_gaussian_square(self):
        """Test gaussian square pulse."""
        amp = 0.5
        center = 10
        width = 2
        sigma = 0.1
        times, dt = np.linspace(0, 20, 2001, retstep=True)
        gaussian_square_arr = continuous.gaussian_square(times, amp, center, width, sigma)

        self.assertEqual(gaussian_square_arr[1000], amp)
        # test half gaussian rise/fall
        self.assertAlmostEqual(np.sum(gaussian_square_arr[:900]*dt)*2,
                               amp*np.sqrt(2*np.pi*sigma**2), places=2)
        self.assertAlmostEqual(np.sum(gaussian_square_arr[1100:]*dt)*2,
                               amp*np.sqrt(2*np.pi*sigma**2), places=2)

        # test for continuity at gaussian/square boundaries
        gauss_rise_end_time = center-width/2
        gauss_fall_start_time = center+width/2
        epsilon = 0.01
        rise_times, dt_rise = np.linspace(gauss_rise_end_time-epsilon,
                                          gauss_rise_end_time+epsilon, 1001, retstep=True)
        fall_times, dt_fall = np.linspace(gauss_fall_start_time-epsilon,
                                          gauss_fall_start_time+epsilon, 1001, retstep=True)
        gaussian_square_rise_arr = continuous.gaussian_square(rise_times, amp, center, width, sigma)
        gaussian_square_fall_arr = continuous.gaussian_square(fall_times, amp, center, width, sigma)

        # should be locally approximated by amp*dt^2/(2*sigma^2)
        self.assertAlmostEqual(amp*dt_rise**2/(2*sigma**2),
                               gaussian_square_rise_arr[500]-gaussian_square_rise_arr[499])
        self.assertAlmostEqual(amp*dt_fall**2/(2*sigma**2),
                               gaussian_square_fall_arr[501]-gaussian_square_fall_arr[500])
