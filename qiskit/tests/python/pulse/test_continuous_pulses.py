# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Tests continuous pulse functions."""

import numpy as np

from qiskit.test import QiskitTestCase
import qiskit.pulse.library.continuous as continuous


class TestContinuousPulses(QiskitTestCase):
    """Test continuous pulses."""

    def test_constant(self):
        """Test constant pulse."""
        amp = 0.5j
        samples = 50
        times = np.linspace(0, 10, samples)

        constant_arr = continuous.constant(times, amp=amp)

        self.assertEqual(constant_arr.dtype, np.complex_)
        np.testing.assert_equal(constant_arr, amp)
        self.assertEqual(len(constant_arr), samples)

    def test_zero(self):
        """Test constant pulse."""
        times = np.linspace(0, 10, 50)
        zero_arr = continuous.zero(times)

        self.assertEqual(zero_arr.dtype, np.complex_)
        np.testing.assert_equal(zero_arr, 0.0)
        self.assertEqual(len(zero_arr), 50)

    def test_square(self):
        """Test square wave."""
        amp = 0.5
        freq = 0.2
        samples = 100
        times = np.linspace(0, 10, samples)
        square_arr = continuous.square(times, amp=amp, freq=freq)
        # with new phase
        square_arr_phased = continuous.square(times, amp=amp, freq=freq, phase=np.pi/2)

        self.assertEqual(square_arr.dtype, np.complex_)

        self.assertAlmostEqual(square_arr[0], amp)
        # test constant
        self.assertAlmostEqual(square_arr[1]-square_arr[0], 0.0)
        self.assertAlmostEqual(square_arr[25], -amp)
        self.assertAlmostEqual(square_arr_phased[0], -amp)
        # Assert bounded between -amp and amp
        self.assertTrue(np.all((-amp <= square_arr) & (square_arr <= amp)))
        self.assertEqual(len(square_arr), samples)

    def test_sawtooth(self):
        """Test sawtooth wave."""
        amp = 0.5
        freq = 0.2
        samples = 101
        times, dt = np.linspace(0, 10, samples, retstep=True)
        sawtooth_arr = continuous.sawtooth(times, amp=amp, freq=freq)
        # with new phase
        sawtooth_arr_phased = continuous.sawtooth(times, amp=amp,
                                                  freq=freq, phase=np.pi/2)

        self.assertEqual(sawtooth_arr.dtype, np.complex_)

        self.assertAlmostEqual(sawtooth_arr[0], 0.0)
        # test slope
        self.assertAlmostEqual((sawtooth_arr[1]-sawtooth_arr[0])/dt, 2*amp*freq)
        self.assertAlmostEqual(sawtooth_arr[24], 0.48)
        self.assertAlmostEqual(sawtooth_arr[50], 0.)
        self.assertAlmostEqual(sawtooth_arr[75], -amp)
        self.assertAlmostEqual(sawtooth_arr_phased[0], -amp)
        # Assert bounded between -amp and amp
        self.assertTrue(np.all((-amp <= sawtooth_arr) & (sawtooth_arr <= amp)))
        self.assertEqual(len(sawtooth_arr), samples)

    def test_triangle(self):
        """Test triangle wave."""
        amp = 0.5
        freq = 0.2
        samples = 101
        times, dt = np.linspace(0, 10, samples, retstep=True)
        triangle_arr = continuous.triangle(times, amp=amp, freq=freq)
        # with new phase
        triangle_arr_phased = continuous.triangle(times, amp=amp,
                                                  freq=freq, phase=np.pi/2)

        self.assertEqual(triangle_arr.dtype, np.complex_)

        self.assertAlmostEqual(triangle_arr[0], 0.0)
        # test slope
        self.assertAlmostEqual((triangle_arr[1]-triangle_arr[0])/dt, 4*amp*freq)
        self.assertAlmostEqual(triangle_arr[12], 0.48)
        self.assertAlmostEqual(triangle_arr[13], 0.48)
        self.assertAlmostEqual(triangle_arr[50], 0.)
        self.assertAlmostEqual(triangle_arr_phased[0], amp)
        # Assert bounded between -amp and amp
        self.assertTrue(np.all((-amp <= triangle_arr) & (triangle_arr <= amp)))
        self.assertEqual(len(triangle_arr), samples)

    def test_cos(self):
        """Test cosine wave."""
        amp = 0.5
        period = 5
        freq = 1/period
        samples = 101
        times = np.linspace(0, 10, samples)
        cos_arr = continuous.cos(times, amp=amp, freq=freq)
        # with new phase
        cos_arr_phased = continuous.cos(times, amp=amp,
                                        freq=freq, phase=np.pi/2)

        self.assertEqual(cos_arr.dtype, np.complex_)

        # Assert starts at 1
        self.assertAlmostEqual(cos_arr[0], amp)
        self.assertAlmostEqual(cos_arr[6], 0.3644, places=2)
        self.assertAlmostEqual(cos_arr[25], -amp)
        self.assertAlmostEqual(cos_arr[50], amp)
        self.assertAlmostEqual(cos_arr_phased[0], 0.0)
        # Assert bounded between -amp and amp
        self.assertTrue(np.all((-amp <= cos_arr) & (cos_arr <= amp)))
        self.assertEqual(len(cos_arr), samples)

    def test_sin(self):
        """Test sine wave."""
        amp = 0.5
        period = 5
        freq = 1/period
        samples = 101
        times = np.linspace(0, 10, samples)
        sin_arr = continuous.sin(times, amp=amp, freq=freq)
        # with new phase
        sin_arr_phased = continuous.sin(times, amp=0.5,
                                        freq=1/5, phase=np.pi/2)

        self.assertEqual(sin_arr.dtype, np.complex_)

        # Assert starts at 1
        self.assertAlmostEqual(sin_arr[0], 0.0)
        self.assertAlmostEqual(sin_arr[6], 0.3427, places=2)
        self.assertAlmostEqual(sin_arr[25], 0.0)
        self.assertAlmostEqual(sin_arr[13], amp, places=2)
        self.assertAlmostEqual(sin_arr_phased[0], amp)
        # Assert bounded between -amp and amp
        self.assertTrue(np.all((-amp <= sin_arr) & (sin_arr <= amp)))
        self.assertEqual(len(sin_arr), samples)

    def test_gaussian(self):
        """Test gaussian pulse."""
        amp = 0.5
        center = 10
        sigma = 2
        times, dt = np.linspace(0, 20, 1001, retstep=True)
        gaussian_arr = continuous.gaussian(times, amp, center, sigma)
        gaussian_arr_zeroed = continuous.gaussian(np.array([-1, 10]), amp, center,
                                                  sigma, zeroed_width=2*(center+1),
                                                  rescale_amp=True)

        self.assertEqual(gaussian_arr.dtype, np.complex_)

        center_time = np.argmax(gaussian_arr)
        self.assertAlmostEqual(times[center_time], center)
        self.assertAlmostEqual(gaussian_arr[center_time], amp)
        self.assertAlmostEqual(gaussian_arr_zeroed[0], 0., places=6)
        self.assertAlmostEqual(gaussian_arr_zeroed[1], amp)
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

        self.assertEqual(gaussian_deriv_arr.dtype, np.complex_)

        self.assertAlmostEqual(continuous.gaussian_deriv(np.array([0]), amp, center, sigma)[0],
                               0, places=5)
        self.assertAlmostEqual(np.sum(gaussian_arr*dt), amp*np.sqrt(2*np.pi*sigma**2), places=3)

    def test_sech(self):
        """Test sech pulse."""
        amp = 0.5
        center = 20
        sigma = 2
        times, dt = np.linspace(0, 40, 1001, retstep=True)
        sech_arr = continuous.sech(times, amp, center, sigma)
        sech_arr_zeroed = continuous.sech(np.array([-1, 20]), amp, center,
                                          sigma)

        self.assertEqual(sech_arr.dtype, np.complex_)

        center_time = np.argmax(sech_arr)
        self.assertAlmostEqual(times[center_time], center)
        self.assertAlmostEqual(sech_arr[center_time], amp)
        self.assertAlmostEqual(sech_arr_zeroed[0], 0., places=2)
        self.assertAlmostEqual(sech_arr_zeroed[1], amp)
        self.assertAlmostEqual(np.sum(sech_arr*dt), amp*np.pi*sigma, places=3)

    def test_sech_deriv(self):
        """Test sech derivative pulse."""
        amp = 0.5
        center = 20
        sigma = 2
        times = np.linspace(0, 40, 1000)

        sech_deriv_arr = continuous.sech_deriv(times, amp, center, sigma)

        self.assertEqual(sech_deriv_arr.dtype, np.complex_)

        self.assertAlmostEqual(continuous.sech_deriv(np.array([0]), amp, center, sigma)[0],
                               0, places=3)

    def test_gaussian_square(self):
        """Test gaussian square pulse."""
        amp = 0.5
        center = 10
        width = 2
        sigma = 0.1
        times, dt = np.linspace(0, 20, 2001, retstep=True)
        gaussian_square_arr = continuous.gaussian_square(times, amp, center, width, sigma)

        self.assertEqual(gaussian_square_arr.dtype, np.complex_)

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

    def test_drag(self):
        """Test drag pulse."""
        amp = 0.5
        center = 10
        sigma = 0.1
        beta = 0
        times = np.linspace(0, 20, 2001)
        # test that we recover gaussian for beta=0
        gaussian_arr = continuous.gaussian(times, amp, center, sigma,
                                           zeroed_width=2*(center+1), rescale_amp=True)

        drag_arr = continuous.drag(times, amp, center, sigma, beta=beta,
                                   zeroed_width=2*(center+1), rescale_amp=True)

        self.assertEqual(drag_arr.dtype, np.complex_)

        np.testing.assert_equal(drag_arr, gaussian_arr)

    def test_period_deprecation_warning(self):
        """Tests for DeprecationWarning"""
        amp = 0.5
        period = 5.
        samples = 101
        times, _ = np.linspace(0, 10, samples, retstep=True)
        self.assertWarns(DeprecationWarning,
                         lambda: continuous.triangle(times, amp=amp, period=period))
        self.assertWarns(DeprecationWarning,
                         lambda: continuous.sawtooth(times, amp=amp, period=period))
        self.assertWarns(DeprecationWarning,
                         lambda: continuous.square(times, amp=amp, period=period))
