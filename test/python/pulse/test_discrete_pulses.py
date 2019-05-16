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

"""Tests discrete sampled pulse functions."""

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.pulse import SamplePulse
import qiskit.pulse.pulse_lib as pulse_lib
import qiskit.pulse.pulse_lib.continuous as continuous


class TestDiscretePulses(QiskitTestCase):
    """Test discreted sampled pulses."""

    def test_constant(self):
        """Test discrete sampled constant pulse."""
        amp = 0.5j
        duration = 10
        times = np.arange(0, duration)
        constant_ref = continuous.constant(times, amp=amp)
        constant_pulse = pulse_lib.constant(duration, amp=amp)
        self.assertIsInstance(constant_pulse, SamplePulse)
        np.testing.assert_array_almost_equal(constant_pulse.samples, constant_ref)

    def test_zero(self):
        """Test discrete sampled constant pulse."""
        duration = 10
        times = np.arange(0, duration)
        zero_ref = continuous.zero(times)
        zero_pulse = pulse_lib.zero(duration)
        self.assertIsInstance(zero_pulse, SamplePulse)
        np.testing.assert_array_almost_equal(zero_pulse.samples, zero_ref)

    def test_square(self):
        """Test discrete sampled square wave."""
        amp = 0.5
        period = 5
        duration = 10
        times = np.arange(0, duration)
        square_ref = continuous.square(times, amp=amp, period=period)
        square_pulse = pulse_lib.square(duration, amp=amp, period=period)
        self.assertIsInstance(square_pulse, SamplePulse)
        np.testing.assert_array_almost_equal(square_pulse.samples, square_ref)

        # test single cycle
        cycle_period = duration
        square_cycle_ref = continuous.square(times, amp=amp, period=cycle_period)
        square_cycle_pulse = pulse_lib.square(duration, amp=amp)
        np.testing.assert_array_almost_equal(square_cycle_pulse.samples, square_cycle_ref)

    def test_sawtooth(self):
        """Test discrete sampled sawtooth wave."""
        amp = 0.5
        period = 5
        duration = 10
        times = np.arange(0, duration)
        sawtooth_ref = continuous.sawtooth(times, amp=amp, period=period)
        sawtooth_pulse = pulse_lib.sawtooth(duration, amp=amp, period=period)
        self.assertIsInstance(sawtooth_pulse, SamplePulse)
        np.testing.assert_array_equal(sawtooth_pulse.samples, sawtooth_ref)

        # test single cycle
        cycle_period = duration
        sawtooth_cycle_ref = continuous.sawtooth(times, amp=amp, period=cycle_period)
        sawtooth_cycle_pulse = pulse_lib.sawtooth(duration, amp=amp)
        np.testing.assert_array_almost_equal(sawtooth_cycle_pulse.samples, sawtooth_cycle_ref)

    def test_triangle(self):
        """Test discrete sampled triangle wave."""
        amp = 0.5
        period = 5
        duration = 10
        times = np.arange(0, duration)
        triangle_ref = continuous.triangle(times, amp=amp, period=period)
        triangle_pulse = pulse_lib.triangle(duration, amp=amp, period=period)
        self.assertIsInstance(triangle_pulse, SamplePulse)
        np.testing.assert_array_almost_equal(triangle_pulse.samples, triangle_ref)

        # test single cycle
        cycle_period = duration
        triangle_cycle_ref = continuous.triangle(times, amp=amp, period=cycle_period)
        triangle_cycle_pulse = pulse_lib.triangle(duration, amp=amp)
        np.testing.assert_array_equal(triangle_cycle_pulse.samples, triangle_cycle_ref)

    def test_cos(self):
        """Test discrete sampled cosine wave."""
        amp = 0.5
        period = 5
        freq = 1/period
        duration = 10
        times = np.arange(0, duration)
        cos_ref = continuous.cos(times, amp=amp, freq=freq)
        cos_pulse = pulse_lib.cos(duration, amp=amp, freq=freq)
        self.assertIsInstance(cos_pulse, SamplePulse)
        np.testing.assert_array_almost_equal(cos_pulse.samples, cos_ref)

        # test single cycle
        cycle_freq = 1/duration
        cos_cycle_ref = continuous.cos(times, amp=amp, freq=cycle_freq)
        cos_cycle_pulse = pulse_lib.cos(duration, amp=amp)
        np.testing.assert_array_almost_equal(cos_cycle_pulse.samples, cos_cycle_ref)

    def test_sin(self):
        """Test discrete sampled sine wave."""
        amp = 0.5
        period = 5
        freq = 1/period
        duration = 10
        times = np.arange(0, duration)
        sin_ref = continuous.sin(times, amp=amp, freq=freq)
        sin_pulse = pulse_lib.sin(duration, amp=amp, freq=freq)
        self.assertIsInstance(sin_pulse, SamplePulse)
        np.testing.assert_array_equal(sin_pulse.samples, sin_ref)

        # test single cycle
        cycle_freq = 1/duration
        sin_cycle_ref = continuous.sin(times, amp=amp, freq=cycle_freq)
        sin_cycle_pulse = pulse_lib.sin(duration, amp=amp)
        np.testing.assert_array_almost_equal(sin_cycle_pulse.samples, sin_cycle_ref)

    def test_gaussian(self):
        """Test gaussian pulse."""
        amp = 0.5
        sigma = 2
        duration = 10
        center = duration/2
        times = np.arange(0, duration)
        gaussian_ref = continuous.gaussian(times, amp, center, sigma,
                                           zeroed_width=2*(center+1), rescale_amp=True)
        gaussian_pulse = pulse_lib.gaussian(duration, amp, sigma)
        self.assertIsInstance(gaussian_pulse, SamplePulse)
        np.testing.assert_array_almost_equal(gaussian_pulse.samples, gaussian_ref)

    def test_gaussian_deriv(self):
        """Test discrete sampled gaussian derivative pulse."""
        amp = 0.5
        sigma = 2
        duration = 10
        center = duration/2
        times = np.arange(0, duration)
        gaussian_deriv_ref = continuous.gaussian_deriv(times, amp, center, sigma)
        gaussian_deriv_pulse = pulse_lib.gaussian_deriv(duration, amp, sigma)
        self.assertIsInstance(gaussian_deriv_pulse, SamplePulse)
        np.testing.assert_array_almost_equal(gaussian_deriv_pulse.samples, gaussian_deriv_ref)

    def test_gaussian_square(self):
        """Test discrete sampled gaussian square pulse."""
        amp = 0.5
        sigma = 0.1
        risefall = 2
        duration = 10
        center = duration/2
        width = duration-2*risefall
        center = duration/2
        times = np.arange(0, duration)
        gaussian_square_ref = continuous.gaussian_square(times, amp, center, width, sigma)
        gaussian_square_pulse = pulse_lib.gaussian_square(duration, amp, sigma, risefall)
        self.assertIsInstance(gaussian_square_pulse, SamplePulse)
        np.testing.assert_array_almost_equal(gaussian_square_pulse.samples, gaussian_square_ref)

    def test_drag(self):
        """Test discrete sampled drag pulse."""
        amp = 0.5
        sigma = 0.1
        beta = 0
        duration = 10
        center = 10/2
        times = np.arange(0, duration)
        # reference drag pulse
        drag_ref = continuous.drag(times, amp, center, sigma, beta=beta,
                                   zeroed_width=2*(center+1), rescale_amp=True)
        drag_pulse = pulse_lib.drag(duration, amp, sigma, beta=beta)
        self.assertIsInstance(drag_pulse, SamplePulse)
        np.testing.assert_array_almost_equal(drag_pulse.samples, drag_ref)
