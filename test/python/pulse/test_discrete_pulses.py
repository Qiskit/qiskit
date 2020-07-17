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
from qiskit.pulse import Waveform, PulseError
import qiskit.pulse.library as library
import qiskit.pulse.library.continuous as continuous


class TestDiscretePulses(QiskitTestCase):
    """Test discreted sampled pulses."""

    def test_constant(self):
        """Test discrete sampled constant pulse."""
        amp = 0.5j
        duration = 10
        times = np.arange(0, duration) + 0.5  # to match default midpoint sampling strategy
        constant_ref = continuous.constant(times, amp=amp)
        constant_pulse = library.constant(duration, amp=amp)
        self.assertIsInstance(constant_pulse, Waveform)
        np.testing.assert_array_almost_equal(constant_pulse.samples, constant_ref)

    def test_zero(self):
        """Test discrete sampled constant pulse."""
        duration = 10
        times = np.arange(0, duration) + 0.5
        zero_ref = continuous.zero(times)
        zero_pulse = library.zero(duration)
        self.assertIsInstance(zero_pulse, Waveform)
        np.testing.assert_array_almost_equal(zero_pulse.samples, zero_ref)

    def test_square(self):
        """Test discrete sampled square wave."""
        amp = 0.5
        freq = 0.2
        duration = 10
        times = np.arange(0, duration) + 0.5
        square_ref = continuous.square(times, amp=amp, freq=freq)
        square_pulse = library.square(duration, amp=amp, freq=freq)
        self.assertIsInstance(square_pulse, Waveform)
        np.testing.assert_array_almost_equal(square_pulse.samples, square_ref)

        # test single cycle
        cycle_freq = 1./duration
        square_cycle_ref = continuous.square(times, amp=amp, freq=cycle_freq)
        square_cycle_pulse = library.square(duration, amp=amp)
        np.testing.assert_array_almost_equal(square_cycle_pulse.samples, square_cycle_ref)

    def test_sawtooth(self):
        """Test discrete sampled sawtooth wave."""
        amp = 0.5
        freq = 0.2
        duration = 10
        times = np.arange(0, duration) + 0.5
        sawtooth_ref = continuous.sawtooth(times, amp=amp, freq=freq)
        sawtooth_pulse = library.sawtooth(duration, amp=amp, freq=freq)
        self.assertIsInstance(sawtooth_pulse, Waveform)
        np.testing.assert_array_equal(sawtooth_pulse.samples, sawtooth_ref)

        # test single cycle
        cycle_freq = 1./duration
        sawtooth_cycle_ref = continuous.sawtooth(times, amp=amp, freq=cycle_freq)
        sawtooth_cycle_pulse = library.sawtooth(duration, amp=amp)
        np.testing.assert_array_almost_equal(sawtooth_cycle_pulse.samples, sawtooth_cycle_ref)

    def test_triangle(self):
        """Test discrete sampled triangle wave."""
        amp = 0.5
        freq = 0.2
        duration = 10
        times = np.arange(0, duration) + 0.5
        triangle_ref = continuous.triangle(times, amp=amp, freq=freq)
        triangle_pulse = library.triangle(duration, amp=amp, freq=freq)
        self.assertIsInstance(triangle_pulse, Waveform)
        np.testing.assert_array_almost_equal(triangle_pulse.samples, triangle_ref)

        # test single cycle
        cycle_freq = 1./duration
        triangle_cycle_ref = continuous.triangle(times, amp=amp, freq=cycle_freq)
        triangle_cycle_pulse = library.triangle(duration, amp=amp)
        np.testing.assert_array_equal(triangle_cycle_pulse.samples, triangle_cycle_ref)

    def test_cos(self):
        """Test discrete sampled cosine wave."""
        amp = 0.5
        period = 5
        freq = 1/period
        duration = 10
        times = np.arange(0, duration) + 0.5
        cos_ref = continuous.cos(times, amp=amp, freq=freq)
        cos_pulse = library.cos(duration, amp=amp, freq=freq)
        self.assertIsInstance(cos_pulse, Waveform)
        np.testing.assert_array_almost_equal(cos_pulse.samples, cos_ref)

        # test single cycle
        cycle_freq = 1/duration
        cos_cycle_ref = continuous.cos(times, amp=amp, freq=cycle_freq)
        cos_cycle_pulse = library.cos(duration, amp=amp)
        np.testing.assert_array_almost_equal(cos_cycle_pulse.samples, cos_cycle_ref)

    def test_sin(self):
        """Test discrete sampled sine wave."""
        amp = 0.5
        period = 5
        freq = 1/period
        duration = 10
        times = np.arange(0, duration) + 0.5
        sin_ref = continuous.sin(times, amp=amp, freq=freq)
        sin_pulse = library.sin(duration, amp=amp, freq=freq)
        self.assertIsInstance(sin_pulse, Waveform)
        np.testing.assert_array_equal(sin_pulse.samples, sin_ref)

        # test single cycle
        cycle_freq = 1/duration
        sin_cycle_ref = continuous.sin(times, amp=amp, freq=cycle_freq)
        sin_cycle_pulse = library.sin(duration, amp=amp)
        np.testing.assert_array_almost_equal(sin_cycle_pulse.samples, sin_cycle_ref)

    def test_gaussian(self):
        """Test gaussian pulse."""
        amp = 0.5
        sigma = 2
        duration = 10
        center = duration/2
        times = np.arange(0, duration) + 0.5
        gaussian_ref = continuous.gaussian(times, amp, center, sigma,
                                           zeroed_width=2*center, rescale_amp=True)
        gaussian_pulse = library.gaussian(duration, amp, sigma)
        self.assertIsInstance(gaussian_pulse, Waveform)
        np.testing.assert_array_almost_equal(gaussian_pulse.samples, gaussian_ref)

    def test_gaussian_deriv(self):
        """Test discrete sampled gaussian derivative pulse."""
        amp = 0.5
        sigma = 2
        duration = 10
        center = duration/2
        times = np.arange(0, duration) + 0.5
        gaussian_deriv_ref = continuous.gaussian_deriv(times, amp, center, sigma)
        gaussian_deriv_pulse = library.gaussian_deriv(duration, amp, sigma)
        self.assertIsInstance(gaussian_deriv_pulse, Waveform)
        np.testing.assert_array_almost_equal(gaussian_deriv_pulse.samples, gaussian_deriv_ref)

    def test_sech(self):
        """Test sech pulse."""
        amp = 0.5
        sigma = 2
        duration = 10
        center = duration/2
        times = np.arange(0, duration) + 0.5
        sech_ref = continuous.sech(times, amp, center, sigma,
                                   zeroed_width=2*center, rescale_amp=True)
        sech_pulse = library.sech(duration, amp, sigma)
        self.assertIsInstance(sech_pulse, Waveform)
        np.testing.assert_array_almost_equal(sech_pulse.samples, sech_ref)

    def test_sech_deriv(self):
        """Test discrete sampled sech derivative pulse."""
        amp = 0.5
        sigma = 2
        duration = 10
        center = duration/2
        times = np.arange(0, duration) + 0.5
        sech_deriv_ref = continuous.sech_deriv(times, amp, center, sigma)
        sech_deriv_pulse = library.sech_deriv(duration, amp, sigma)
        self.assertIsInstance(sech_deriv_pulse, Waveform)
        np.testing.assert_array_almost_equal(sech_deriv_pulse.samples, sech_deriv_ref)

    def test_gaussian_square(self):
        """Test discrete sampled gaussian square pulse."""
        amp = 0.5
        sigma = 0.1
        risefall = 2
        duration = 10
        center = duration/2
        width = duration-2*risefall
        center = duration/2
        times = np.arange(0, duration) + 0.5
        gaussian_square_ref = continuous.gaussian_square(times, amp, center, width, sigma)
        gaussian_square_pulse = library.gaussian_square(duration, amp, sigma, risefall)
        self.assertIsInstance(gaussian_square_pulse, Waveform)
        np.testing.assert_array_almost_equal(gaussian_square_pulse.samples, gaussian_square_ref)

    def test_gaussian_square_args(self):
        """Gaussian square allows the user to specify risefall or width. Test this."""
        amp = 0.5
        sigma = 0.1
        duration = 10
        # risefall and width consistent: no error
        library.gaussian_square(duration, amp, sigma, 2, width=6)
        # supply width instead: no error
        library.gaussian_square(duration, amp, sigma, width=6)
        with self.assertRaises(PulseError):
            library.gaussian_square(duration, amp, sigma, width=2, risefall=2)
        with self.assertRaises(PulseError):
            library.gaussian_square(duration, amp, sigma)

    def test_drag(self):
        """Test discrete sampled drag pulse."""
        amp = 0.5
        sigma = 0.1
        beta = 0
        duration = 10
        center = 10/2
        times = np.arange(0, duration) + 0.5
        # reference drag pulse
        drag_ref = continuous.drag(times, amp, center, sigma, beta=beta,
                                   zeroed_width=2*(center+1), rescale_amp=True)
        drag_pulse = library.drag(duration, amp, sigma, beta=beta)
        self.assertIsInstance(drag_pulse, Waveform)
        np.testing.assert_array_almost_equal(drag_pulse.samples, drag_ref)

    def test_period_deprecation_warning(self):
        """Tests for DeprecationWarning"""
        amp = 0.5
        period = 5.
        duration = 10
        self.assertWarns(DeprecationWarning,
                         lambda: library.triangle(duration, amp=amp, period=period))
        self.assertWarns(DeprecationWarning,
                         lambda: library.sawtooth(duration, amp=amp, period=period))
        self.assertWarns(DeprecationWarning,
                         lambda: library.square(duration, amp=amp, period=period))
