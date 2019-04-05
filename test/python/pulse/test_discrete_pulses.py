# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name, unexpected-keyword-arg, no-value-for-parameter

"""Tests discrete sampled pulse functions."""

import numpy as np

from qiskit.test import QiskitTestCase
import qiskit.pulse_lib.pulses as pulses
import pulse_lib.discrete as discrete
import pulse_lib.continuous as continuous


class TestDiscretePulses(QiskitTestCase):
    """Test discreted sampled pulses."""

    def test_constant(self):
        """Test discrete sampled constant pulse."""
        amp = 0.5j
        times = np.arange(0, 10)
        ref_constant = continuous.constant(times, amp=amp)

    def test_zero(self):
        """Test discrete sampled constant pulse."""
        times = np.arange(0, 10)
        ref_zero = continuous.zero(times)

    def test_square(self):
        """Test discrete sampled square wave."""
        amp = 0.5
        period = 5
        times = np.arange(0, 10)
        ref_square = continuous.square(times, amp=amp, period=period)

    def test_sawtooth(self):
        """Test discrete sampled sawtooth wave."""
        amp = 0.5
        period = 5
        times = np.arange(0, 10)
        ref_sawtooth = continuous.sawtooth(times, amp=amp, period=period)

    def test_triangle(self):
        """Test discrete sampled triangle wave."""
        amp = 0.5
        period = 5
        times, dt = np.linspace(0, 10, 101, retstep=True)
        triangle_arr = continuous.triangle(times, amp=amp, period=period)

    def test_cos(self):
        """Test discrete sampled cosine wave."""
        amp = 0.5
        period = 5
        freq = 1/period
        times = np.linspace(0, 10)
        ref_cos = continuous.cos(times, amp=amp, freq=freq)

    def test_sin(self):
        """Test discrete sampled sine wave."""
        amp = 0.5
        period = 5
        freq = 1/period
        times = np.linspace(0, 10)
        ref_sin = continuous.sin(times, amp=amp, freq=freq)

    def test_gaussian(self):
        """Test gaussian pulse."""
        amp = 0.5
        center = 10
        sigma = 2
        times, dt = np.arange(0, 10)
        ref_gaussian = continuous.gaussian(times, amp, center, sigma)

    def test_gaussian_deriv(self):
        """Test discrete sampled gaussian derivative pulse."""
        amp = 0.5
        center = 10
        sigma = 2
        times = np.arange(0, 10)
        ref_gaussian_deriv = continuous.gaussian_deriv(times, amp, center, sigma)

    def test_gaussian_square(self):
        """Test discrete sampled gaussian square pulse."""
        amp = 0.5
        center = 10
        width = 2
        sigma = 0.1
        times = np.arange(0, 10)
        ref_gaussian_square = continuous.gaussian_square(times, amp, center, width, sigma)

    def test_drag(self):
        """Test discrete sampled drag pulse."""
        amp = 0.5
        center = 10
        sigma = 0.1
        beta = 0
        times = np.arange(0, 10)
        # reference drag pulse
        ref_drag = continuous.drag(times, amp, center, sigma, beta=beta,
                                   zero_at=-1, rescale_amp=True)
