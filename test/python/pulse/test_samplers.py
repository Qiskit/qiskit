# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name, unexpected-keyword-arg, no-value-for-parameter

"""Tests pulse function samplers."""

import numpy as np

from qiskit.test import QiskitTestCase
import qiskit.pulse.commands as commands
import qiskit.pulse.samplers as samplers


def linear(times: np.ndarray, m: float, b: float = 0.1) -> np.ndarray:
    """Linear test function
    Args:
        times: Input times.
        m: Slope.
        b: Intercept
    Returns:
        np.ndarray
    """
    return m*times+b


class TestSampler(QiskitTestCase):
    """Test continuous pulse function samplers."""

    def test_left_sampler(self):
        """Test left sampler."""
        m = 0.1
        b = 0.1
        duration = 2
        left_linear_pulse_fun = samplers.left(linear)
        reference = np.array([0.1, 0.2], dtype=np.complex)

        pulse = left_linear_pulse_fun(duration, m=m, b=b)
        self.assertIsInstance(pulse, commands.SamplePulse)
        np.testing.assert_array_almost_equal(pulse.samples, reference)

    def test_right_sampler(self):
        """Test right sampler."""
        m = 0.1
        b = 0.1
        duration = 2
        right_linear_pulse_fun = samplers.right(linear)
        reference = np.array([0.2, 0.3], dtype=np.complex)

        pulse = right_linear_pulse_fun(duration, m=m, b=b)
        self.assertIsInstance(pulse, commands.SamplePulse)
        np.testing.assert_array_almost_equal(pulse.samples, reference)

    def test_midpoint_sampler(self):
        """Test midpoint sampler."""
        m = 0.1
        b = 0.1
        duration = 2
        midpoint_linear_pulse_fun = samplers.midpoint(linear)
        reference = np.array([0.15, 0.25], dtype=np.complex)

        pulse = midpoint_linear_pulse_fun(duration, m=m, b=b)
        self.assertIsInstance(pulse, commands.SamplePulse)
        np.testing.assert_array_almost_equal(pulse.samples, reference)

    def test_sampler_name(self):
        """Test that sampler setting of pulse name works."""
        m = 0.1
        b = 0.1
        duration = 2
        left_linear_pulse_fun = samplers.left(linear)

        pulse = left_linear_pulse_fun(duration, m=m, b=b, name='test')
        self.assertIsInstance(pulse, commands.SamplePulse)
        self.assertEqual(pulse.name, 'test')

    def test_default_arg_sampler(self):
        """Test that default arguments work with sampler."""
        m = 0.1
        duration = 2
        left_linear_pulse_fun = samplers.left(linear)
        reference = np.array([0.1, 0.2], dtype=np.complex)

        pulse = left_linear_pulse_fun(duration, m=m)
        self.assertIsInstance(pulse, commands.SamplePulse)
        np.testing.assert_array_almost_equal(pulse.samples, reference)
