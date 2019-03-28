# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name, unexpected-keyword-arg, no-value-for-parameter

"""Tests pulse function samplers."""

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.pulse.commands.functional_pulse import FunctionalPulseCommand
import qiskit.pulse.samplers as samplers


def linear(x: np.ndarray, m: float, b: float) -> np.ndarray:
    """Linear test function
    Args:
        x: Input times.
        m: Slope.
        b: Intercept
    Returns:
        np.ndarray
    """
    return m*x+b


class TestSampler(QiskitTestCase):
    """Test analytic pulse function samplers."""

    def test_left_sampler(self):
        """Test left sampler."""
        m = 0.1
        b = 0.1
        duration = 2
        left_linear_pulse_fun = samplers.left(linear)
        reference = np.array([0.1, 0.2], dtype=np.complex)

        pulse = left_linear_pulse_fun(duration, m=m, b=b)
        self.assertIsInstance(pulse, FunctionalPulseCommand)
        np.testing.assert_array_almost_equal(pulse.samples, reference)

    def test_right_sampler(self):
        """Test right sampler."""
        m = 0.1
        b = 0.1
        duration = 2
        right_linear_pulse_fun = samplers.right(linear)
        reference = np.array([0.2, 0.3], dtype=np.complex)

        pulse = right_linear_pulse_fun(duration, m=m, b=b)
        self.assertIsInstance(pulse, FunctionalPulseCommand)
        np.testing.assert_array_almost_equal(pulse.samples, reference)

    def test_midpoint_sampler(self):
        """Test midpoint sampler."""
        m = 0.1
        b = 0.1
        duration = 2
        midpoint_linear_pulse_fun = samplers.midpoint(linear)
        reference = np.array([0.15, 0.25], dtype=np.complex)

        pulse = midpoint_linear_pulse_fun(duration, m=m, b=b)
        self.assertIsInstance(pulse, FunctionalPulseCommand)
        np.testing.assert_array_almost_equal(pulse.samples, reference)
