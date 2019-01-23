# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test cases for the functional pulse"""

import numpy as np


from qiskit.pulse.pulse import FunctionalPulse
from qiskit.test import QiskitTestCase


class TestFunctionalPulse(QiskitTestCase):
    """FunctionalPulse tests."""

    def test_gaussian(self):
        """Test gaussian pulse
        """

        @FunctionalPulse
        def gaussian(width, dt, amp, t0, sig):
            x = np.arange(0, width, dt)
            return amp * np.exp(-(x - t0) ** 2 / sig ** 2)

        pulse_instance = gaussian(9, dt=1, amp=1, t0=4, sig=1)
        expected_samp = [[1.1253517471925912e-07, 0],
                         [0.00012340980408667956, 0],
                         [0.01831563888873418, 0],
                         [0.36787944117144233, 0],
                         [1.0, 0],
                         [0.36787944117144233, 0],
                         [0.01831563888873418, 0],
                         [0.00012340980408667956, 0],
                         [1.1253517471925912e-07, 0]]
        self.assertEqual(pulse_instance.tolist(), expected_samp)

        # Parameter update (complex pulse)
        pulse_instance.params = {'amp': 0.5-0.5j}
        expected_samp = [[5.626758735962956e-08, -5.626758735962956e-08],
                         [6.170490204333978e-05, -6.170490204333978e-05],
                         [0.00915781944436709, -0.00915781944436709],
                         [0.18393972058572117, -0.18393972058572117],
                         [0.5, -0.5],
                         [0.18393972058572117, -0.18393972058572117],
                         [0.00915781944436709, -0.00915781944436709],
                         [6.170490204333978e-05, -6.170490204333978e-05],
                         [5.626758735962956e-08, -5.626758735962956e-08]]
        self.assertEqual(pulse_instance.tolist(), expected_samp)

    def test_square(self):
        """ Test square pulse
        """

        @FunctionalPulse
        def square(width, dt, amp, t0, t1):
            x = np.arange(0, width, dt)
            return np.where((x >= t0) & (x <= t1), amp, 0)

        pulse_instance = square(9, dt=1, amp=1, t0=3, t1=5)
        expected_samp = [[0.0, 0.0],
                         [0.0, 0.0],
                         [0.0, 0.0],
                         [1.0, 0.0],
                         [1.0, 0.0],
                         [1.0, 0.0],
                         [0.0, 0.0],
                         [0.0, 0.0],
                         [0.0, 0.0]]
        self.assertEqual(pulse_instance.tolist(), expected_samp)

        # Parameter update (complex pulse)
        pulse_instance.params = {'amp': 0.5-0.5j}
        expected_samp = [[0.0, 0.0],
                         [0.0, 0.0],
                         [0.0, 0.0],
                         [0.5, -0.5],
                         [0.5, -0.5],
                         [0.5, -0.5],
                         [0.0, 0.0],
                         [0.0, 0.0],
                         [0.0, 0.0]]
        self.assertEqual(pulse_instance.tolist(), expected_samp)
