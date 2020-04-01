# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test pulse builder context utilities."""

from math import pi
from qiskit.pulse.pulse_lib import gaussian
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q
from qiskit.pulse import (build, delay, left_barrier, measure, play, sequential,
                          shift_phase, u2, DriveChannel, Schedule)


class TestBuilderContext(QiskitTestCase):
    """Test the pulse builder context."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()

    def test_context(self):
        backend = FakeOpenPulse2Q()

        schedule = Schedule()
        with build(backend, schedule):
            delay(0, 1000)
            u2(0, 0, pi/2)
            delay(0, 1000)
            u2(0, 0, pi)
            with left_barrier():
                play(DriveChannel(0), gaussian(500, 0.1, 125))
                shift_phase(DriveChannel(0), pi/2)
                play(DriveChannel(0), gaussian(500, 0.1, 125))
                u2(1, 0, pi/2)
            with sequential():
                u2(0, 0, pi/2)
                u2(1, 0, pi/2)
                u2(0, 0, pi/2)
            measure(0)


class TestTransforms(TestBuilderContext):
    """Test builder transforms."""


class TestInstructions(TestBuilderContext):
    """Test builder instructions."""


class TestUtilities(TestBuilderContext):
    """Test builder utilities."""


class TestMacros(TestBuilderContext):
    """Test builder macros."""


class TestGates(TestBuilderContext):
    """Test builder gates."""
