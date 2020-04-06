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
import unittest

import qiskit.pulse as pulse
from qiskit.pulse.pulse_lib import gaussian
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q
from qiskit.pulse import pulse_lib, instructions


class TestBuilderContext(QiskitTestCase):
    """Test the pulse builder context."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()

    def test_context(self):
        """Test a general program build."""
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            pulse.delay(0, 1000)
            pulse.u2(0, 0, pi/2)
            pulse.delay(0, 1000)
            pulse.u2(0, 0, pi)
            with pulse.left_align():
                pulse.play(pulse.DriveChannel(0), gaussian(500, 0.1, 125))
                pulse.shift_phase(pulse.DriveChannel(0), pi/2)
                pulse.play(pulse.DriveChannel(0), gaussian(500, 0.1, 125))
                pulse.u2(1, 0, pi/2)
            with pulse.sequential():
                pulse.u2(0, 0, pi/2)
                pulse.u2(1, 0, pi/2)
                pulse.u2(0, 0, pi/2)
            pulse.measure(0)


class TestTransforms(TestBuilderContext):
    """Test builder transforms."""
    def test_parallel(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        test_pulse = pulse_lib.ConstantPulse(10, 1.0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.parallel():
                pulse.play(d0, test_pulse)
                pulse.play(d1, test_pulse)
                pulse.play(d0, test_pulse)

        reference = pulse.Schedule()
        # d0
        reference += instructions.Play(test_pulse, d0)
        reference += instructions.Play(test_pulse, d0)
        # d1
        reference += instructions.Play(test_pulse, d1)

        self.assertEqual(schedule, reference)

    def test_sequential(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        test_pulse = pulse_lib.ConstantPulse(10, 1.0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.sequential():
                pulse.play(d0, test_pulse)
                pulse.play(d1, test_pulse)
                pulse.play(d0, test_pulse)

        reference = pulse.Schedule()
        # d0
        reference.insert(0, instructions.Play(test_pulse, d0), mutate=True)
        reference.insert(20, instructions.Play(test_pulse, d0), mutate=True)
        # d1
        reference.insert(10, instructions.Play(test_pulse, d1), mutate=True)

        self.assertEqual(schedule, reference)

    def test_left_align(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        test_pulse = pulse_lib.ConstantPulse(10, 1.0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.left_align():
                pulse.play(d0, test_pulse)
                pulse.play(d1, test_pulse)
                pulse.play(d0, test_pulse)

        reference = pulse.Schedule()
        # d0
        reference += instructions.Play(test_pulse, d0)
        reference += instructions.Play(test_pulse, d0)
        # d1
        reference += instructions.Play(test_pulse, d1)

        self.assertEqual(schedule, reference)

    def test_right_align(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        test_pulse = pulse_lib.ConstantPulse(10, 1.0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.right_align():
                pulse.play(d0, test_pulse)
                pulse.play(d1, test_pulse)
                pulse.play(d0, test_pulse)

        reference = pulse.Schedule()
        # d0
        reference += instructions.Play(test_pulse, d0)
        reference += instructions.Play(test_pulse, d0)
        # d1
        reference.insert(10, instructions.Play(test_pulse, d1), mutate=True)

        self.assertEqual(schedule, reference)


class TestInstructions(TestBuilderContext):
    """Test builder instructions."""
    @unittest.expectedFailure
    def test_barrier(self):
        d0 = pulse.DriveChannel(0)
        d1 = pulse.DriveChannel(1)
        test_pulse = pulse_lib.ConstantPulse(10, 1.0)

        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.left_align():
                pulse.play(d0, test_pulse)
                pulse.barrier(d0, d1)
                pulse.play(d1, test_pulse)

        reference = pulse.Schedule()
        # d0
        reference += instructions.Play(test_pulse, d0)
        # d1
        reference.insert(10, instructions.Play(test_pulse, d1), mutate=True)

        self.assertEqual(schedule, reference)

        # test barrier on qubits
        schedule = pulse.Schedule()
        with pulse.build(self.backend, schedule):
            with pulse.left_align():
                pulse.play(d0, test_pulse)
                pulse.barrier(0, 1)
                pulse.play(d1, test_pulse)

        self.assertEqual(schedule, reference)


class TestUtilities(TestBuilderContext):
    """Test builder utilities."""


class TestMacros(TestBuilderContext):
    """Test builder macros."""


class TestGates(TestBuilderContext):
    """Test builder gates."""
