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

# pylint: disable=missing-docstring

"""Test the PulseDefaults part of the backend."""
import warnings

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.qobj import PulseQobjInstruction
from qiskit.pulse import SamplePulse, Schedule, PulseError
from qiskit.pulse.channels import DriveChannel
from qiskit.pulse.schedule import ParameterizedSchedule


class TestPulseDefaults(QiskitTestCase):
    """Test the PulseDefaults creation and method usage."""

    def setUp(self):
        self.defs = FakeOpenPulse2Q().defaults()
        self.inst_map = self.defs.instruction_schedules

    def test_buffer(self):
        """Test getting the buffer value."""
        self.assertEqual(self.defs.buffer, 10)

    def test_freq_est(self):
        """Test extracting qubit frequencies."""
        warnings.simplefilter("ignore")
        self.assertEqual(self.defs.qubit_freq_est[1],
                         5.0 * 1e9)
        self.assertEqual(self.defs.meas_freq_est[0],
                         6.5 * 1e9)
        warnings.simplefilter("default")

    def test_instructions(self):
        """Test `instructions`."""
        instructions = self.inst_map.instructions
        for inst in ['u1', 'u3', 'cx', 'measure']:
            self.assertTrue(inst in instructions)

    def test_has(self):
        """Test `has` and `assert_has`."""
        self.assertTrue(self.inst_map.has('u1', [0]))
        self.assertTrue(self.inst_map.has('cx', (0, 1)))
        self.assertTrue(self.inst_map.has('u3', 0))
        self.assertTrue(self.inst_map.has('measure', [0, 1]))
        self.assertFalse(self.inst_map.has('u1', [0, 1]))
        with self.assertRaises(PulseError):
            self.inst_map.assert_has('dne', [0])
        with self.assertRaises(PulseError):
            self.inst_map.assert_has('cx', 100)

    def test_default_building(self):
        """Test building of ops definition is properly built from backend."""
        self.assertTrue(self.inst_map.has('u1', (0,)))
        self.assertTrue(self.inst_map.has('u3', (0,)))
        self.assertTrue(self.inst_map.has('u3', 1))
        self.assertTrue(self.inst_map.has('cx', (0, 1)))
        self.assertEqual(self.inst_map.get_parameters('u1', 0), ('P1',))
        u1_minus_pi = self.inst_map.get('u1', 0, P1=1)
        fc_cmd = u1_minus_pi.instructions[0][-1].command
        self.assertEqual(fc_cmd.phase, -np.pi)

    def test_str(self):
        """Test that __str__ method works."""
        self.assertEqual("<PulseDefaults(<InstructionScheduleMap(1Q instructions:\n  q0:",
                         str(self.defs)[:61])
        self.assertTrue("Multi qubit instructions:\n  (0, 1): " in str(self.defs)[70:])
        self.assertTrue("Qubit Frequencies [GHz]\n[4.9, 5.0]\nMeasurement Frequencies [GHz]\n[6.5, "
                        "6.6] )>" in str(self.defs)[100:])
