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

    def test_qubits_with_inst(self):
        """Test `qubits_with_inst`."""
        self.assertEqual(self.inst_map.qubits_with_inst('u1'), [0, 1])
        self.assertEqual(self.inst_map.qubits_with_inst('u3'), [0, 1])
        self.assertEqual(self.inst_map.qubits_with_inst('cx'), [(0, 1)])
        self.assertEqual(self.inst_map.qubits_with_inst('measure'), [(0, 1)])
        with self.assertRaises(PulseError):
            self.inst_map.qubits_with_inst('none')

    def test_qubit_insts(self):
        """Test `qubit_insts`."""
        self.assertEqual(self.inst_map.qubit_insts(0), {'u1', 'u2', 'u3'})
        self.assertEqual(self.inst_map.qubit_insts(1), {'u1', 'u2', 'u3'})
        self.assertEqual(self.inst_map.qubit_insts((0, 1)), {'cx', 'ParametrizedGate', 'measure'})
        with self.assertRaises(PulseError):
            self.inst_map.qubit_insts(10)

    def test_add(self):
        """Test add, and that errors are raised when expected."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(DriveChannel(0)))
        inst_map = FakeOpenPulse2Q().defaults().instruction_schedules
        inst_map.add('tmp', 1, sched)
        inst_map.add('tmp', 0, sched)
        self.assertIn('tmp', inst_map.instructions)
        self.assertEqual(inst_map.qubits_with_inst('tmp'), [0, 1])
        self.assertTrue('tmp' in inst_map.qubit_insts(0))
        with self.assertRaises(PulseError):
            inst_map.add('tmp', (), sched)
        with self.assertRaises(PulseError):
            inst_map.add('tmp', 1, "not a schedule")

    def test_get(self):
        """Test `get`."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(DriveChannel(0)))
        inst_map = FakeOpenPulse2Q().defaults().instruction_schedules
        inst_map.add('tmp', 0, sched)
        self.assertEqual(sched.instructions, inst_map.get('tmp', (0,)).instructions)

    def test_remove(self):
        """Test removing a defined operation and removing an undefined operation."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(DriveChannel(0)))
        self.inst_map.add('tmp', 0, sched)
        self.inst_map.remove('tmp', 0)
        self.assertFalse(self.inst_map.has('tmp', 0))
        with self.assertRaises(PulseError):
            self.inst_map.remove('not_there', (0,))
        self.assertFalse('tmp' in self.inst_map.qubit_insts(0))

    def test_pop(self):
        """Test pop with default."""
        sched = Schedule()
        sched = sched.append(SamplePulse(np.ones(5))(DriveChannel(0)))
        self.inst_map.add('tmp', 100, sched)
        self.assertEqual(self.inst_map.pop('tmp', 100), sched)
        self.assertFalse(self.inst_map.has('tmp', 100))
        with self.assertRaises(PulseError):
            self.inst_map.qubit_insts(100)
        with self.assertRaises(PulseError):
            self.inst_map.qubits_with_inst('tmp')
        with self.assertRaises(PulseError):
            self.inst_map.pop('not_there', (0,))

    def test_parameterized_schedule(self):
        """Test adding parameterized schedule."""
        converter = QobjToInstructionConverter([], buffer=0)
        qobj = PulseQobjInstruction(name='pv', ch='u1', t0=10, val='P2*cos(np.pi*P1)')
        converted_instruction = converter(qobj)

        self.inst_map.add('pv_test', 0, converted_instruction)
        self.assertEqual(self.inst_map.get_parameters('pv_test', 0), ('P1', 'P2'))

        sched = self.inst_map.get('pv_test', 0, P1=0, P2=-1)
        self.assertEqual(sched.instructions[0][-1].command.value, -1)
        with self.assertRaises(PulseError):
            self.inst_map.get('pv_test', 0, 0, P1=-1)
        with self.assertRaises(PulseError):
            self.inst_map.get('pv_test', 0, P1=1, P2=2, P3=3)

    def test_sequenced_parameterized_schedule(self):
        """Test parametrized schedule consists of multiple instruction. """
        converter = QobjToInstructionConverter([], buffer=0)
        qobjs = [PulseQobjInstruction(name='fc', ch='d0', t0=10, phase='P1'),
                 PulseQobjInstruction(name='fc', ch='d0', t0=20, phase='P2'),
                 PulseQobjInstruction(name='fc', ch='d0', t0=30, phase='P3')]
        converted_instruction = [converter(qobj) for qobj in qobjs]

        self.inst_map.add('inst_seq', 0, ParameterizedSchedule(*converted_instruction,
                                                               name='inst_seq'))

        with self.assertRaises(PulseError):
            self.inst_map.get('inst_seq', 0, P1=1, P2=2, P3=3, P4=4, P5=5)

        with self.assertRaises(PulseError):
            self.inst_map.get('inst_seq', 0, P1=1)

        with self.assertRaises(PulseError):
            self.inst_map.get('inst_seq', 0, 1, 2, 3, P1=1)

        sched = self.inst_map.get('inst_seq', 0, 1, 2, 3)
        self.assertEqual(sched.instructions[0][-1].command.phase, 1)
        self.assertEqual(sched.instructions[1][-1].command.phase, 2)
        self.assertEqual(sched.instructions[2][-1].command.phase, 3)

        sched = self.inst_map.get('inst_seq', 0, P1=1, P2=2, P3=3)
        self.assertEqual(sched.instructions[0][-1].command.phase, 1)
        self.assertEqual(sched.instructions[1][-1].command.phase, 2)
        self.assertEqual(sched.instructions[2][-1].command.phase, 3)

        sched = self.inst_map.get('inst_seq', 0, 1, 2, P3=3)
        self.assertEqual(sched.instructions[0][-1].command.phase, 1)
        self.assertEqual(sched.instructions[1][-1].command.phase, 2)
        self.assertEqual(sched.instructions[2][-1].command.phase, 3)

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
        for chan in u1_minus_pi.channels:
            # buffer no longer supported
            self.assertEqual(chan.buffer, 0)

    def test_str(self):
        """Test that __str__ method works."""
        self.assertEqual("<PulseDefaults(<InstructionScheduleMap(1Q instructions:\n  q0:",
                         str(self.defs)[:61])
        self.assertTrue("Multi qubit instructions:\n  (0, 1): " in str(self.defs)[70:])
        self.assertTrue("Qubit Frequencies [GHz]\n[4.9, 5.0]\nMeasurement Frequencies [GHz]\n[6.5, "
                        "6.6] )>" in str(self.defs)[100:])
