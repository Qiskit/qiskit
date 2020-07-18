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

"""Test the InstructionScheduleMap."""
import numpy as np

import qiskit.pulse.library as library
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.qobj import PulseQobjInstruction
from qiskit.pulse import InstructionScheduleMap, Play, Waveform, Schedule, PulseError
from qiskit.pulse.channels import DriveChannel
from qiskit.pulse.schedule import ParameterizedSchedule


class TestInstructionScheduleMap(QiskitTestCase):
    """Test the InstructionScheduleMap."""

    def test_add(self):
        """Test add, and that errors are raised when expected."""
        sched = Schedule()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)))
        inst_map = InstructionScheduleMap()

        inst_map.add('u1', 1, sched)
        inst_map.add('u1', 0, sched)

        self.assertIn('u1', inst_map.instructions)
        self.assertEqual(inst_map.qubits_with_instruction('u1'), [0, 1])
        self.assertTrue('u1' in inst_map.qubit_instructions(0))

        with self.assertRaises(PulseError):
            inst_map.add('u1', (), sched)
        with self.assertRaises(PulseError):
            inst_map.add('u1', 1, "not a schedule")

    def test_instructions(self):
        """Test `instructions`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add('u1', 1, sched)
        inst_map.add('u3', 0, sched)

        instructions = inst_map.instructions
        for inst in ['u1', 'u3']:
            self.assertTrue(inst in instructions)

    def test_has(self):
        """Test `has` and `assert_has`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add('u1', (0,), sched)
        inst_map.add('cx', [0, 1], sched)

        self.assertTrue(inst_map.has('u1', [0]))
        self.assertTrue(inst_map.has('cx', (0, 1)))
        with self.assertRaises(PulseError):
            inst_map.assert_has('dne', [0])
        with self.assertRaises(PulseError):
            inst_map.assert_has('cx', 100)

    def test_has_from_mock(self):
        """Test `has` and `assert_has` from mock data."""
        inst_map = FakeOpenPulse2Q().defaults().instruction_schedule_map
        self.assertTrue(inst_map.has('u1', [0]))
        self.assertTrue(inst_map.has('cx', (0, 1)))
        self.assertTrue(inst_map.has('u3', 0))
        self.assertTrue(inst_map.has('measure', [0, 1]))
        self.assertFalse(inst_map.has('u1', [0, 1]))
        with self.assertRaises(PulseError):
            inst_map.assert_has('dne', [0])
        with self.assertRaises(PulseError):
            inst_map.assert_has('cx', 100)

    def test_qubits_with_instruction(self):
        """Test `qubits_with_instruction`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add('u1', (0,), sched)
        inst_map.add('u1', (1,), sched)
        inst_map.add('cx', [0, 1], sched)

        self.assertEqual(inst_map.qubits_with_instruction('u1'), [0, 1])
        self.assertEqual(inst_map.qubits_with_instruction('cx'), [(0, 1)])
        self.assertEqual(inst_map.qubits_with_instruction('none'), [])

    def test_qubit_instructions(self):
        """Test `qubit_instructions`."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add('u1', (0,), sched)
        inst_map.add('u1', (1,), sched)
        inst_map.add('cx', [0, 1], sched)

        self.assertEqual(inst_map.qubit_instructions(0), ['u1'])
        self.assertEqual(inst_map.qubit_instructions(1), ['u1'])
        self.assertEqual(inst_map.qubit_instructions((0, 1)), ['cx'])
        self.assertEqual(inst_map.qubit_instructions(10), [])

    def test_get(self):
        """Test `get`."""
        sched = Schedule()
        sched.append(Play(Waveform(np.ones(5)), DriveChannel(0)))
        inst_map = InstructionScheduleMap()

        inst_map.add('u1', 0, sched)

        self.assertEqual(sched, inst_map.get('u1', (0,)))

    def test_remove(self):
        """Test removing a defined operation and removing an undefined operation."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add('tmp', 0, sched)
        inst_map.remove('tmp', 0)
        self.assertFalse(inst_map.has('tmp', 0))
        with self.assertRaises(PulseError):
            inst_map.remove('not_there', (0,))
        self.assertFalse('tmp' in inst_map.qubit_instructions(0))

    def test_pop(self):
        """Test pop with default."""
        sched = Schedule()
        inst_map = InstructionScheduleMap()

        inst_map.add('tmp', 100, sched)
        self.assertEqual(inst_map.pop('tmp', 100), sched)
        self.assertFalse(inst_map.has('tmp', 100))

        self.assertEqual(inst_map.qubit_instructions(100), [])
        self.assertEqual(inst_map.qubits_with_instruction('tmp'), [])
        with self.assertRaises(PulseError):
            inst_map.pop('not_there', (0,))

    def test_sequenced_parameterized_schedule(self):
        """Test parametrized schedule consists of multiple instruction. """
        converter = QobjToInstructionConverter([], buffer=0)
        qobjs = [PulseQobjInstruction(name='fc', ch='d0', t0=10, phase='P1'),
                 PulseQobjInstruction(name='fc', ch='d0', t0=20, phase='P2'),
                 PulseQobjInstruction(name='fc', ch='d0', t0=30, phase='P3')]
        converted_instruction = [converter(qobj) for qobj in qobjs]

        inst_map = InstructionScheduleMap()

        inst_map.add('inst_seq', 0, ParameterizedSchedule(*converted_instruction,
                                                          name='inst_seq'))

        with self.assertRaises(PulseError):
            inst_map.get('inst_seq', 0, P1=1, P2=2, P3=3, P4=4, P5=5)

        with self.assertRaises(PulseError):
            inst_map.get('inst_seq', 0, P1=1)

        with self.assertRaises(PulseError):
            inst_map.get('inst_seq', 0, 1, 2, 3, P1=1)

        sched = inst_map.get('inst_seq', 0, 1, 2, 3)
        self.assertEqual(sched.instructions[0][-1].phase, 1)
        self.assertEqual(sched.instructions[1][-1].phase, 2)
        self.assertEqual(sched.instructions[2][-1].phase, 3)

        sched = inst_map.get('inst_seq', 0, P1=1, P2=2, P3=3)
        self.assertEqual(sched.instructions[0][-1].phase, 1)
        self.assertEqual(sched.instructions[1][-1].phase, 2)
        self.assertEqual(sched.instructions[2][-1].phase, 3)

        sched = inst_map.get('inst_seq', 0, 1, 2, P3=3)
        self.assertEqual(sched.instructions[0][-1].phase, 1)
        self.assertEqual(sched.instructions[1][-1].phase, 2)
        self.assertEqual(sched.instructions[2][-1].phase, 3)

    def test_schedule_generator(self):
        """Test schedule generator functionalty."""

        x_test = 10
        amp_test = 1.0

        def test_func(x):
            sched = Schedule()
            sched += Play(library.constant(int(x), amp_test), DriveChannel(0))
            return sched

        ref_sched = Schedule()
        ref_sched += Play(library.constant(x_test, amp_test), DriveChannel(0))

        inst_map = InstructionScheduleMap()
        inst_map.add('f', (0,), test_func)
        self.assertEqual(inst_map.get('f', (0,), x_test), ref_sched)

        self.assertEqual(inst_map.get_parameters('f', (0,)), ('x',))
