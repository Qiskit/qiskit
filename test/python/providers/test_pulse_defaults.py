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
        self.ops_def = self.defs.ops_def

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

    def test_ops(self):
        """Test `ops`."""
        ops = self.ops_def.ops()
        for op in ['u1', 'u3', 'cx', 'measure']:
            self.assertTrue(op in ops)

    def test_has(self):
        """Test `has` and `assert_has`."""
        self.assertTrue(self.ops_def.has('u1', [0]))
        self.assertTrue(self.ops_def.has('cx', (0, 1)))
        self.assertTrue(self.ops_def.has('u3', 0))
        self.assertTrue(self.ops_def.has('measure', [0, 1]))
        self.assertFalse(self.ops_def.has('u1', [0, 1]))
        with self.assertRaises(PulseError):
            self.ops_def.assert_has('dne', [0])
        with self.assertRaises(PulseError):
            self.ops_def.assert_has('cx', 100)

    def test_qubits_with_op(self):
        """Test `qubits_with_op`."""
        self.assertEqual(self.ops_def.qubits_with_op('u1'), [0, 1])
        self.assertEqual(self.ops_def.qubits_with_op('u3'), [0, 1])
        self.assertEqual(self.ops_def.qubits_with_op('cx'), [(0, 1)])
        self.assertEqual(self.ops_def.qubits_with_op('measure'), [(0, 1)])

    def test_qubit_ops(self):
        """Test `qubit_ops`."""
        self.assertEqual(self.ops_def.qubit_ops(0), ['u1', 'u2', 'u3'])
        self.assertEqual(self.ops_def.qubit_ops(1), ['u1', 'u2', 'u3'])
        self.assertEqual(self.ops_def.qubit_ops((0, 1)), ['cx', 'ParametrizedGate', 'measure'])

    def test_add(self):
        """Test add, and that errors are raised when expected."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(DriveChannel(0)))
        ops_def = FakeOpenPulse2Q().defaults().ops_def
        ops_def.add('tmp', 1, sched)
        ops_def.add('tmp', 0, sched)
        self.assertIn('tmp', ops_def.ops())
        self.assertEqual(ops_def.qubits_with_op('tmp'), [0, 1])
        self.assertTrue('tmp' in ops_def.qubit_ops(0))
        with self.assertRaises(PulseError):
            ops_def.add('tmp', (), sched)
        with self.assertRaises(PulseError):
            ops_def.add('tmp', 1, "not a schedule")

    def test_get(self):
        """Test `get`."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(DriveChannel(0)))
        ops_def = FakeOpenPulse2Q().defaults().ops_def
        ops_def.add('tmp', 0, sched)
        self.assertEqual(sched.instructions, ops_def.get('tmp', (0,)).instructions)

    def test_remove(self):
        """Test removing a defined operation and removing an undefined operation."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(DriveChannel(0)))
        self.ops_def.add('tmp', 0, sched)
        self.ops_def.remove('tmp', 0)
        self.assertFalse(self.ops_def.has('tmp', 0))
        with self.assertRaises(PulseError):
            self.ops_def.remove('not_there', (0,))

    def test_pop(self):
        """Test pop with default."""
        sched = Schedule()
        sched = sched.append(SamplePulse(np.ones(5))(DriveChannel(0)))
        self.ops_def.add('tmp', 0, sched)
        self.assertEqual(self.ops_def.pop('tmp', 0), sched)
        self.assertFalse(self.ops_def.has('tmp', 0))
        with self.assertRaises(PulseError):
            self.ops_def.pop('not_there', (0,))

    def test_parameterized_schedule(self):
        """Test adding parameterized schedule."""
        converter = QobjToInstructionConverter([], buffer=0)
        qobj = PulseQobjInstruction(name='pv', ch='u1', t0=10, val='P2*cos(np.pi*P1)')
        converted_instruction = converter(qobj)

        self.ops_def.add('pv_test', 0, converted_instruction)
        self.assertEqual(self.ops_def.get_parameters('pv_test', 0), ('P1', 'P2'))

        sched = self.ops_def.get('pv_test', 0, P1=0, P2=-1)
        self.assertEqual(sched.instructions[0][-1].command.value, -1)
        with self.assertRaises(PulseError):
            self.ops_def.get('pv_test', 0, 0, P1=-1)
        with self.assertRaises(PulseError):
            self.ops_def.get('pv_test', 0, P1=1, P2=2, P3=3)

    def test_sequenced_parameterized_schedule(self):
        """Test parametrized schedule consists of multiple instruction. """
        converter = QobjToInstructionConverter([], buffer=0)
        qobjs = [PulseQobjInstruction(name='fc', ch='d0', t0=10, phase='P1'),
                 PulseQobjInstruction(name='fc', ch='d0', t0=20, phase='P2'),
                 PulseQobjInstruction(name='fc', ch='d0', t0=30, phase='P3')]
        converted_instruction = [converter(qobj) for qobj in qobjs]

        self.ops_def.add('inst_seq', 0, ParameterizedSchedule(*converted_instruction,
                                                              name='inst_seq'))

        with self.assertRaises(PulseError):
            self.ops_def.get('inst_seq', 0, P1=1, P2=2, P3=3, P4=4, P5=5)

        with self.assertRaises(PulseError):
            self.ops_def.get('inst_seq', 0, P1=1)

        with self.assertRaises(PulseError):
            self.ops_def.get('inst_seq', 0, 1, 2, 3, P1=1)

        sched = self.ops_def.get('inst_seq', 0, 1, 2, 3)
        self.assertEqual(sched.instructions[0][-1].command.phase, 1)
        self.assertEqual(sched.instructions[1][-1].command.phase, 2)
        self.assertEqual(sched.instructions[2][-1].command.phase, 3)

        sched = self.ops_def.get('inst_seq', 0, P1=1, P2=2, P3=3)
        self.assertEqual(sched.instructions[0][-1].command.phase, 1)
        self.assertEqual(sched.instructions[1][-1].command.phase, 2)
        self.assertEqual(sched.instructions[2][-1].command.phase, 3)

        sched = self.ops_def.get('inst_seq', 0, 1, 2, P3=3)
        self.assertEqual(sched.instructions[0][-1].command.phase, 1)
        self.assertEqual(sched.instructions[1][-1].command.phase, 2)
        self.assertEqual(sched.instructions[2][-1].command.phase, 3)

    def test_default_building(self):
        """Test building of ops definition is properly built from backend."""
        self.assertTrue(self.ops_def.has('u1', (0,)))
        self.assertTrue(self.ops_def.has('u3', (0,)))
        self.assertTrue(self.ops_def.has('u3', 1))
        self.assertTrue(self.ops_def.has('cx', (0, 1)))
        self.assertEqual(self.ops_def.get_parameters('u1', 0), ('P1',))
        u1_minus_pi = self.ops_def.get('u1', 0, P1=1)
        fc_cmd = u1_minus_pi.instructions[0][-1].command
        self.assertEqual(fc_cmd.phase, -np.pi)
        for chan in u1_minus_pi.channels:
            # buffer no longer supported
            self.assertEqual(chan.buffer, 0)

    # def test_str(self):
    #     """Test that __repr__ method works."""
    #     self.assertEqual(
    #         str(self.ops_def),
    #         "<PulseDefaults(1Q operations:\n  q0: ['u1', 'u3']\n  q1: ['u3']\nMulti qubit "
    #         "operations:\n  (0, 1): ['cx', 'measure']\nQubit Frequencies [GHz]\n[4.9, 5.0]"
    #         "\nMeasurement Frequencies [GHz]\n[6.5, 6.6] )>")
