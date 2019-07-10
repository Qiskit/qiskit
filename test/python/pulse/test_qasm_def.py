# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the QasmToSchedDef object."""

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeProvider
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.qobj import PulseQobjInstruction
from qiskit.pulse import (QasmToSchedDef, SamplePulse, Schedule, PulseChannelSpec,
                          DeviceSpecification, PulseError, PersistentValue)
from qiskit.pulse.schedule import ParameterizedSchedule


class TestQasmToSchedDef(QiskitTestCase):
    """Test QasmToSchedDef methods."""

    def setUp(self):
        self.provider = FakeProvider()
        self.backend = self.provider.get_backend('fake_openpulse_2q')
        self.device = PulseChannelSpec.from_backend(self.backend)

    def test_get_backend(self):
        """Test that backend is fetchable with cmd def present."""

    def test_init(self):
        """Test `init`, `has`."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(self.device.drives[0]))
        qasm_def = QasmToSchedDef({('tmp', 0): sched})
        self.assertTrue(qasm_def.has('tmp', 0))

    def test_add(self):
        """Test `add`, `has`, `get`, `cmdss`."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(self.device.drives[0]))
        qasm_def = QasmToSchedDef()
        qasm_def.add('tmp', 1, sched)
        qasm_def.add('tmp', 0, sched)
        self.assertEqual(sched.instructions, qasm_def.get('tmp', (0,)).instructions)

        self.assertIn('tmp', qasm_def.gates())
        self.assertEqual(qasm_def.gate_qubits('tmp'), [(0,), (1,)])

    def test_pop(self):
        """Test pop with default."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(self.device.drives[0]))
        qasm_def = QasmToSchedDef()
        qasm_def.add('tmp', 0, sched)
        qasm_def.pop('tmp', 0)
        self.assertFalse(qasm_def.has('tmp', 0))

        with self.assertRaises(PulseError):
            qasm_def.pop('not_there', (0,))

    def test_repr(self):
        """Test repr."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(self.device.drives[0]))
        qasm_def = QasmToSchedDef({('tmp', 0): sched})
        repr(qasm_def)

    def test_parameterized_schedule(self):
        """Test building parameterized schedule."""
        qasm_def = QasmToSchedDef()
        converter = QobjToInstructionConverter([], buffer=0)
        qobj = PulseQobjInstruction(name='pv', ch='u1', t0=10, val='P2*cos(np.pi*P1)')
        converted_instruction = converter(qobj)

        qasm_def.add('pv_test', 0, converted_instruction)
        self.assertEqual(qasm_def.get_parameters('pv_test', 0), ('P1', 'P2'))

        sched = qasm_def.get('pv_test', 0, 0, P2=-1)
        self.assertEqual(sched.instructions[0][-1].command.value, -1)

        with self.assertRaises(PulseError):
            qasm_def.get('pv_test', 0, 0, P1=-1)

        with self.assertRaises(PulseError):
            qasm_def.get('pv_test', 0, P1=1, P2=2, P3=3)

        sched = qasm_def.pop('pv_test', 0, 0, P2=-1)
        self.assertEqual(sched.instructions[0][-1].command.value, -1)

        self.assertFalse(qasm_def.has('pv_test', 0))

    def test_sequenced_parameterized_schedule(self):
        """Test parametrized schedule consist of multiple instruction. """
        qasm_def = QasmToSchedDef()
        converter = QobjToInstructionConverter([], buffer=0)
        qobjs = [PulseQobjInstruction(name='fc', ch='d0', t0=10, phase='P1'),
                 PulseQobjInstruction(name='fc', ch='d0', t0=20, phase='P2'),
                 PulseQobjInstruction(name='fc', ch='d0', t0=30, phase='P3')]
        converted_instruction = [converter(qobj) for qobj in qobjs]

        qasm_def.add('inst_seq', 0, ParameterizedSchedule(*converted_instruction, name='inst_seq'))

        with self.assertRaises(PulseError):
            qasm_def.get('inst_seq', 0, P1=1, P2=2, P3=3, P4=4, P5=5)

        with self.assertRaises(PulseError):
            qasm_def.get('inst_seq', 0, P1=1)

        with self.assertRaises(PulseError):
            qasm_def.get('inst_seq', 0, 1, 2, 3, P1=1)

        sched = qasm_def.get('inst_seq', 0, 1, 2, 3)
        self.assertEqual(sched.instructions[0][-1].command.phase, 1)
        self.assertEqual(sched.instructions[1][-1].command.phase, 2)
        self.assertEqual(sched.instructions[2][-1].command.phase, 3)

        sched = qasm_def.get('inst_seq', 0, P1=1, P2=2, P3=3)
        self.assertEqual(sched.instructions[0][-1].command.phase, 1)
        self.assertEqual(sched.instructions[1][-1].command.phase, 2)
        self.assertEqual(sched.instructions[2][-1].command.phase, 3)

        sched = qasm_def.get('inst_seq', 0, 1, 2, P3=3)
        self.assertEqual(sched.instructions[0][-1].command.phase, 1)
        self.assertEqual(sched.instructions[1][-1].command.phase, 2)
        self.assertEqual(sched.instructions[2][-1].command.phase, 3)

    def test_build_cmd_def(self):
        """Test building of parameterized qasm_def from defaults."""
        defaults = self.backend.defaults()
        qasm_def = defaults.build_cmd_def()

        cx_pv = qasm_def.get('cx', (0, 1), P2=0)
        pv_found = False
        for _, instr in cx_pv.instructions:
            cmd = instr.command
            if isinstance(cmd, PersistentValue):
                self.assertEqual(cmd.value, 1)
                pv_found = True
        self.assertTrue(pv_found)

        self.assertEqual(qasm_def.get_parameters('u1', 0), ('P1',))

        u1_minus_pi = qasm_def.get('u1', 0, P1=1)
        fc_cmd = u1_minus_pi.instructions[0][-1].command
        self.assertEqual(fc_cmd.phase, np.pi)


class TestQasmToSchedDefWithDeviceSpecification(QiskitTestCase):
    """Test QasmToSchedDef methods."""
    # TODO: This test will be deprecated in future update.

    def setUp(self):
        self.provider = FakeProvider()
        self.backend = self.provider.get_backend('fake_openpulse_2q')
        self.device = DeviceSpecification.create_from(self.backend)

    def test_get_backend(self):
        """Test that backend is fetchable with cmd def present."""

    def test_init(self):
        """Test `init`, `has`."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(self.device.q[0].drive))
        qasm_def = QasmToSchedDef({('tmp', 0): sched})
        self.assertTrue(qasm_def.has('tmp', 0))

    def test_add(self):
        """Test `add`, `has`, `get`, `gates`."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(self.device.q[0].drive))
        qasm_def = QasmToSchedDef()
        qasm_def.add('tmp', 1, sched)
        qasm_def.add('tmp', 0, sched)
        self.assertEqual(sched.instructions, qasm_def.get('tmp', (0,)).instructions)

        self.assertIn('tmp', qasm_def.gates())
        self.assertEqual(qasm_def.gate_qubits('tmp'), [(0,), (1,)])

    def test_pop(self):
        """Test pop with default."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(self.device.q[0].drive))
        qasm_def = QasmToSchedDef()
        qasm_def.add('tmp', 0, sched)
        qasm_def.pop('tmp', 0)
        self.assertFalse(qasm_def.has('tmp', 0))

        with self.assertRaises(PulseError):
            qasm_def.pop('not_there', (0,))

    def test_repr(self):
        """Test repr."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(self.device.q[0].drive))
        qasm_def = QasmToSchedDef({('tmp', 0): sched})
        repr(qasm_def)

    def test_parameterized_schedule(self):
        """Test building parameterized schedule."""
        qasm_def = QasmToSchedDef()
        converter = QobjToInstructionConverter([], buffer=0)
        qobj = PulseQobjInstruction(name='pv', ch='u1', t0=10, val='P2*cos(np.pi*P1)')
        converted_instruction = converter(qobj)

        qasm_def.add('pv_test', 0, converted_instruction)
        self.assertEqual(qasm_def.get_parameters('pv_test', 0), ('P1', 'P2'))

        sched = qasm_def.get('pv_test', 0, 0, P2=-1)
        self.assertEqual(sched.instructions[0][-1].command.value, -1)

        with self.assertRaises(PulseError):
            qasm_def.get('pv_test', 0, 0, P1=-1)

        with self.assertRaises(PulseError):
            qasm_def.get('pv_test', 0, P1=1, P2=2, P3=3)

        sched = qasm_def.pop('pv_test', 0, 0, P2=-1)
        self.assertEqual(sched.instructions[0][-1].command.value, -1)

        self.assertFalse(qasm_def.has('pv_test', 0))

    def test_sequenced_parameterized_schedule(self):
        """Test parametrized schedule consist of multiple instruction. """
        qasm_def = QasmToSchedDef()
        converter = QobjToInstructionConverter([], buffer=0)
        qobjs = [PulseQobjInstruction(name='fc', ch='d0', t0=10, phase='P1'),
                 PulseQobjInstruction(name='fc', ch='d0', t0=20, phase='P2'),
                 PulseQobjInstruction(name='fc', ch='d0', t0=30, phase='P3')]
        converted_instruction = [converter(qobj) for qobj in qobjs]

        qasm_def.add('inst_seq', 0, ParameterizedSchedule(*converted_instruction, name='inst_seq'))

        with self.assertRaises(PulseError):
            qasm_def.get('inst_seq', 0, P1=1, P2=2, P3=3, P4=4, P5=5)

        with self.assertRaises(PulseError):
            qasm_def.get('inst_seq', 0, P1=1)

        with self.assertRaises(PulseError):
            qasm_def.get('inst_seq', 0, 1, 2, 3, P1=1)

        sched = qasm_def.get('inst_seq', 0, 1, 2, 3)
        self.assertEqual(sched.instructions[0][-1].command.phase, 1)
        self.assertEqual(sched.instructions[1][-1].command.phase, 2)
        self.assertEqual(sched.instructions[2][-1].command.phase, 3)

        sched = qasm_def.get('inst_seq', 0, P1=1, P2=2, P3=3)
        self.assertEqual(sched.instructions[0][-1].command.phase, 1)
        self.assertEqual(sched.instructions[1][-1].command.phase, 2)
        self.assertEqual(sched.instructions[2][-1].command.phase, 3)

        sched = qasm_def.get('inst_seq', 0, 1, 2, P3=3)
        self.assertEqual(sched.instructions[0][-1].command.phase, 1)
        self.assertEqual(sched.instructions[1][-1].command.phase, 2)
        self.assertEqual(sched.instructions[2][-1].command.phase, 3)

    def test_build_cmd_def(self):
        """Test building of parameterized qasm_def from defaults."""
        defaults = self.backend.defaults()
        qasm_def = defaults.build_cmd_def()

        cx_pv = qasm_def.get('cx', (0, 1), P2=0)
        pv_found = False
        for _, instr in cx_pv.instructions:
            cmd = instr.command
            if isinstance(cmd, PersistentValue):
                self.assertEqual(cmd.value, 1)
                pv_found = True
        self.assertTrue(pv_found)

        self.assertEqual(qasm_def.get_parameters('u1', 0), ('P1',))

        u1_minus_pi = qasm_def.get('u1', 0, P1=1)
        fc_cmd = u1_minus_pi.instructions[0][-1].command
        self.assertEqual(fc_cmd.phase, np.pi)

        for chan in u1_minus_pi.channels:
            self.assertEqual(chan.buffer, defaults.buffer)
