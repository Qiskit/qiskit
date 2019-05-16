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

"""Test for the CmdDef object."""

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeProvider
from qiskit.qobj.converters import QobjToInstructionConverter
from qiskit.qobj import PulseQobjInstruction
from qiskit.pulse import (CmdDef, SamplePulse, Schedule, DeviceSpecification,
                          PulseError, PersistentValue)


class TestCmdDef(QiskitTestCase):
    """Test CmdDef methods."""

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
        cmd_def = CmdDef({('tmp', 0): sched})
        self.assertTrue(cmd_def.has('tmp', 0))

    def test_add(self):
        """Test `add`, `has`, `get`, `cmdss`."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(self.device.q[0].drive))
        cmd_def = CmdDef()
        cmd_def.add('tmp', 1, sched)
        cmd_def.add('tmp', 0, sched)
        self.assertEqual(sched.instructions, cmd_def.get('tmp', (0,)).instructions)

        self.assertIn('tmp', cmd_def.cmds())
        self.assertEqual(cmd_def.cmd_qubits('tmp'), [(0,), (1,)])

    def test_pop(self):
        """Test pop with default."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(self.device.q[0].drive))
        cmd_def = CmdDef()
        cmd_def.add('tmp', 0, sched)
        cmd_def.pop('tmp', 0)
        self.assertFalse(cmd_def.has('tmp', 0))

        with self.assertRaises(PulseError):
            cmd_def.pop('not_there', (0,))

    def test_repr(self):
        """Test repr."""
        sched = Schedule()
        sched.append(SamplePulse(np.ones(5))(self.device.q[0].drive))
        cmd_def = CmdDef({('tmp', 0): sched})
        repr(cmd_def)

    def test_parameterized_schedule(self):
        """Test building parameterized schedule."""
        cmd_def = CmdDef()
        converter = QobjToInstructionConverter([], buffer=0)
        qobj = PulseQobjInstruction(name='pv', ch='u1', t0=10, val='P2*cos(np.pi*P1)')
        converted_instruction = converter(qobj)

        cmd_def.add('pv_test', 0, converted_instruction)
        self.assertEqual(cmd_def.get_parameters('pv_test', 0), ('P1', 'P2'))

        sched = cmd_def.get('pv_test', 0, P1='0', P2=-1)
        self.assertEqual(sched.instructions[0][-1].command.value, -1)

    def test_build_cmd_def(self):
        """Test building of parameterized cmd_def from defaults."""
        defaults = self.backend.defaults()
        cmd_def = defaults.build_cmd_def()

        cx_pv = cmd_def.get('cx', (0, 1), P2=0)
        pv_found = False
        for _, instr in cx_pv.instructions:
            cmd = instr.command
            if isinstance(cmd, PersistentValue):
                self.assertEqual(cmd.value, 1)
                pv_found = True
        self.assertTrue(pv_found)

        self.assertEqual(cmd_def.get_parameters('u1', 0), ('P1',))

        u1_minus_pi = cmd_def.get('u1', 0, P1=1)
        fc_cmd = u1_minus_pi.instructions[0][-1].command
        self.assertEqual(fc_cmd.phase, np.pi)
