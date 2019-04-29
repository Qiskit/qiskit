# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test for the CmdDef object."""

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeProvider
from qiskit.pulse import (CmdDef, SamplePulse, Schedule, DeviceSpecification)


class TestCmdDef(QiskitTestCase):
    """Test CmdDef methods."""

    def setUp(self):
        self.provider = FakeProvider()
        self.backend = self.provider.get_backend('fake_openpulse_2q')
        self.device = DeviceSpecification.create_from(self.backend)

    def test_get_backend(self):
        """Test that backend is fetchable with cmd def present."""

    def test_init(self):
        """Test `init`, `has_cmd`."""
        sched = Schedule(self.channel_store)
        sched.drive[0].add(SamplePulse(np.ones(5.)))
        cmd_def = CmdDef({('tmp', 0): sched})
        self.assertTrue(cmd_def.has_cmd('tmp', 0))

    def test_add(self):
        """Test `add`, `has_cmd`, `get`, `cmd_types`."""
        schedule = Schedule(self.channel_store)
        schedule.drive[0].add(SamplePulse(np.ones(5.)))
        cmd_def = CmdDef()
        cmd_def.add('tmp', 0, schedule)
        self.assertEqual(schedule, cmd_def.get('tmp', (0,)))

        self.assertIn('tmp', cmd_def.cmd_types())

    def test_default_get(self):
        """Test default get method."""
        cmd_def = CmdDef()
        sched = Schedule(self.channel_store)
        sched.drive[0].add(SamplePulse(np.ones(5.)))
        get_sched = cmd_def.get('tmp', (0,), sched)
        self.assertEqual(sched, get_sched)

    def test_pop(self):
        """Test pop with default."""
        sched = Schedule(self.channel_store)
        sched.drive[0].add(SamplePulse(np.ones(5.)))
        cmd_def = CmdDef()
        cmd_def.add('tmp', 0, sched)
        popped_sched = cmd_def.pop('tmp', 0)
        self.assertFalse(cmd_def.has_cmd('tmp', 0))

        default_sched = cmd_def.pop('not_there', (0,), sched)
        self.assertEqual(popped_sched, default_sched)

        self.assertNotIn('tmp', cmd_def.cmd_types())

    def test_iter(self):
        """Test iterator."""
        sched = Schedule(self.channel_store)
        sched.drive[0].add(SamplePulse(np.ones(5.)))
        cmd_def = CmdDef()
        cmd_def.add('tmp', (0,), sched)
        cmd_def.add('tmp', 1, sched)
        cmd_def.add('tmp_2', (0, 1), sched)

        for ((cmd_name, qubits), sched) in cmd_def:
            self.assertTrue(cmd_def.has_cmd(cmd_name, qubits))

        self.assertEqual(list(cmd_def), 3)

    def test_dunder(self):
        """test __setitem__."""
        schedule = Schedule(self.channel_store)
        schedule.drive[0].add(SamplePulse(np.ones(5.)))
        cmd_def = CmdDef()
        cmd_def['tmp', 0, 1] = schedule
        self.assertEqual(schedule, cmd_def['tmp', 0, 1])

        self.assertIn(('tmp', 0, 1), cmd_def)

    def test_compare(self):
        """test compare"""
        schedule = Schedule(self.channel_store)
        schedule.drive[0].add(SamplePulse(np.ones(5.)))
        cmd_def_1 = CmdDef()
        cmd_def_2 = CmdDef()
        cmd_def_1.add('tmp', 0, schedule)
        cmd_def_2.add('tmp', 0, schedule)
        self.assertEqual(cmd_def_1, cmd_def_2)

    def test_repr(self):
        """Test repr"""
        sched = Schedule(self.channel_store)
        sched.drive[0].add(SamplePulse(np.ones(5.)))
        cmd_def = CmdDef({('tmp', 0): sched})
        repr(cmd_def)
