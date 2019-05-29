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

"""Test cases for the pulse utilities."""
import unittest

import numpy as np

from qiskit import pulse
from qiskit.pulse.utils import add_implicit_acquires, align_measures
from qiskit.pulse.cmd_def import CmdDef
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q


class TestAutoMerge(QiskitTestCase):
    """Test the helper function which aligns acquires."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()
        self.device = pulse.DeviceSpecification.create_from(self.backend)
        self.config = self.backend.configuration()
        self.defaults = self.backend.defaults()
        self.cmd_def = CmdDef.from_defaults(self.defaults.cmd_def,
                                            self.defaults.pulse_library)
        self.short_pulse = pulse.SamplePulse(samples=np.array([0.02739068], dtype=np.complex128),
                                             name='p0')

    def test_align_measures(self):
        """Test that one acquire is delayed to match the time of the later acquire."""
        acquire = pulse.Acquire(5)
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, self.short_pulse(self.device.q[0].drive))
        sched = sched.insert(1, acquire(self.device.q[0], self.device.mem[0]))
        sched = sched.insert(10, acquire(self.device.q[1], self.device.mem[1]))
        sched = align_measures(sched, self.cmd_def)
        self.assertEqual(sched.instructions[1][0], 10)
        self.assertEqual(sched.instructions[2][0], 10)

    def test_align_post_u2(self):
        """Test that acquires are scheduled no sooner than the duration of the longest u2 gate.
        """
        acquire = pulse.Acquire(5)
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, self.short_pulse(self.device.q[0].drive))
        sched = sched.insert(1, acquire(self.device.q[0], self.device.mem[0]))
        sched = align_measures(sched, self.cmd_def)
        self.assertEqual(sched.instructions[1][0], 4)

    def test_error_multi_acquire(self):
        """Test that an error is raised if multiple acquires occur on the same channel."""
        acquire = pulse.Acquire(5)
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, self.short_pulse(self.device.q[0].drive))
        sched = sched.insert(4, acquire(self.device.q[0], self.device.mem[0]))
        sched = sched.insert(10, acquire(self.device.q[0], self.device.mem[0]))
        with self.assertRaises(ValueError):
            align_measures(sched, self.cmd_def)

    def test_error_post_acquire_pulse(self):
        """Test that an error is raised if a pulse occurs on a channel after an acquire."""
        acquire = pulse.Acquire(5)
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, self.short_pulse(self.device.q[0].drive))
        sched = sched.insert(4, acquire(self.device.q[0], self.device.mem[0]))
        # No error with separate channel
        sched = sched.insert(10, self.short_pulse(self.device.q[1].drive))
        align_measures(sched, self.cmd_def)
        sched = sched.insert(10, self.short_pulse(self.device.q[0].drive))
        with self.assertRaises(ValueError):
            align_measures(sched, self.cmd_def)


class TestAddImplicitAcquires(QiskitTestCase):
    """Test the helper function which makes implicit acquires explicit."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()
        self.device = pulse.DeviceSpecification.create_from(self.backend)
        self.config = self.backend.configuration()
        self.defaults = self.backend.defaults()
        self.cmd_def = CmdDef.from_defaults(self.defaults.cmd_def,
                                            self.defaults.pulse_library)
        self.short_pulse = pulse.SamplePulse(samples=np.array([0.02739068], dtype=np.complex128),
                                             name='p0')
        acquire = pulse.Acquire(5)
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, self.short_pulse(self.device.q[0].drive))
        self.sched = sched.insert(5, acquire(self.device.q, self.device.mem))

    def test_add_implicit(self):
        """Test that implicit acquires are made explicit according to the meas map."""
        sched = add_implicit_acquires(self.sched, [[0, 1]])
        self.assertEqual({a.index for a in sched.instructions[1][1].acquires}, {0, 1})

    def test_add_across_meas_map_sublists(self):
        """Test that implicit acquires in separate meas map sublists are all added."""
        sched = add_implicit_acquires(self.sched, [[0, 2], [1, 3]])
        self.assertEqual({a.index for a in sched.instructions[1][1].acquires}, {0, 2, 1, 3})

    def test_dont_add_all(self):
        """Test that acquires aren't added if no qubits in the sublist aren't being acquired."""
        sched = add_implicit_acquires(self.sched, [[4, 5], [0, 2], [1, 3]])
        self.assertEqual({a.index for a in sched.instructions[1][1].acquires}, {0, 2, 1, 3})


if __name__ == '__main__':
    unittest.main()
