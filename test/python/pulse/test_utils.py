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

    def test_align_measures(self):
        short_pulse = pulse.SamplePulse(samples=np.array([0.02739068], dtype=np.complex128),
                                        name='p0')
        acquire = pulse.Acquire(5)
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, short_pulse(self.device.q[0].drive))
        sched = sched.insert(1, acquire(self.device.q[0], self.device.mem[0]))
        sched = sched.insert(10, acquire(self.device.q[1], self.device.mem[1]))
        sched = align_measures(sched,
                               CmdDef.from_defaults(self.defaults.cmd_def,
                                                    self.defaults.pulse_library))
        self.assertEqual(sched.instructions[1][0], 10)
        self.assertEqual(sched.instructions[2][0], 10)


    def test_align_post_u2(self):
        pass

    def test_error_multi_acquire(self):
        pass

    def test_error_post_acquire_pulse(self):
        pass


class TestAddImplicitAcquires(QiskitTestCase):
    """Test the helper function which makes implicit acquires explicit."""

    def setUp(self):
        pass

    def test_add_implicit(self):
        pass

    def test_add_across_meas_map_sublists(self):
        pass

    def test_dont_add_all(self):
        pass


if __name__ == '__main__':
    unittest.main()
