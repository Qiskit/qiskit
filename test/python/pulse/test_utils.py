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
from qiskit.pulse.cmd_def import CmdDef
from qiskit.pulse.commands import AcquireInstruction
from qiskit.pulse.exceptions import PulseError
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOpenPulse2Q

from qiskit.pulse.utils import add_implicit_acquires, align_measures


class TestAutoMerge(QiskitTestCase):
    """Test the helper function which aligns acquires."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()
        self.device = pulse.PulseChannelSpec.from_backend(self.backend)
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
        sched = sched.insert(0, self.short_pulse(self.device.drives[0]))
        sched = sched.insert(1, acquire(self.device.acquires[0], self.device.memoryslots[0]))
        sched = sched.insert(10, acquire(self.device.acquires[1], self.device.memoryslots[1]))
        sched = align_measures([sched], self.cmd_def)[0]
        for time, inst in sched.instructions:
            if isinstance(inst, AcquireInstruction):
                self.assertEqual(time, 10)
        sched = align_measures([sched], self.cmd_def, align_time=20)[0]
        for time, inst in sched.instructions:
            if isinstance(inst, AcquireInstruction):
                self.assertEqual(time, 20)

    def test_align_post_u3(self):
        """Test that acquires are scheduled no sooner than the duration of the longest X gate.
        """
        acquire = pulse.Acquire(5)
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, self.short_pulse(self.device.drives[0]))
        sched = sched.insert(1, acquire(self.device.acquires[0], self.device.memoryslots[0]))
        sched = align_measures([sched], self.cmd_def)[0]
        for time, inst in sched.instructions:
            if isinstance(inst, AcquireInstruction):
                self.assertEqual(time, 4)
        sched = align_measures([sched], self.cmd_def, max_calibration_duration=10)[0]
        for time, inst in sched.instructions:
            if isinstance(inst, AcquireInstruction):
                self.assertEqual(time, 10)

    def test_error_multi_acquire(self):
        """Test that an error is raised if multiple acquires occur on the same channel."""
        acquire = pulse.Acquire(5)
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, self.short_pulse(self.device.drives[0]))
        sched = sched.insert(4, acquire(self.device.acquires[0], self.device.memoryslots[0]))
        sched = sched.insert(10, acquire(self.device.acquires[0], self.device.memoryslots[0]))
        with self.assertRaises(PulseError):
            align_measures([sched], self.cmd_def)

    def test_error_post_acquire_pulse(self):
        """Test that an error is raised if a pulse occurs on a channel after an acquire."""
        acquire = pulse.Acquire(5)
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, self.short_pulse(self.device.drives[0]))
        sched = sched.insert(4, acquire(self.device.acquires[0], self.device.memoryslots[0]))
        # No error with separate channel
        sched = sched.insert(10, self.short_pulse(self.device.drives[1]))
        align_measures([sched], self.cmd_def)
        sched = sched.insert(10, self.short_pulse(self.device.drives[0]))
        with self.assertRaises(PulseError):
            align_measures([sched], self.cmd_def)

    def test_align_across_schedules(self):
        """Test that acquires are aligned together across multiple schedules."""
        acquire = pulse.Acquire(5)
        sched1 = pulse.Schedule(name='fake_experiment')
        sched1 = sched1.insert(0, self.short_pulse(self.device.drives[0]))
        sched1 = sched1.insert(10, acquire(self.device.acquires[0], self.device.memoryslots[0]))
        sched2 = pulse.Schedule(name='fake_experiment')
        sched2 = sched2.insert(3, self.short_pulse(self.device.drives[0]))
        sched2 = sched2.insert(25, acquire(self.device.acquires[0], self.device.memoryslots[0]))
        schedules = align_measures([sched1, sched2], self.cmd_def)
        for time, inst in schedules[0].instructions:
            if isinstance(inst, AcquireInstruction):
                self.assertEqual(time, 25)
        for time, inst in schedules[0].instructions:
            if isinstance(inst, AcquireInstruction):
                self.assertEqual(time, 25)


class TestAddImplicitAcquires(QiskitTestCase):
    """Test the helper function which makes implicit acquires explicit."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()
        self.device = pulse.PulseChannelSpec.from_backend(self.backend)
        self.config = self.backend.configuration()
        self.defaults = self.backend.defaults()
        self.cmd_def = CmdDef.from_defaults(self.defaults.cmd_def,
                                            self.defaults.pulse_library)
        self.short_pulse = pulse.SamplePulse(samples=np.array([0.02739068], dtype=np.complex128),
                                             name='p0')
        acquire = pulse.Acquire(5)
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, self.short_pulse(self.device.drives[0]))
        self.sched = sched.insert(5, acquire(self.device.acquires, self.device.memoryslots))

    def test_add_implicit(self):
        """Test that implicit acquires are made explicit according to the meas map."""
        sched = add_implicit_acquires(self.sched, [[0, 1]])
        acquired_qubits = set()
        for _, inst in sched.instructions:
            if isinstance(inst, AcquireInstruction):
                acquired_qubits.update({a.index for a in inst.acquires})
        self.assertEqual(acquired_qubits, {0, 1})

    def test_add_across_meas_map_sublists(self):
        """Test that implicit acquires in separate meas map sublists are all added."""
        sched = add_implicit_acquires(self.sched, [[0, 2], [1, 3]])
        acquired_qubits = set()
        for _, inst in sched.instructions:
            if isinstance(inst, AcquireInstruction):
                acquired_qubits.update({a.index for a in inst.acquires})
        self.assertEqual(acquired_qubits, {0, 1, 2, 3})

    def test_dont_add_all(self):
        """Test that acquires aren't added if no qubits in the sublist aren't being acquired."""
        sched = add_implicit_acquires(self.sched, [[4, 5], [0, 2], [1, 3]])
        acquired_qubits = set()
        for _, inst in sched.instructions:
            if isinstance(inst, AcquireInstruction):
                acquired_qubits.update({a.index for a in inst.acquires})
        self.assertEqual(acquired_qubits, {0, 1, 2, 3})


class TestAutoMergeWithDeviceSpecification(QiskitTestCase):
    """Test the helper function which aligns acquires."""
    # TODO: This test will be deprecated in future update.

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
        sched = align_measures([sched], self.cmd_def)[0]
        for time, inst in sched.instructions:
            if isinstance(inst, AcquireInstruction):
                self.assertEqual(time, 10)
        sched = align_measures([sched], self.cmd_def, align_time=20)[0]
        for time, inst in sched.instructions:
            if isinstance(inst, AcquireInstruction):
                self.assertEqual(time, 20)

    def test_align_post_u3(self):
        """Test that acquires are scheduled no sooner than the duration of the longest X gate.
        """
        acquire = pulse.Acquire(5)
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, self.short_pulse(self.device.q[0].drive))
        sched = sched.insert(1, acquire(self.device.q[0], self.device.mem[0]))
        sched = align_measures([sched], self.cmd_def)[0]
        for time, inst in sched.instructions:
            if isinstance(inst, AcquireInstruction):
                self.assertEqual(time, 4)
        sched = align_measures([sched], self.cmd_def, max_calibration_duration=10)[0]
        for time, inst in sched.instructions:
            if isinstance(inst, AcquireInstruction):
                self.assertEqual(time, 10)

    def test_error_multi_acquire(self):
        """Test that an error is raised if multiple acquires occur on the same channel."""
        acquire = pulse.Acquire(5)
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, self.short_pulse(self.device.q[0].drive))
        sched = sched.insert(4, acquire(self.device.q[0], self.device.mem[0]))
        sched = sched.insert(10, acquire(self.device.q[0], self.device.mem[0]))
        with self.assertRaises(PulseError):
            align_measures([sched], self.cmd_def)

    def test_error_post_acquire_pulse(self):
        """Test that an error is raised if a pulse occurs on a channel after an acquire."""
        acquire = pulse.Acquire(5)
        sched = pulse.Schedule(name='fake_experiment')
        sched = sched.insert(0, self.short_pulse(self.device.q[0].drive))
        sched = sched.insert(4, acquire(self.device.q[0], self.device.mem[0]))
        # No error with separate channel
        sched = sched.insert(10, self.short_pulse(self.device.q[1].drive))
        align_measures([sched], self.cmd_def)
        sched = sched.insert(10, self.short_pulse(self.device.q[0].drive))
        with self.assertRaises(PulseError):
            align_measures([sched], self.cmd_def)

    def test_align_across_schedules(self):
        """Test that acquires are aligned together across multiple schedules."""
        acquire = pulse.Acquire(5)
        sched1 = pulse.Schedule(name='fake_experiment')
        sched1 = sched1.insert(0, self.short_pulse(self.device.q[0].drive))
        sched1 = sched1.insert(10, acquire(self.device.q[0], self.device.mem[0]))
        sched2 = pulse.Schedule(name='fake_experiment')
        sched2 = sched2.insert(3, self.short_pulse(self.device.q[0].drive))
        sched2 = sched2.insert(25, acquire(self.device.q[0], self.device.mem[0]))
        schedules = align_measures([sched1, sched2], self.cmd_def)
        for time, inst in schedules[0].instructions:
            if isinstance(inst, AcquireInstruction):
                self.assertEqual(time, 25)
        for time, inst in schedules[0].instructions:
            if isinstance(inst, AcquireInstruction):
                self.assertEqual(time, 25)


class TestAddImplicitAcquiresWithDeviceSpecification(QiskitTestCase):
    """Test the helper function which makes implicit acquires explicit."""
    # TODO: This test will be deprecated in future update.

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
        acquired_qubits = set()
        for _, inst in sched.instructions:
            if isinstance(inst, AcquireInstruction):
                acquired_qubits.update({a.index for a in inst.acquires})
        self.assertEqual(acquired_qubits, {0, 1})

    def test_add_across_meas_map_sublists(self):
        """Test that implicit acquires in separate meas map sublists are all added."""
        sched = add_implicit_acquires(self.sched, [[0, 2], [1, 3]])
        acquired_qubits = set()
        for _, inst in sched.instructions:
            if isinstance(inst, AcquireInstruction):
                acquired_qubits.update({a.index for a in inst.acquires})
        self.assertEqual(acquired_qubits, {0, 1, 2, 3})

    def test_dont_add_all(self):
        """Test that acquires aren't added if no qubits in the sublist aren't being acquired."""
        sched = add_implicit_acquires(self.sched, [[4, 5], [0, 2], [1, 3]])
        acquired_qubits = set()
        for _, inst in sched.instructions:
            if isinstance(inst, AcquireInstruction):
                acquired_qubits.update({a.index for a in inst.acquires})
        self.assertEqual(acquired_qubits, {0, 1, 2, 3})


if __name__ == '__main__':
    unittest.main()
