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

"""Tests for comparing the outputs of circuit drawer with expected ones."""

import os
import unittest

from qiskit.tools.visualization import HAS_MATPLOTLIB, pulse_drawer
from qiskit.pulse.channels import PulseChannelSpec
from qiskit.pulse.commands import FrameChange, Acquire, PersistentValue, Snapshot, Delay
from qiskit.pulse.schedule import Schedule
from qiskit.pulse import pulse_lib

from .visualization import QiskitVisualizationTestCase, path_to_diagram_reference


class TestPulseVisualizationImplementation(QiskitVisualizationTestCase):
    """Visual accuracy of visualization tools outputs tests."""

    pulse_matplotlib_reference = path_to_diagram_reference('pulse_matplotlib_ref.png')
    instr_matplotlib_reference = path_to_diagram_reference('instruction_matplotlib_ref.png')
    schedule_matplotlib_reference = path_to_diagram_reference('schedule_matplotlib_ref.png')

    def setUp(self):
        self.schedule = Schedule()
        self.device = PulseChannelSpec(n_qubits=2, n_control=1, n_registers=2)

    def sample_pulse(self):
        """Generate a sample pulse."""
        return pulse_lib.gaussian(20, 0.8, 1.0, name='test')

    def sample_instruction(self):
        """Generate a sample instruction."""
        return self.sample_pulse()(self.device.drives[0])

    def sample_schedule(self):
        """Generate a sample schedule that includes the most common elements of
           pulse schedules."""
        gp0 = pulse_lib.gaussian(duration=20, amp=1.0, sigma=1.0)
        gp1 = pulse_lib.gaussian(duration=20, amp=-1.0, sigma=2.0)
        gs0 = pulse_lib.gaussian_square(duration=20, amp=-1.0, sigma=2.0, risefall=3)

        fc_pi_2 = FrameChange(phase=1.57)
        acquire = Acquire(10)
        delay = Delay(100)
        sched = Schedule()
        sched = sched.append(gp0(self.device.drives[0]))
        sched = sched.insert(0, PersistentValue(value=0.2 + 0.4j)(self.device.controls[0]))
        sched = sched.insert(60, FrameChange(phase=-1.57)(self.device.drives[0]))
        sched = sched.insert(30, gp1(self.device.drives[1]))
        sched = sched.insert(60, gp0(self.device.controls[0]))
        sched = sched.insert(60, gs0(self.device.measures[0]))
        sched = sched.insert(90, fc_pi_2(self.device.drives[0]))
        sched = sched.insert(90, acquire(self.device.acquires[1],
                                         self.device.memoryslots[1],
                                         self.device.registers[1]))
        sched = sched.append(delay(self.device.drives[0]))
        sched = sched + sched
        sched |= Snapshot("snapshot_1", "snap_type") << 60
        sched |= Snapshot("snapshot_2", "snap_type") << 120
        return sched

    # TODO: Enable for refactoring purposes and enable by default when we can
    # decide if the backend is available or not.
    @unittest.skipIf(not HAS_MATPLOTLIB, 'matplotlib not available.')
    @unittest.skip('Useful for refactoring purposes, skipping by default.')
    def test_pulse_matplotlib_drawer(self):
        filename = self._get_resource_path('current_pulse_matplotlib_ref.png')
        pulse = self.sample_pulse()
        pulse_drawer(pulse, filename=filename)
        self.assertImagesAreEqual(filename, self.pulse_matplotlib_reference)
        os.remove(filename)

    # TODO: Enable for refactoring purposes and enable by default when we can
    # decide if the backend is available or not.
    @unittest.skipIf(not HAS_MATPLOTLIB, 'matplotlib not available.')
    @unittest.skip('Useful for refactoring purposes, skipping by default.')
    def test_instruction_matplotlib_drawer(self):
        filename = self._get_resource_path('current_instruction_matplotlib_ref.png')
        pulse_instruction = self.sample_instruction()
        pulse_drawer(pulse_instruction, filename=filename)
        self.assertImagesAreEqual(filename, self.instr_matplotlib_reference)
        os.remove(filename)

    # TODO: Enable for refactoring purposes and enable by default when we can
    # decide if the backend is available or not.
    @unittest.skipIf(not HAS_MATPLOTLIB, 'matplotlib not available.')
    @unittest.skip('Useful for refactoring purposes, skipping by default.')
    def test_schedule_matplotlib_drawer(self):
        filename = self._get_resource_path('current_schedule_matplotlib_ref.png')
        schedule = self.sample_schedule()
        pulse_drawer(schedule, filename=filename)
        self.assertImagesAreEqual(filename, self.schedule_matplotlib_reference)
        os.remove(filename)

    # TODO: Enable for refactoring purposes and enable by default when we can
    # decide if the backend is available or not.
    @unittest.skipIf(not HAS_MATPLOTLIB, 'matplotlib not available.')
    @unittest.skip('Useful for refactoring purposes, skipping by default.')
    def test_truncate_acquisition(self):
        sched = Schedule()
        acquire = Acquire(30)
        sched = sched.insert(0, acquire(self.device.acquires[1],
                                        self.device.memoryslots[1],
                                        self.device.registers[1]))
        # Check ValueError is not thrown
        sched.draw(plot_range=(0, 15))


if __name__ == '__main__':
    unittest.main(verbosity=2)
