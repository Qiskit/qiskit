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

from qiskit.pulse import library
from qiskit.pulse.channels import (DriveChannel, MeasureChannel, ControlChannel, AcquireChannel,
                                   MemorySlot, RegisterSlot)
from qiskit.pulse.instructions import (SetFrequency, Play, Acquire, Delay, Snapshot, ShiftFrequency,
                                       SetPhase, ShiftPhase)
from qiskit.pulse.schedule import Schedule
from qiskit.tools.visualization import HAS_MATPLOTLIB
from qiskit.visualization import pulse_drawer

from .visualization import QiskitVisualizationTestCase, path_to_diagram_reference


class TestPulseVisualizationImplementation(QiskitVisualizationTestCase):
    """Visual accuracy of visualization tools outputs tests."""

    pulse_matplotlib_reference = path_to_diagram_reference('pulse_matplotlib_ref.png')
    instr_matplotlib_reference = path_to_diagram_reference('instruction_matplotlib_ref.png')
    schedule_matplotlib_reference = path_to_diagram_reference('schedule_matplotlib_ref.png')
    trunc_sched_mpl_reference = path_to_diagram_reference('truncated_schedule_matplotlib_ref.png')
    schedule_show_framechange_ref = path_to_diagram_reference('schedule_show_framechange_ref.png')
    parametric_matplotlib_reference = path_to_diagram_reference('parametric_matplotlib_ref.png')

    def setUp(self):
        self.schedule = Schedule(name='test_schedule')

    def sample_pulse(self):
        """Generate a sample pulse."""
        return library.gaussian(20, 0.8, 1.0, name='test')

    def sample_instruction(self):
        """Generate a sample instruction."""
        return self.sample_pulse()(DriveChannel(0))

    def sample_schedule(self):
        """Generate a sample schedule that includes the most common elements of
           pulse schedules."""
        gp0 = library.gaussian(duration=20, amp=1.0, sigma=1.0)
        gp1 = library.gaussian(duration=20, amp=-1.0, sigma=2.0)
        gs0 = library.gaussian_square(duration=20, amp=-1.0, sigma=2.0, risefall=3)

        sched = Schedule(name='test_schedule')
        sched = sched.append(gp0(DriveChannel(0)))
        sched = sched.insert(0, library.Constant(duration=60, amp=0.2 + 0.4j)(
            ControlChannel(0)))
        sched = sched.insert(60, ShiftPhase(-1.57, DriveChannel(0)))
        sched = sched.insert(60, SetFrequency(8.0, DriveChannel(0)))
        sched = sched.insert(60, SetPhase(3.14, DriveChannel(0)))
        sched = sched.insert(70, ShiftFrequency(4.0e6, DriveChannel(0)))
        sched = sched.insert(30, Play(gp1, DriveChannel(1)))
        sched = sched.insert(60, Play(gp0, ControlChannel(0)))
        sched = sched.insert(60, Play(gs0, MeasureChannel(0)))
        sched = sched.insert(90, ShiftPhase(1.57, DriveChannel(0)))
        sched = sched.insert(90, Acquire(10,
                                         AcquireChannel(1),
                                         MemorySlot(1),
                                         RegisterSlot(1)))
        sched = sched.append(Delay(100, DriveChannel(0)))
        sched = sched + sched
        sched |= Snapshot("snapshot_1", "snap_type") << 60
        sched |= Snapshot("snapshot_2", "snap_type") << 120
        return sched

    @unittest.skipIf(not HAS_MATPLOTLIB, 'matplotlib not available.')
    @unittest.skip('Useful for refactoring purposes, skipping by default.')
    def test_parametric_pulse_schedule(self):
        """Test that parametric instructions/schedules can be drawn."""
        filename = self._get_resource_path('current_parametric_matplotlib_ref.png')
        schedule = Schedule(name='test_parametric')
        schedule += library.Gaussian(duration=25, sigma=4, amp=0.5j)(DriveChannel(0))
        pulse_drawer(schedule, filename=filename)
        self.assertImagesAreEqual(filename, self.parametric_matplotlib_reference)
        os.remove(filename)

    @unittest.skipIf(not HAS_MATPLOTLIB, 'matplotlib not available.')
    @unittest.skip('Useful for refactoring purposes, skipping by default.')
    def test_play(self):
        """Test that Play instructions can be drawn. The output should be the same as the
        parametric_pulse_schedule test.
        """
        filename = self._get_resource_path('current_play_matplotlib_ref.png')
        schedule = Schedule(name='test_parametric')
        schedule += Play(library.Gaussian(duration=25, sigma=4, amp=0.5j), DriveChannel(0))
        pulse_drawer(schedule, filename=filename)
        self.assertImagesAreEqual(filename, self.parametric_matplotlib_reference)
        os.remove(filename)

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
    def test_truncated_schedule_matplotlib_drawer(self):
        filename = self._get_resource_path('current_truncated_schedule_matplotlib_ref.png')
        schedule = self.sample_schedule()
        pulse_drawer(schedule, plot_range=(150, 300), filename=filename)
        self.assertImagesAreEqual(filename, self.trunc_sched_mpl_reference)
        os.remove(filename)

    # TODO: Enable for refactoring purposes and enable by default when we can
    # decide if the backend is available or not.
    @unittest.skipIf(not HAS_MATPLOTLIB, 'matplotlib not available.')
    @unittest.skip('Useful for refactoring purposes, skipping by default.')
    def test_truncate_acquisition(self):
        sched = Schedule(name='test_schedule')
        sched = sched.insert(0, Acquire(30, AcquireChannel(1),
                                        MemorySlot(1),
                                        RegisterSlot(1)))
        # Check ValueError is not thrown
        sched.draw(plot_range=(0, 15))

    # TODO: Enable for refactoring purposes and enable by default when we can
    # decide if the backend is available or not.
    @unittest.skipIf(not HAS_MATPLOTLIB, 'matplotlib not available.')
    @unittest.skip('Useful for refactoring purposes, skipping by default.')
    def test_schedule_drawer_show_framechange(self):
        filename = self._get_resource_path('current_show_framechange_ref.png')
        gp0 = library.gaussian(duration=20, amp=1.0, sigma=1.0)
        sched = Schedule(name='test_schedule')
        sched = sched.append(Play(gp0, DriveChannel(0)))
        sched = sched.insert(60, ShiftPhase(-1.57, DriveChannel(0)))
        sched = sched.insert(30, ShiftPhase(-1.50, DriveChannel(1)))
        sched = sched.insert(70, ShiftPhase(1.50, DriveChannel(1)))
        pulse_drawer(sched, filename=filename, show_framechange_channels=False)
        self.assertImagesAreEqual(filename, self.schedule_show_framechange_ref)
        os.remove(filename)


if __name__ == '__main__':
    unittest.main(verbosity=2)
