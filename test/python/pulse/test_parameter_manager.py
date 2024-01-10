# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Test cases for parameter manager."""

from qiskit import pulse
from qiskit.circuit import Parameter
from qiskit.pulse.exceptions import PulseError, UnassignedDurationError
from qiskit.test import QiskitTestCase


class TestAssignFromProgram(QiskitTestCase):
    """Test managing parameters from programs. Parameter manager is implicitly called."""

    def test_attribute_parameters(self):
        """Test the ``parameter`` attributes."""
        sigma = Parameter("sigma")
        amp = Parameter("amp")

        waveform = pulse.library.Gaussian(duration=128, sigma=sigma, amp=amp)

        block = pulse.ScheduleBlock()
        block += pulse.Play(waveform, pulse.DriveChannel(10))

        ref_set = {amp, sigma}

        self.assertSetEqual(set(block.parameters), ref_set)

    def test_parametric_pulses(self):
        """Test Parametric Pulses with parameters determined by ParameterExpressions
        in the Play instruction."""
        sigma = Parameter("sigma")
        amp = Parameter("amp")

        waveform = pulse.library.Gaussian(duration=128, sigma=sigma, amp=amp)

        block = pulse.ScheduleBlock()
        block += pulse.Play(waveform, pulse.DriveChannel(10))
        block.assign_parameters({amp: 0.2, sigma: 4}, inplace=True)

        self.assertEqual(block.blocks[0].pulse.amp, 0.2)
        self.assertEqual(block.blocks[0].pulse.sigma, 4.0)


class TestScheduleTimeslots(QiskitTestCase):
    """Test for edge cases of timing overlap on parametrized channels.

    Note that this test is dedicated to `Schedule` since `ScheduleBlock` implicitly
    assigns instruction time t0 that doesn't overlap with existing instructions.
    """

    def test_overlapping_pulses(self):
        """Test that an error is still raised when overlapping instructions are assigned."""
        param_idx = Parameter("q")

        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(param_idx))
        with self.assertRaises(PulseError):
            schedule |= pulse.Play(
                pulse.Waveform([0.5, 0.5, 0.5, 0.5]), pulse.DriveChannel(param_idx)
            )

    def test_overlapping_on_assignment(self):
        """Test that assignment will catch against existing instructions."""
        param_idx = Parameter("q")

        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(1))
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(param_idx))
        with self.assertRaises(PulseError):
            schedule.assign_parameters({param_idx: 1})

    def test_overlapping_on_expression_assigment_to_zero(self):
        """Test constant*zero expression conflict."""
        param_idx = Parameter("q")

        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(param_idx))
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(2 * param_idx))
        with self.assertRaises(PulseError):
            schedule.assign_parameters({param_idx: 0})

    def test_merging_upon_assignment(self):
        """Test that schedule can match instructions on a channel."""
        param_idx = Parameter("q")

        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(1))
        schedule = schedule.insert(
            4, pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(param_idx))
        )
        schedule.assign_parameters({param_idx: 1})

        self.assertEqual(schedule.ch_duration(pulse.DriveChannel(1)), 8)
        self.assertEqual(schedule.channels, (pulse.DriveChannel(1),))

    def test_overlapping_on_multiple_assignment(self):
        """Test that assigning one qubit then another raises error when overlapping."""
        param_idx1 = Parameter("q1")
        param_idx2 = Parameter("q2")

        schedule = pulse.Schedule()
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(param_idx1))
        schedule |= pulse.Play(pulse.Waveform([1, 1, 1, 1]), pulse.DriveChannel(param_idx2))
        schedule.assign_parameters({param_idx1: 2})

        with self.assertRaises(PulseError):
            schedule.assign_parameters({param_idx2: 2})

    def test_cannot_build_schedule_with_unassigned_duration(self):
        """Test we cannot build schedule with parameterized instructions"""
        dur = Parameter("dur")
        ch = pulse.DriveChannel(0)

        test_play = pulse.Play(pulse.Gaussian(dur, 0.1, dur / 4), ch)

        sched = pulse.Schedule()
        with self.assertRaises(UnassignedDurationError):
            sched.insert(0, test_play)
