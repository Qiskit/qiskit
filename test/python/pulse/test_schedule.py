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

"""Test cases for the pulse schedule."""
import unittest
from unittest.mock import patch

import numpy as np

from qiskit.pulse import (
    Play,
    Waveform,
    ShiftPhase,
    Acquire,
    Snapshot,
    Delay,
    Gaussian,
    Drag,
    GaussianSquare,
    Constant,
    functional_pulse,
)
from qiskit.pulse.channels import (
    MemorySlot,
    DriveChannel,
    ControlChannel,
    AcquireChannel,
    SnapshotChannel,
    MeasureChannel,
)
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.schedule import Schedule, _overlaps, _find_insertion_index
from test import QiskitTestCase  # pylint: disable=wrong-import-order
from qiskit.utils.deprecate_pulse import decorate_test_methods, ignore_pulse_deprecation_warnings


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class BaseTestSchedule(QiskitTestCase):
    """Schedule tests."""

    @ignore_pulse_deprecation_warnings
    def setUp(self):
        super().setUp()

        @functional_pulse
        def linear(duration, slope, intercept):
            x = np.linspace(0, duration - 1, duration)
            return slope * x + intercept

        self.linear = linear


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestScheduleBuilding(BaseTestSchedule):
    """Test construction of schedules."""

    def test_empty_schedule(self):
        """Test empty schedule."""
        sched = Schedule()
        self.assertEqual(0, sched.start_time)
        self.assertEqual(0, sched.stop_time)
        self.assertEqual(0, sched.duration)
        self.assertEqual(0, len(sched))
        self.assertEqual((), sched.children)
        self.assertEqual({}, sched.timeslots)
        self.assertEqual([], list(sched.instructions))
        self.assertFalse(sched)

    def test_overlapping_schedules(self):
        """Test overlapping schedules."""

        def my_test_make_schedule(acquire: int, memoryslot: int, shift: int):
            sched1 = Acquire(acquire, AcquireChannel(0), MemorySlot(memoryslot))
            sched2 = Acquire(acquire, AcquireChannel(1), MemorySlot(memoryslot)).shift(shift)

            return Schedule(sched1, sched2)

        self.assertIsInstance(my_test_make_schedule(4, 0, 4), Schedule)
        self.assertRaisesRegex(
            PulseError, r".*MemorySlot\(0\).*overlaps .*", my_test_make_schedule, 4, 0, 2
        )
        self.assertRaisesRegex(
            PulseError, r".*MemorySlot\(1\).*overlaps .*", my_test_make_schedule, 4, 1, 0
        )

    @patch("qiskit.utils.is_main_process", return_value=True)
    def test_auto_naming(self, is_main_process_mock):
        """Test that a schedule gets a default name, incremented per instance"""

        del is_main_process_mock

        sched_0 = Schedule()
        sched_0_name_count = int(sched_0.name[len("sched") :])

        sched_1 = Schedule()
        sched_1_name_count = int(sched_1.name[len("sched") :])
        self.assertEqual(sched_1_name_count, sched_0_name_count + 1)

        sched_2 = Schedule()
        sched_2_name_count = int(sched_2.name[len("sched") :])
        self.assertEqual(sched_2_name_count, sched_1_name_count + 1)

    def test_parametric_commands_in_sched(self):
        """Test that schedules can be built with parametric commands."""
        sched = Schedule(name="test_parametric")
        sched += Play(Gaussian(duration=25, sigma=4, amp=0.5, angle=np.pi / 2), DriveChannel(0))
        sched += Play(Drag(duration=25, amp=0.4, angle=0.5, sigma=7.8, beta=4), DriveChannel(1))
        sched += Play(Constant(duration=25, amp=1), DriveChannel(2))
        sched_duration = sched.duration
        sched += (
            Play(GaussianSquare(duration=1500, amp=0.2, sigma=8, width=140), MeasureChannel(0))
            << sched_duration
        )
        self.assertEqual(sched.duration, 1525)
        self.assertTrue("sigma" in sched.instructions[0][1].pulse.parameters)

    def test_numpy_integer_input(self):
        """Test that mixed integer duration types can build a schedule (#5754)."""
        sched = Schedule()
        sched += Delay(np.int32(25), DriveChannel(0))
        sched += Play(Constant(duration=30, amp=0.1), DriveChannel(0))
        self.assertEqual(sched.duration, 55)

    def test_negative_time_raises(self):
        """Test that a negative time will raise an error."""
        sched = Schedule()
        sched += Delay(1, DriveChannel(0))
        with self.assertRaises(PulseError):
            sched.shift(-10)

    def test_shift_float_time_raises(self):
        """Test that a floating time will raise an error with shift."""
        sched = Schedule()
        sched += Delay(1, DriveChannel(0))
        with self.assertRaises(PulseError):
            sched.shift(0.1)

    def test_insert_float_time_raises(self):
        """Test that a floating time will raise an error with insert."""
        sched = Schedule()
        sched += Delay(1, DriveChannel(0))
        with self.assertRaises(PulseError):
            sched.insert(10.1, sched)

    def test_shift_unshift(self):
        """Test shift and then unshifting of schedule"""
        reference_sched = Schedule()
        reference_sched += Delay(10, DriveChannel(0))
        shifted_sched = reference_sched.shift(10).shift(-10)
        self.assertEqual(shifted_sched, reference_sched)

    def test_duration(self):
        """Test schedule.duration."""
        reference_sched = Schedule()
        reference_sched = reference_sched.insert(10, Delay(10, DriveChannel(0)))
        reference_sched = reference_sched.insert(10, Delay(50, DriveChannel(1)))
        reference_sched = reference_sched.insert(10, ShiftPhase(0.1, DriveChannel(0)))

        reference_sched = reference_sched.insert(100, ShiftPhase(0.1, DriveChannel(1)))

        self.assertEqual(reference_sched.duration, 100)
        self.assertEqual(reference_sched.duration, 100)

    def test_ch_duration(self):
        """Test schedule.ch_duration."""
        reference_sched = Schedule()
        reference_sched = reference_sched.insert(10, Delay(10, DriveChannel(0)))
        reference_sched = reference_sched.insert(10, Delay(50, DriveChannel(1)))
        reference_sched = reference_sched.insert(10, ShiftPhase(0.1, DriveChannel(0)))

        reference_sched = reference_sched.insert(100, ShiftPhase(0.1, DriveChannel(1)))

        self.assertEqual(reference_sched.ch_duration(DriveChannel(0)), 20)
        self.assertEqual(reference_sched.ch_duration(DriveChannel(1)), 100)
        self.assertEqual(
            reference_sched.ch_duration(*reference_sched.channels), reference_sched.duration
        )

    def test_ch_start_time(self):
        """Test schedule.ch_start_time."""
        reference_sched = Schedule()
        reference_sched = reference_sched.insert(10, Delay(10, DriveChannel(0)))
        reference_sched = reference_sched.insert(10, Delay(50, DriveChannel(1)))
        reference_sched = reference_sched.insert(10, ShiftPhase(0.1, DriveChannel(0)))

        reference_sched = reference_sched.insert(100, ShiftPhase(0.1, DriveChannel(1)))

        self.assertEqual(reference_sched.ch_start_time(DriveChannel(0)), 10)
        self.assertEqual(reference_sched.ch_start_time(DriveChannel(1)), 10)

    def test_ch_stop_time(self):
        """Test schedule.ch_stop_time."""
        reference_sched = Schedule()
        reference_sched = reference_sched.insert(10, Delay(10, DriveChannel(0)))
        reference_sched = reference_sched.insert(10, Delay(50, DriveChannel(1)))
        reference_sched = reference_sched.insert(10, ShiftPhase(0.1, DriveChannel(0)))

        reference_sched = reference_sched.insert(100, ShiftPhase(0.1, DriveChannel(1)))

        self.assertEqual(reference_sched.ch_stop_time(DriveChannel(0)), 20)
        self.assertEqual(reference_sched.ch_stop_time(DriveChannel(1)), 100)

    def test_timeslots(self):
        """Test schedule.timeslots."""
        reference_sched = Schedule()
        reference_sched = reference_sched.insert(10, Delay(10, DriveChannel(0)))
        reference_sched = reference_sched.insert(10, Delay(50, DriveChannel(1)))
        reference_sched = reference_sched.insert(10, ShiftPhase(0.1, DriveChannel(0)))

        reference_sched = reference_sched.insert(100, ShiftPhase(0.1, DriveChannel(1)))

        self.assertEqual(reference_sched.timeslots[DriveChannel(0)], [(10, 10), (10, 20)])
        self.assertEqual(reference_sched.timeslots[DriveChannel(1)], [(10, 60), (100, 100)])

    def test_inherit_from(self):
        """Test creating schedule with another schedule."""
        ref_metadata = {"test": "value"}
        ref_name = "test"

        base_sched = Schedule(name=ref_name, metadata=ref_metadata)
        new_sched = Schedule.initialize_from(base_sched)

        self.assertEqual(new_sched.name, ref_name)
        self.assertDictEqual(new_sched.metadata, ref_metadata)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestReplace(BaseTestSchedule):
    """Test schedule replacement."""

    def test_replace_instruction(self):
        """Test replacement of simple instruction"""
        old = Play(Constant(100, 1.0), DriveChannel(0))
        new = Play(Constant(100, 0.1), DriveChannel(0))

        sched = Schedule(old)
        new_sched = sched.replace(old, new)

        self.assertEqual(new_sched, Schedule(new))

        # test replace inplace
        sched.replace(old, new, inplace=True)
        self.assertEqual(sched, Schedule(new))

    def test_replace_schedule(self):
        """Test replacement of schedule."""

        old = Schedule(
            Delay(10, DriveChannel(0)),
            Delay(100, DriveChannel(1)),
        )
        new = Schedule(
            Play(Constant(10, 1.0), DriveChannel(0)),
            Play(Constant(100, 0.1), DriveChannel(1)),
        )
        const = Play(Constant(100, 1.0), DriveChannel(0))

        sched = Schedule()
        sched += const
        sched += old

        new_sched = sched.replace(old, new)

        ref_sched = Schedule()
        ref_sched += const
        ref_sched += new
        self.assertEqual(new_sched, ref_sched)

        # test replace inplace
        sched.replace(old, new, inplace=True)
        self.assertEqual(sched, ref_sched)

    def test_replace_fails_on_overlap(self):
        """Test that replacement fails on overlap."""
        old = Play(Constant(20, 1.0), DriveChannel(0))
        new = Play(Constant(100, 0.1), DriveChannel(0))

        sched = Schedule()
        sched += old
        sched += Delay(100, DriveChannel(0))

        with self.assertRaises(PulseError):
            sched.replace(old, new)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestDelay(BaseTestSchedule):
    """Test Delay Instruction"""

    def setUp(self):
        super().setUp()
        self.delay_time = 10

    def test_delay_snapshot_channel(self):
        """Test Delay on DriveChannel"""

        snapshot_ch = SnapshotChannel()
        snapshot = Snapshot(label="test")
        # should pass as is an append
        sched = Delay(self.delay_time, snapshot_ch) + snapshot
        self.assertIsInstance(sched, Schedule)
        # should fail due to overlap
        with self.assertRaises(PulseError):
            sched = Delay(self.delay_time, snapshot_ch) | snapshot << 5
            self.assertIsInstance(sched, Schedule)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestScheduleFilter(BaseTestSchedule):
    """Test Schedule filtering methods"""

    def test_filter_exclude_name(self):
        """Test the name of the schedules after applying filter and exclude functions."""
        sched = Schedule(name="test-schedule")
        sched = sched.insert(10, Acquire(5, AcquireChannel(0), MemorySlot(0)))
        sched = sched.insert(10, Acquire(5, AcquireChannel(1), MemorySlot(1)))
        excluded = sched.exclude(channels=[AcquireChannel(0)])
        filtered = sched.filter(channels=[AcquireChannel(1)])

        # check if the excluded and filtered schedule have the same name as sched
        self.assertEqual(sched.name, filtered.name)
        self.assertEqual(sched.name, excluded.name)


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestScheduleEquality(BaseTestSchedule):
    """Test equality of schedules."""

    def test_different_channels(self):
        """Test equality is False if different channels."""
        self.assertNotEqual(
            Schedule(ShiftPhase(0, DriveChannel(0))), Schedule(ShiftPhase(0, DriveChannel(1)))
        )

    def test_same_time_equal(self):
        """Test equal if instruction at same time."""

        self.assertEqual(
            Schedule((0, ShiftPhase(0, DriveChannel(1)))),
            Schedule((0, ShiftPhase(0, DriveChannel(1)))),
        )

    def test_different_time_not_equal(self):
        """Test that not equal if instruction at different time."""
        self.assertNotEqual(
            Schedule((0, ShiftPhase(0, DriveChannel(1)))),
            Schedule((1, ShiftPhase(0, DriveChannel(1)))),
        )

    def test_single_channel_out_of_order(self):
        """Test that schedule with single channel equal when out of order."""
        instructions = [
            (0, ShiftPhase(0, DriveChannel(0))),
            (15, Play(Waveform(np.ones(10)), DriveChannel(0))),
            (5, Play(Waveform(np.ones(10)), DriveChannel(0))),
        ]

        self.assertEqual(Schedule(*instructions), Schedule(*reversed(instructions)))

    def test_multiple_channels_out_of_order(self):
        """Test that schedule with multiple channels equal when out of order."""
        instructions = [
            (0, ShiftPhase(0, DriveChannel(1))),
            (1, Acquire(10, AcquireChannel(0), MemorySlot(1))),
        ]

        self.assertEqual(Schedule(*instructions), Schedule(*reversed(instructions)))

    def test_same_commands_on_two_channels_at_same_time_out_of_order(self):
        """Test that schedule with same commands on two channels at the same time equal
        when out of order."""
        sched1 = Schedule()
        sched1 = sched1.append(Delay(100, DriveChannel(1)))
        sched1 = sched1.append(Delay(100, ControlChannel(1)))
        sched2 = Schedule()
        sched2 = sched2.append(Delay(100, ControlChannel(1)))
        sched2 = sched2.append(Delay(100, DriveChannel(1)))
        self.assertEqual(sched1, sched2)

    def test_different_name_equal(self):
        """Test that names are ignored when checking equality."""

        self.assertEqual(
            Schedule((0, ShiftPhase(0, DriveChannel(1), name="fc1")), name="s1"),
            Schedule((0, ShiftPhase(0, DriveChannel(1), name="fc2")), name="s2"),
        )


@decorate_test_methods(ignore_pulse_deprecation_warnings)
class TestTimingUtils(QiskitTestCase):
    """Test the Schedule helper functions."""

    def test_overlaps(self):
        """Test the `_overlaps` function."""
        a = (0, 1)
        b = (1, 4)
        c = (2, 3)
        d = (3, 5)
        self.assertFalse(_overlaps(a, b))
        self.assertFalse(_overlaps(b, a))
        self.assertFalse(_overlaps(a, d))
        self.assertTrue(_overlaps(b, c))
        self.assertTrue(_overlaps(c, b))
        self.assertTrue(_overlaps(b, d))
        self.assertTrue(_overlaps(d, b))

    def test_overlaps_zero_duration(self):
        """Test the `_overlaps` function for intervals with duration zero."""
        a = 0
        b = 1
        self.assertFalse(_overlaps((a, a), (a, a)))
        self.assertFalse(_overlaps((a, a), (a, b)))
        self.assertFalse(_overlaps((a, b), (a, a)))
        self.assertFalse(_overlaps((a, b), (b, b)))
        self.assertFalse(_overlaps((b, b), (a, b)))
        self.assertTrue(_overlaps((a, a + 2), (a + 1, a + 1)))
        self.assertTrue(_overlaps((a + 1, a + 1), (a, a + 2)))

    def test_find_insertion_index(self):
        """Test the `_find_insertion_index` function."""
        intervals = [(1, 2), (4, 5)]
        self.assertEqual(_find_insertion_index(intervals, (2, 3)), 1)
        self.assertEqual(_find_insertion_index(intervals, (3, 4)), 1)
        self.assertEqual(intervals, [(1, 2), (4, 5)])
        intervals = [(1, 2), (4, 5), (6, 7)]
        self.assertEqual(_find_insertion_index(intervals, (2, 3)), 1)
        self.assertEqual(_find_insertion_index(intervals, (0, 1)), 0)
        self.assertEqual(_find_insertion_index(intervals, (5, 6)), 2)
        self.assertEqual(_find_insertion_index(intervals, (8, 9)), 3)

        longer_intervals = [(1, 2), (2, 3), (4, 5), (5, 6), (7, 9), (11, 11)]
        self.assertEqual(_find_insertion_index(longer_intervals, (4, 4)), 2)
        self.assertEqual(_find_insertion_index(longer_intervals, (5, 5)), 3)
        self.assertEqual(_find_insertion_index(longer_intervals, (3, 4)), 2)
        self.assertEqual(_find_insertion_index(longer_intervals, (3, 4)), 2)

        # test when two identical zero duration timeslots are present
        intervals = [(0, 10), (73, 73), (73, 73), (90, 101)]
        self.assertEqual(_find_insertion_index(intervals, (42, 73)), 1)
        self.assertEqual(_find_insertion_index(intervals, (73, 81)), 3)

    def test_find_insertion_index_when_overlapping(self):
        """Test that `_find_insertion_index` raises an error when the new_interval _overlaps."""
        intervals = [(10, 20), (44, 55), (60, 61), (80, 1000)]
        with self.assertRaises(PulseError):
            _find_insertion_index(intervals, (60, 62))
        with self.assertRaises(PulseError):
            _find_insertion_index(intervals, (100, 1500))

        intervals = [(0, 1), (10, 15)]
        with self.assertRaises(PulseError):
            _find_insertion_index(intervals, (7, 13))

    def test_find_insertion_index_empty_list(self):
        """Test that the insertion index is properly found for empty lists."""
        self.assertEqual(_find_insertion_index([], (0, 1)), 0)


if __name__ == "__main__":
    unittest.main()
