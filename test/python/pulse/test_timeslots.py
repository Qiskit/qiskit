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

"""Test cases for the timeslots."""

import unittest

from qiskit.pulse.channels import AcquireChannel, DriveChannel
from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.timeslots import Interval, Timeslot, TimeslotCollection
from qiskit.test import QiskitTestCase


class TestInterval(QiskitTestCase):
    """Interval tests."""

    def test_default(self):
        """Test valid interval creation without error.
        """
        interval = Interval(1, 3)
        self.assertEqual(1, interval.start)
        self.assertEqual(3, interval.stop)
        self.assertEqual(2, interval.duration)

    def test_check_overlap(self):
        """Test valid interval creation without error.
        """
        self.assertEqual(True, Interval(1, 3).has_overlap(Interval(2, 4)))
        self.assertEqual(True, Interval(1, 3).has_overlap(Interval(2, 2)))
        self.assertEqual(False, Interval(1, 3).has_overlap(Interval(1, 1)))
        self.assertEqual(False, Interval(1, 3).has_overlap(Interval(3, 3)))
        self.assertEqual(False, Interval(1, 3).has_overlap(Interval(0, 1)))
        self.assertEqual(False, Interval(1, 3).has_overlap(Interval(3, 4)))

    def test_shift_interval(self):
        """Test valid interval creation without error.
        """
        interval = Interval(1, 3)
        shift = interval.shift(10)
        self.assertEqual(11, shift.start)
        self.assertEqual(13, shift.stop)
        self.assertEqual(2, shift.duration)
        # keep original interval unchanged
        self.assertEqual(1, interval.start)

    def test_invalid_interval(self):
        """Test invalid instantiation with negative time bounds."""
        with self.assertRaises(PulseError):
            Interval(-1, 5)
        with self.assertRaises(PulseError):
            Interval(2, -5)
        with self.assertRaises(PulseError):
            Interval(5, 2)

    def test_interval_equality(self):
        """Test equality and inequality of intervals."""
        interval_1 = Interval(3, 10)
        interval_2 = Interval(3, 10)
        interval_3 = Interval(2, 10)
        self.assertTrue(interval_1 == interval_2)
        self.assertFalse(interval_1 == interval_3)

    def test_representation_of_interval_object(self):
        """Test string representation of intervals."""
        interval = Interval(3, 10)
        self.assertEqual(str(interval), 'Interval(3, 10)')


class TestTimeslot(QiskitTestCase):
    """Timeslot tests."""

    def test_default(self):
        """Test valid time-slot creation without error.
        """
        slot = Timeslot(Interval(1, 3), AcquireChannel(0))
        self.assertEqual(Interval(1, 3), slot.interval)
        self.assertEqual(AcquireChannel(0), slot.channel)

        self.assertEqual(1, slot.start)
        self.assertEqual(3, slot.stop)
        self.assertEqual(2, slot.duration)

    def test_shift(self):
        """Test shifting of Timeslot."""
        slot_1 = Timeslot(Interval(1, 3), AcquireChannel(0))
        slot_2 = Timeslot(Interval(1+5, 3+5), AcquireChannel(0))
        shifted_slot = slot_1.shift(+5)
        self.assertTrue(slot_2 == shifted_slot)

    def test_representation_of_timeslot_object(self):
        """Test representation of Timeslot object."""
        slot = Timeslot(Interval(1, 5), AcquireChannel(0))
        self.assertEqual(repr(slot), 'Timeslot(AcquireChannel(0), (1, 5))')

    def test_check_overlap(self):
        """Test valid interval creation without error.
        """
        dc0 = DriveChannel(0)
        dc1 = DriveChannel(1)

        self.assertEqual(True,
                         Timeslot(Interval(1, 3), dc0).has_overlap(Timeslot(Interval(2, 4), dc0)))

        self.assertEqual(True,
                         Timeslot(Interval(1, 3), dc0).has_overlap(Timeslot(Interval(2, 2), dc0)))

        self.assertEqual(False,
                         Timeslot(Interval(1, 3), dc0).has_overlap(Timeslot(Interval(1, 1), dc0)))
        self.assertEqual(False,
                         Timeslot(Interval(1, 3), dc0).has_overlap(Timeslot(Interval(3, 3), dc0)))
        self.assertEqual(False,
                         Timeslot(Interval(1, 3), dc0).has_overlap(Timeslot(Interval(0, 1), dc0)))
        self.assertEqual(False,
                         Timeslot(Interval(1, 3), dc0).has_overlap(Timeslot(Interval(3, 4), dc0)))

        # check no overlap if different channels
        self.assertEqual(False,
                         Timeslot(Interval(1, 3), dc0).has_overlap(Timeslot(Interval(1, 3), dc1)))


class TestTimeslotCollection(QiskitTestCase):
    """TimeslotCollection tests."""

    def test_can_create_with_valid_timeslots(self):
        """Test valid time-slot collection creation without error.
        """
        slots = [Timeslot(Interval(1, 3), AcquireChannel(0)),
                 Timeslot(Interval(3, 5), AcquireChannel(0))]
        TimeslotCollection(*slots)

    def test_empty_collection(self):
        """Test empty collection creation and its operations.
        """
        empty = TimeslotCollection()
        self.assertEqual(True, empty.is_mergeable_with(empty))

        # can merge with normal collection
        normal = TimeslotCollection(Timeslot(Interval(1, 3), AcquireChannel(0)),
                                    Timeslot(Interval(3, 5), AcquireChannel(0)))
        self.assertEqual(True, empty.is_mergeable_with(normal))
        self.assertEqual(normal, empty.merge(normal))

    def test_can_merge_two_mergeable_collections(self):
        """Test if merge two mergeable time-slot collections.
        """
        # same interval but different channel
        col1 = TimeslotCollection(Timeslot(Interval(1, 3), AcquireChannel(0)))
        col2 = TimeslotCollection(Timeslot(Interval(1, 3), AcquireChannel(1)))
        self.assertEqual(True, col1.is_mergeable_with(col2))
        expected = TimeslotCollection(Timeslot(Interval(1, 3), AcquireChannel(0)),
                                      Timeslot(Interval(1, 3), AcquireChannel(1)))
        self.assertEqual(expected, col1.merge(col2))

        # same channel but different interval
        col1 = TimeslotCollection(Timeslot(Interval(1, 3), AcquireChannel(0)))
        col2 = TimeslotCollection(Timeslot(Interval(3, 5), AcquireChannel(0)))
        self.assertEqual(True, col1.is_mergeable_with(col2))
        expected = TimeslotCollection(Timeslot(Interval(1, 3), AcquireChannel(0)),
                                      Timeslot(Interval(3, 5), AcquireChannel(0)))
        self.assertEqual(expected, col1.merge(col2))

    def test_unmergeable_collections(self):
        """Test if return false for unmergeable collections.
        """
        col1 = TimeslotCollection(Timeslot(Interval(1, 3), AcquireChannel(0)))
        col2 = TimeslotCollection(Timeslot(Interval(2, 4), AcquireChannel(0)))
        self.assertEqual(False, col1.is_mergeable_with(col2))

    def test_shift(self):
        """Test if `shift` shifts the collection for specified time.
        """
        actual = TimeslotCollection(Timeslot(Interval(1, 3), AcquireChannel(0))).shift(10)
        expected = TimeslotCollection(Timeslot(Interval(11, 13), AcquireChannel(0)))
        self.assertEqual(expected, actual)

    def test_is_mergeable_does_not_mutate_TimeslotCollection(self):
        """Test that is_mergeable_with does not mutate the given TimeslotCollection"""

        # Different channel but different interval
        col1 = TimeslotCollection(Timeslot(Interval(1, 2), AcquireChannel(0)))
        expected_channels = col1.channels
        col2 = TimeslotCollection(Timeslot(Interval(3, 4), AcquireChannel(1)))
        col1.is_mergeable_with(col2)
        self.assertEqual(col1.channels, expected_channels)

        # Different channel but same interval
        col1 = TimeslotCollection(Timeslot(Interval(1, 2), AcquireChannel(0)))
        expected_channels = col1.channels
        col2 = TimeslotCollection(Timeslot(Interval(1, 2), AcquireChannel(1)))
        col1.is_mergeable_with(col2)
        self.assertEqual(col1.channels, expected_channels)

    def test_merged_timeslots_do_not_overlap_internally(self):
        """Test to make sure the timeslots do not overlap internally."""
        # same interval but different channel
        int1 = Timeslot(Interval(1, 3), AcquireChannel(0))
        col1 = TimeslotCollection(Timeslot(Interval(5, 7), AcquireChannel(0)))
        col2 = TimeslotCollection(Timeslot(Interval(5, 7), AcquireChannel(1)))

        merged = TimeslotCollection(int1, col1, col2)

        from_interal = TimeslotCollection(*merged.timeslots)

        self.assertEqual(merged, from_interal)

    def test_zero_duration_timeslot(self):
        """Test that TimeslotCollection works properly for zero duration timeslots."""
        # test that inserting zero duration pulses at start and stop of pulse works.
        TimeslotCollection(Timeslot(Interval(0, 10), AcquireChannel(0)),
                           Timeslot(Interval(0, 0), AcquireChannel(0)),
                           Timeslot(Interval(10, 10), AcquireChannel(0)))

        with self.assertRaises(PulseError):
            TimeslotCollection(Timeslot(Interval(0, 10), AcquireChannel(0)),
                               Timeslot(Interval(5, 5), AcquireChannel(0)))

    def test_complement(self):
        """Test complement of TimeslotCollection is properly created."""
        test_collection = TimeslotCollection(
            Timeslot(Interval(5, 7), DriveChannel(0)),
            Timeslot(Interval(10, 12), DriveChannel(0)),
            Timeslot(Interval(0, 3), DriveChannel(1)))
        ref_complement = TimeslotCollection(
            Timeslot(Interval(0, 5), DriveChannel(0)),
            Timeslot(Interval(7, 10), DriveChannel(0)),
            Timeslot(Interval(3, 12), DriveChannel(1)))
        complement_collection = test_collection.complement()

        self.assertEqual(ref_complement, complement_collection)

    def test_complement_set_stop_time(self):
        """Test complement of TimeslotCollection obeys stop_time."""
        stop_time = 20

        test_collection = TimeslotCollection(
            Timeslot(Interval(5, 7), DriveChannel(0)),
            Timeslot(Interval(10, 12), DriveChannel(0)),
            Timeslot(Interval(0, 3), DriveChannel(1)))
        ref_complement = TimeslotCollection(
            Timeslot(Interval(0, 5), DriveChannel(0)),
            Timeslot(Interval(7, 10), DriveChannel(0)),
            Timeslot(Interval(12, stop_time), DriveChannel(0)),
            Timeslot(Interval(3, stop_time), DriveChannel(1)))
        complement_collection = test_collection.complement(stop_time=stop_time)

        self.assertEqual(ref_complement, complement_collection)


if __name__ == '__main__':
    unittest.main()
