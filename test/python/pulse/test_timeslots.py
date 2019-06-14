# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test cases for the timeslots."""
import unittest

from qiskit.pulse.channels import AcquireChannel
from qiskit.pulse.timeslots import Interval, Timeslot, TimeslotCollection
from qiskit.test import QiskitTestCase


class TestInterval(QiskitTestCase):
    """Interval tests."""

    def test_default(self):
        """Test valid interval creation without error.
        """
        interval = Interval(1, 3)
        self.assertEqual(1, interval.begin)
        self.assertEqual(3, interval.end)
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
        self.assertEqual(11, shift.begin)
        self.assertEqual(13, shift.end)
        self.assertEqual(2, shift.duration)
        # keep original interval unchanged
        self.assertEqual(1, interval.begin)


class TestTimeslot(QiskitTestCase):
    """Timeslot tests."""

    def test_default(self):
        """Test valid time-slot creation without error.
        """
        slot = Timeslot(Interval(1, 3), AcquireChannel(0))
        self.assertEqual(Interval(1, 3), slot.interval)
        self.assertEqual(AcquireChannel(0), slot.channel)


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
        self.assertEqual(normal, empty.merged(normal))

    def test_can_merge_two_mergeable_collections(self):
        """Test if merge two mergeable time-slot collections.
        """
        # same interval but different channel
        col1 = TimeslotCollection(Timeslot(Interval(1, 3), AcquireChannel(0)))
        col2 = TimeslotCollection(Timeslot(Interval(1, 3), AcquireChannel(1)))
        self.assertEqual(True, col1.is_mergeable_with(col2))
        expected = TimeslotCollection(Timeslot(Interval(1, 3), AcquireChannel(0)),
                                      Timeslot(Interval(1, 3), AcquireChannel(1)))
        self.assertEqual(expected, col1.merged(col2))

        # same channel but different interval
        col1 = TimeslotCollection(Timeslot(Interval(1, 3), AcquireChannel(0)))
        col2 = TimeslotCollection(Timeslot(Interval(3, 5), AcquireChannel(0)))
        self.assertEqual(True, col1.is_mergeable_with(col2))
        expected = TimeslotCollection(Timeslot(Interval(1, 3), AcquireChannel(0)),
                                      Timeslot(Interval(3, 5), AcquireChannel(0)))
        self.assertEqual(expected, col1.merged(col2))

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


if __name__ == '__main__':
    unittest.main()
