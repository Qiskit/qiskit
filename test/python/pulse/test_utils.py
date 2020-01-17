# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases for the pulse utilities."""

from qiskit.test import QiskitTestCase

from qiskit.pulse.exceptions import PulseError
from qiskit.pulse.utils import Interval, overlaps, insertion_index


class TestPulseUtils(QiskitTestCase):
    """Test the pulse utility functions."""

    def test_overlaps(self):
        """Test the `overlaps` function."""
        # pylint: disable=invalid-name
        a = Interval(0, 1)
        b = Interval(1, 4)
        c = Interval(2, 3)
        d = Interval(3, 5)
        self.assertFalse(overlaps(a, b))
        self.assertFalse(overlaps(b, a))
        self.assertFalse(overlaps(a, d))
        self.assertTrue(overlaps(b, c))
        self.assertTrue(overlaps(c, b))
        self.assertTrue(overlaps(b, d))
        self.assertTrue(overlaps(d, b))

    def test_insertion_index(self):
        """Test the `insertion_index` function."""
        intervals = [Interval(1, 2), Interval(4, 5)]
        self.assertEqual(insertion_index(intervals, Interval(2, 3)), 1)
        self.assertEqual(insertion_index(intervals, Interval(3, 4)), 1)
        self.assertEqual(intervals, [Interval(1, 2), Interval(4, 5)])
        intervals = [Interval(1, 2), Interval(4, 5), Interval(6, 7)]
        self.assertEqual(insertion_index(intervals, Interval(2, 3)), 1)
        self.assertEqual(insertion_index(intervals, Interval(0, 1)), 0)
        self.assertEqual(insertion_index(intervals, Interval(5, 6)), 2)
        self.assertEqual(insertion_index(intervals, Interval(8, 9)), 3)

    def test_insertion_index_when_overlapping(self):
        """Test that `insertion_index` raises an error when the new_interval overlaps."""
        intervals = [Interval(10, 20), Interval(44, 55), Interval(60, 61), Interval(80, 1000)]
        with self.assertRaises(PulseError):
            insertion_index(intervals, Interval(60, 62))
        with self.assertRaises(PulseError):
            insertion_index(intervals, Interval(100, 1500))
