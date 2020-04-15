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
from qiskit.pulse.timing import _overlaps, _insertion_index


class TestTimingUtils(QiskitTestCase):
    """Test the pulse utility functions."""

    def test_overlaps(self):
        """Test the `_overlaps` function."""
        # pylint: disable=invalid-name
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
        # pylint: disable=invalid-name
        a = 0
        b = 1
        self.assertFalse(_overlaps((a, a), (a, a)))
        self.assertFalse(_overlaps((a, a), (a, b)))
        self.assertFalse(_overlaps((a, b), (a, a)))
        self.assertFalse(_overlaps((a, b), (b, b)))
        self.assertFalse(_overlaps((b, b), (a, b)))
        self.assertTrue(_overlaps((a, a + 2), (a + 1, a + 1)))
        self.assertTrue(_overlaps((a + 1, a + 1), (a, a + 2)))

    def test_insertion_index(self):
        """Test the `_insertion_index` function."""
        intervals = [(1, 2), (4, 5)]
        self.assertEqual(_insertion_index(intervals, (2, 3)), 1)
        self.assertEqual(_insertion_index(intervals, (3, 4)), 1)
        self.assertEqual(intervals, [(1, 2), (4, 5)])
        intervals = [(1, 2), (4, 5), (6, 7)]
        self.assertEqual(_insertion_index(intervals, (2, 3)), 1)
        self.assertEqual(_insertion_index(intervals, (0, 1)), 0)
        self.assertEqual(_insertion_index(intervals, (5, 6)), 2)
        self.assertEqual(_insertion_index(intervals, (8, 9)), 3)

    def test_insertion_index_when_overlapping(self):
        """Test that `_insertion_index` raises an error when the new_interval _overlaps."""
        intervals = [(10, 20), (44, 55), (60, 61), (80, 1000)]
        with self.assertRaises(PulseError):
            _insertion_index(intervals, (60, 62))
        with self.assertRaises(PulseError):
            _insertion_index(intervals, (100, 1500))

    def test_insertion_index_empty_list(self):
        """Test that the insertion index is properly found for empty lists."""
        self.assertEqual(_insertion_index([], (0, 1)), 0)
