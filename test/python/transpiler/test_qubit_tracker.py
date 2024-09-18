# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test QubitTracker functionality."""

import unittest

from qiskit.exceptions import QiskitError
from qiskit._accelerate.qubit_tracker import QubitTracker

from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestQubitTracker(QiskitTestCase):
    """Test QubitTracker functionality."""

    def test_basic_usage(self):
        """Test basic usage: defining the tracker, calling `used` and `reset`,
        querying the number of clean and dirty ancilla qubits.
        """
        qubits = range(11)
        clean = {0, 2, 4, 6, 8, 10}
        dirty = {1, 3, 5, 7, 9}
        tracker = QubitTracker(qubits, clean, dirty)
        self.assertEqual(tracker.num_clean(), 6)
        self.assertEqual(tracker.num_dirty(), 5)
        self.assertEqual(tracker.num_clean(active_qubits=[2, 3, 4, 5]), 4)
        self.assertEqual(tracker.num_dirty(active_qubits=[2, 3, 4, 5]), 3)
        tracker.used([4, 3, 0])  # clean becomes {2, 6, 8, 10}, dirty becomes {0, 1, 3, 4, 5, 7, 9}
        self.assertEqual(tracker.num_clean(), 4)
        self.assertEqual(tracker.num_dirty(), 7)
        tracker.reset([2, 5, 9])  # clean becomes {2, 5, 6, 8, 9, 10}, dirty becomes {0, 1, 3, 4, 7}
        self.assertEqual(tracker.num_clean(), 6)
        self.assertEqual(tracker.num_dirty(), 5)
        tracker.drop([10, 0, 1, 5])  # clean becomes {2, 6, 8, 9}, dirty becomes {3, 4, 7}
        self.assertEqual(tracker.num_clean(), 4)
        self.assertEqual(tracker.num_dirty(), 3)
        self.assertEqual(set(tracker.borrow(3)), {2, 6, 8})
        self.assertEqual(set(tracker.borrow(5)), {2, 6, 8, 9, 3})
        self.assertEqual(set(tracker.borrow(2, active_qubits=[8, 3, 2])), {6, 9})
        self.assertEqual(set(tracker.borrow(3, active_qubits=[8, 3, 2])), {6, 9, 4})

    def test_borrow_raises(self):
        """Test that `borrow` raises an exception when there are not enough available qubits."""
        tracker = QubitTracker(range(5), {0, 1, 2}, {3, 4})
        with self.assertRaises(QiskitError):
            tracker.borrow(3, [0, 1, 2])

    def test_copy(self):
        """Test `copy` method."""
        tracker = QubitTracker(range(11), {0, 2, 4, 6, 8, 10}, {1, 3, 5, 7, 9})

        with self.subTest("copy without qubit_map and without drop"):
            tracker1 = tracker.copy()
            self.assertEqual(tracker1.num_clean(), 6)
            self.assertEqual(tracker1.num_dirty(), 5)

        with self.subTest("copy with qubit_map and without drop"):
            tracker2 = tracker.copy(
                qubit_map={2: 11, 4: 13, 5: 5, 7: 2}
            )  # clean: {11, 13}, dirty: {2, 5}
            self.assertEqual(tracker2.num_clean(), 2)
            self.assertEqual(tracker2.num_dirty(), 2)

        with self.subTest("copy without qubit_map and with drop"):
            tracker3 = tracker.copy(drop=[10, 2, 6, 7, 5, 3])  # clean: {0, 4, 8}, dirty: {1, 9}
            self.assertEqual(tracker3.num_clean(), 3)
            self.assertEqual(tracker3.num_dirty(), 2)


if __name__ == "__main__":
    unittest.main()
