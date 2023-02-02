# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test ElideSwap pass"""

import unittest

from qiskit import QuantumCircuit
from qiskit.transpiler.passes import ElideSwaps
from qiskit.test import QiskitTestCase


class TestElideSwaps(QiskitTestCase):
    """Test swap elision logical optimization pass."""

    def setUp(self):
        super().setUp()
        self.swap_pass = ElideSwaps()

    def test_no_swap(self):
        """Test no swap means no transform."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        res = self.swap_pass(qc)
        self.assertEqual(res, qc)

    def test_swap_in_middle(self):
        """Test swap in middle of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.swap(0, 1)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(0, 2)
        expected.barrier(0, 1, 2)
        expected.measure(1, 0)
        expected.measure(0, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_swap_at_beginning(self):
        """Test swap in beginning of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.swap(0, 1)
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        expected = QuantumCircuit(3, 3)
        expected.h(1)
        expected.cx(0, 2)
        expected.barrier(0, 1, 2)
        expected.measure(1, 0)
        expected.measure(0, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_swap_at_end(self):
        """Test swap in beginning of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)
        qc.swap(0, 1)

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(1, 2)
        expected.barrier(0, 1, 2)
        expected.measure(0, 0)
        expected.measure(1, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)

    def test_swap_before_measure(self):
        """Test swap in beginning of bell is elided."""
        qc = QuantumCircuit(3, 3)
        qc.h(0)
        qc.cx(1, 2)
        qc.barrier(0, 1, 2)
        qc.swap(0, 1)
        qc.measure(0, 0)
        qc.measure(1, 1)
        qc.measure(2, 2)

        expected = QuantumCircuit(3, 3)
        expected.h(0)
        expected.cx(1, 2)
        expected.barrier(0, 1, 2)
        expected.measure(1, 0)
        expected.measure(0, 1)
        expected.measure(2, 2)

        res = self.swap_pass(qc)
        self.assertEqual(res, expected)


if __name__ == "__main__":
    unittest.main()
