# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests utility functions for operator classes."""

import unittest

from qiskit.quantum_info import SparsePauliOp, anti_commutator, commutator, double_commutator
from qiskit.test import QiskitTestCase


class TestOperatorUtils(QiskitTestCase):
    """Test utility functions for operator classes."""

    def test_commutator(self):
        """Test commutator function on SparsePauliOp."""
        Z = SparsePauliOp("Z")
        I = SparsePauliOp("I")
        zero = SparsePauliOp("I", 0)
        self.assertTrue(commutator(Z, I).equiv(zero))

    def test_anti_commutator(self):
        """Test anti_commutator function on SparsePauliOp."""
        Z = SparsePauliOp("Z")
        X = SparsePauliOp("X")
        zero = SparsePauliOp("I", 0)
        self.assertTrue(anti_commutator(Z, X).equiv(zero))

    def test_double_commutator(self):
        """Test double_commutator function on SparsePauliOp."""
        X = SparsePauliOp("X")
        Y = SparsePauliOp("Y")
        Z = SparsePauliOp("Z")
        zero = SparsePauliOp("I", 0)
        self.assertTrue(double_commutator(X, Y, Z, sign=False).equiv(zero))
        self.assertTrue(double_commutator(X, Y, X, sign=True).equiv(zero))


if __name__ == "__main__":
    unittest.main()
