# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Tests for Stabilizerstate quantum state class."""

import unittest
import logging
import numpy as np
from numpy.testing import assert_allclose

from qiskit.test import QiskitTestCase
from qiskit import QiskitError
from qiskit import QuantumRegister, QuantumCircuit
from qiskit import transpile

from qiskit.quantum_info.random import random_clifford
from qiskit.quantum_info.states import StabilizerState
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.predicates import matrix_equal

logger = logging.getLogger(__name__)


class TestStabilizerState(QiskitTestCase):
    """Tests for StabilizerState class."""

    def test_init_clifford(self):
        """Test initialization from Clifford."""
        stab1 = StabilizerState(random_clifford(2))
        stab2 = StabilizerState(stab1)
        self.assertEqual(stab1, stab2)

    def test_to_operator(self):
        """Test to_operator method for returning projector."""
        for _ in range(10):
            stab = StabilizerState(random_clifford(2))
            target = Operator(stab)
            op = StabilizerState(stab).to_operator()
            self.assertEqual(op, target)

    def test_conjugate(self):
        """Test conjugate method."""
        for _ in range(10):
            stab = StabilizerState(random_clifford(2))
            target = StabilizerState(stab.conjugate())
            state = StabilizerState(stab).conjugate()
            self.assertEqual(state, target)

    def test_tensor(self):
        """Test tensor method."""
        for _ in range(10):
            cliff1 = random_clifford(2)
            cliff2 = random_clifford(3)
            stab1 = StabilizerState(cliff1)
            stab2 = StabilizerState(cliff2)
            target = StabilizerState(cliff1.tensor(cliff2))
            state = stab1.tensor(stab2)
            self.assertEqual(state, target)

    def test_compose(self):
        """Test tensor method."""
        for _ in range(10):
            cliff1 = random_clifford(2)
            cliff2 = random_clifford(2)
            stab1 = StabilizerState(cliff1)
            stab2 = StabilizerState(cliff2)
            target = StabilizerState(cliff1.compose(cliff2))
            state = stab1.compose(stab2)
            self.assertEqual(state, target)


if __name__ == '__main__':
    unittest.main()
