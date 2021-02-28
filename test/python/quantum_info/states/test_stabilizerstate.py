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
from test import combine
from ddt import ddt

import logging
import numpy as np

from qiskit.test import QiskitTestCase
from qiskit import QiskitError
from qiskit import QuantumRegister, QuantumCircuit
from qiskit import transpile

from qiskit.quantum_info.random import random_clifford, random_pauli
from qiskit.quantum_info.states import StabilizerState
from qiskit.quantum_info.operators.operator import Operator


logger = logging.getLogger(__name__)


@ddt
class TestStabilizerState(QiskitTestCase):
    """Tests for StabilizerState class."""

    rng = np.random.default_rng(1234)
    samples = 10

    @combine(num_qubits=[2, 3, 4, 5])
    def test_init_clifford(self, num_qubits):
        """Test initialization from Clifford."""
        stab1 = StabilizerState(random_clifford(num_qubits, seed=self.rng))
        stab2 = StabilizerState(stab1)
        self.assertEqual(stab1, stab2)

    @combine(num_qubits=[2, 3, 4, 5])
    def test_init_circuit(self, num_qubits):
        """Test initialization from a Clifford circuit."""
        cliff = random_clifford(num_qubits, seed=self.rng)
        stab1 = StabilizerState(cliff.to_circuit())
        stab2 = StabilizerState(cliff)
        self.assertEqual(stab1, stab2)

    @combine(num_qubits=[2, 3, 4, 5])
    def test_init_instruction(self, num_qubits):
        """Test initialization from a Clifford instruction."""
        cliff = random_clifford(num_qubits, seed=self.rng)
        stab1 = StabilizerState(cliff.to_instruction())
        stab2 = StabilizerState(cliff)
        self.assertEqual(stab1, stab2)

    @combine(num_qubits=[2, 3, 4, 5])
    def test_init_pauli(self, num_qubits):
        """Test initialization from pauli."""
        pauli = random_pauli(num_qubits, seed=self.rng)
        stab1 = StabilizerState(pauli)
        stab2 = StabilizerState(stab1)
        self.assertEqual(stab1, stab2)

    @combine(num_qubits=[2, 3, 4, 5])
    def test_to_operator(self, num_qubits):
        """Test to_operator method for returning projector."""
        for _ in range(self.samples):
            stab = StabilizerState(random_clifford(num_qubits, seed=self.rng))
            target = Operator(stab)
            op = StabilizerState(stab).to_operator()
            self.assertEqual(op, target)

    @combine(num_qubits=[2, 3])
    def test_conjugate(self, num_qubits):
        """Test conjugate method."""
        for _ in range(self.samples):
            stab = StabilizerState(random_clifford(num_qubits, seed=self.rng))
            target = StabilizerState(stab.conjugate())
            state = StabilizerState(stab).conjugate()
            self.assertEqual(state, target)

    @combine(num_qubits=[2, 3])
    def test_transpose(self, num_qubits):
        """Test transpose method."""
        for _ in range(self.samples):
            stab = StabilizerState(random_clifford(num_qubits, seed=self.rng))
            target = StabilizerState(stab.transpose())
            state = StabilizerState(stab).transpose()
            self.assertEqual(state, target)

    def test_tensor(self):
        """Test tensor method."""
        for _ in range(self.samples):
            cliff1 = random_clifford(2, seed=self.rng)
            cliff2 = random_clifford(3, seed=self.rng)
            stab1 = StabilizerState(cliff1)
            stab2 = StabilizerState(cliff2)
            target = StabilizerState(cliff1.tensor(cliff2))
            state = stab1.tensor(stab2)
            self.assertEqual(state, target)

    def test_expand(self):
        """Test expand method."""
        for _ in range(self.samples):
            cliff1 = random_clifford(2, seed=self.rng)
            cliff2 = random_clifford(3, seed=self.rng)
            stab1 = StabilizerState(cliff1)
            stab2 = StabilizerState(cliff2)
            target = StabilizerState(cliff1.expand(cliff2))
            state = stab1.expand(stab2)
            self.assertEqual(state, target)

    @combine(num_qubits=[2, 3])
    def test_compose(self, num_qubits):
        """Test tensor method."""
        for _ in range(self.samples):
            cliff1 = random_clifford(num_qubits, seed=self.rng)
            cliff2 = random_clifford(num_qubits, seed=self.rng)
            stab1 = StabilizerState(cliff1)
            stab2 = StabilizerState(cliff2)
            target = StabilizerState(cliff1.compose(cliff2))
            state = stab1.compose(stab2)
            self.assertEqual(state, target)


if __name__ == '__main__':
    unittest.main()
