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
import logging
from ddt import ddt

import numpy as np

from qiskit.test import QiskitTestCase
# from qiskit import QiskitError
from qiskit import QuantumRegister, QuantumCircuit
# from qiskit import transpile

from qiskit.quantum_info.random import random_clifford, random_pauli
from qiskit.quantum_info.states import StabilizerState
from qiskit.circuit.library import (IGate, XGate, YGate, ZGate, HGate,
                                    SGate, SdgGate, CXGate, CZGate,
                                    SwapGate)
from qiskit.quantum_info.operators import Clifford, Operator


logger = logging.getLogger(__name__)


@ddt
class TestStabilizerState(QiskitTestCase):
    """Tests for StabilizerState class."""

    rng = np.random.default_rng(12345)
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

    @combine(num_qubits=[2, 3, 4])
    def test_evolve(self, num_qubits):
        """Test evolve method."""
        for _ in range(self.samples):
            cliff1 = random_clifford(num_qubits, seed=self.rng)
            cliff2 = random_clifford(num_qubits, seed=self.rng)
            stab1 = StabilizerState(cliff1)
            stab2 = StabilizerState(cliff2)
            target = StabilizerState(cliff1.compose(cliff2))
            state = stab1.evolve(stab2)
            self.assertEqual(state, target)

    @combine(num_qubits_1=[4, 5, 6], num_qubits_2=[1, 2, 3])
    def test_evolve_subsystem(self, num_qubits_1, num_qubits_2):
        """Test subsystem evolve method."""
        for _ in range(self.samples):
            cliff1 = random_clifford(num_qubits_1, seed=self.rng)
            cliff2 = random_clifford(num_qubits_2, seed=self.rng)
            stab1 = StabilizerState(cliff1)
            stab2 = StabilizerState(cliff2)
            qargs = sorted(np.random.choice(range(num_qubits_1), num_qubits_2, replace=False))
            target = StabilizerState(cliff1.compose(cliff2, qargs))
            state = stab1.evolve(stab2, qargs)
            self.assertEqual(state, target)

    def test_measure_single_qubit(self):
        """Test a measurement of a single qubit"""
        for _ in range(self.samples):
            cliff = Clifford(XGate())
            stab = StabilizerState(cliff)
            value = stab._measure_and_update(0)
            self.assertEqual(value, 1)

            cliff = Clifford(IGate())
            stab = StabilizerState(cliff)
            value = stab._measure_and_update(0)
            self.assertEqual(value, 0)

            num_qubits = 4
            qc = QuantumCircuit(num_qubits)
            stab = StabilizerState(qc)
            for i in range(num_qubits):
                value = stab._measure_and_update(i)
                self.assertEqual(value, 0)

            for i in range(num_qubits):
                qc.x(i)
            stab = StabilizerState(qc)
            for i in range(num_qubits):
                value = stab._measure_and_update(i)
                self.assertEqual(value, 1)

            qc = QuantumCircuit(num_qubits)
            qc.h(0)
            qc.cx(0, 1)
            qc.cx(0, 2)
            qc.cx(0, 3)
            stab = StabilizerState(qc)
            value = []
            for i in range(num_qubits):
                value.append(stab._measure_and_update(i))
                self.assertEqual(value[i], value[0])


if __name__ == '__main__':
    unittest.main()
