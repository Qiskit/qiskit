# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""Test Qiskit's QuantumCircuit class."""

from ddt import ddt, data
from qiskit import QuantumCircuit
from qiskit.test import QiskitTestCase
import numpy as np
from math import sqrt


@ddt
class TestCircuitToMatrix(QiskitTestCase):
    """QuantumCircuit to_matrix tests."""
    
    def setUp(self):
        self._2 = 1 / sqrt(2)  # Normalization coefficient for a matrix with determinant 2.

    @data(1, 2, 3)
    def test_empty_circuits_to_matrix(self, n_qubits):
        """Teset an empty quantum circuit's conversion to an identity matrix.
        """
        qc = QuantumCircuit(n_qubits)
        self.assertNumpyArrayEqual(qc.to_matrix(), np.eye(2 ** n_qubits))

    # --- Tests for the X gate. ---

    def test_circuit_size_1_gate_x_to_matrix(self):
        """Test a 1-qubit QuantumCircuit.to_matrix after applying the gate X on qubit 0.
        """
        qc = QuantumCircuit(1)
        qc.x(0)
        self.assertNumpyArrayEqual(qc.to_matrix(), np.array([[0, 1],
                                                             [1, 0]]))

    def test_circuit_size_2_gate_x0_to_matrix(self):
        """xTest a 2-qubit QuantumCircuit.to_matrix after applying the gate X on qubit 0.
        """
        qc = QuantumCircuit(2)
        qc.x(0)
        self.assertNumpyArrayEqual(qc.to_matrix(), np.array([[0, 1, 0, 0],
                                                             [1, 0, 0, 0],
                                                             [0, 0, 0, 1],
                                                             [0, 0, 1, 0]]))

    def test_circuit_size_2_gate_x1_to_matrix(self):
        """xTest a 2-qubit QuantumCircuit.to_matrix after applying the gate X on qubit 1.
        """
        qc = QuantumCircuit(2)
        qc.x(1)
        self.assertNumpyArrayEqual(qc.to_matrix(), np.array([[0, 0, 1, 0],
                                                             [0, 0, 0, 1],
                                                             [1, 0, 0, 0],
                                                             [0, 1, 0, 0]]))

    def test_circuit_size_3_gate_x0_to_matrix(self):
        """xTest a 3-qubit QuantumCircuit.to_matrix after applying the gate X on qubit 0.
        """
        qc = QuantumCircuit(3)
        qc.x(0)
        self.assertNumpyArrayEqual(qc.to_matrix(), np.array([[0, 1, 0, 0, 0, 0, 0, 0],
                                                             [1, 0, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 1, 0, 0, 0, 0],
                                                             [0, 0, 1, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 1, 0, 0],
                                                             [0, 0, 0, 0, 1, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 1],
                                                             [0, 0, 0, 0, 0, 0, 1, 0]]))

    def test_circuit_size_3_gate_x1_to_matrix(self):
        """xTest a 3-qubit QuantumCircuit.to_matrix after applying the gate X on qubit 1.
        """
        qc = QuantumCircuit(3)
        qc.x(1)
        self.assertNumpyArrayEqual(qc.to_matrix(), np.array([[0, 0, 1, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 1, 0, 0, 0, 0],
                                                             [1, 0, 0, 0, 0, 0, 0, 0],
                                                             [0, 1, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, 1, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 1],
                                                             [0, 0, 0, 0, 1, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 1, 0, 0]]))

    def test_circuit_size_3_gate_x2_to_matrix(self):
        """xTest a 3-qubit QuantumCircuit.to_matrix after applying the gate X on qubit 2.
        """
        qc = QuantumCircuit(3)
        qc.x(2)
        self.assertNumpyArrayEqual(qc.to_matrix(), np.array([[0, 0, 0, 0, 1, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 1, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, 1, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 1],
                                                             [1, 0, 0, 0, 0, 0, 0, 0],
                                                             [0, 1, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, 1, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 1, 0, 0, 0, 0]]))

    # --- Tests for the CX gate. ---

    def test_circuit_size_2_gate_cx0_to_matrix(self):
        """xTest a 2-qubit QuantumCircuit.to_matrix after applying the gate X on qubit 0.
        """
        qc = QuantumCircuit(2)
        qc.cx(1, 0)  # control, target
        self.assertNumpyArrayEqual(qc.to_matrix(), np.array([[1, 0, 0, 0],
                                                             [0, 1, 0, 0],
                                                             [0, 0, 0, 1],
                                                             [0, 0, 1, 0]]))

    def test_circuit_size_2_gate_cx1_to_matrix(self):
        """xTest a 2-qubit QuantumCircuit.to_matrix after applying the gate X on qubit 1.
        """
        qc = QuantumCircuit(2)
        qc.cx(0, 1)  # control, target
        self.assertNumpyArrayEqual(qc.to_matrix(), np.array([[1, 0, 0, 0],
                                                             [0, 0, 0, 1],
                                                             [0, 0, 1, 0],
                                                             [0, 1, 0, 0]]))

    # --- Tests for the H gate. ---

    def test_circuit_size_1_gate_h_to_matrix(self):
        """xTest a 1-qubit QuantumCircuit.to_matrix after applying the gate H on qubit 0.
        """
        qc = QuantumCircuit(1)
        qc.h(0)
        self.assertNumpyArrayEqual(qc.to_matrix(), np.array([[self._2, self._2],
                                                             [self._2, -self._2]]))

    def test_circuit_size_2_gate_h0_to_matrix(self):
        """xTest a 2-qubit QuantumCircuit.to_matrix after applying the gate H on qubit 0.
        """
        qc = QuantumCircuit(2)
        qc.h(0)
        self.assertNumpyArrayEqual(qc.to_matrix(), np.array([[self._2, self._2, 0, 0],
                                                             [self._2, -self._2, 0, 0],
                                                             [0, 0, self._2, self._2],
                                                             [0, 0, self._2, -self._2]]))

    def test_circuit_size_2_gate_h1_to_matrix(self):
        """xTest a 2-qubit QuantumCircuit.to_matrix after applying the gate H on qubit 1.
        """
        qc = QuantumCircuit(2)
        qc.h(1)
        self.assertNumpyArrayEqual(qc.to_matrix(), np.array([[self._2, 0, self._2, 0],
                                                             [0, self._2, 0, self._2],
                                                             [self._2, 0, -self._2, 0],
                                                             [0, self._2, 0, -self._2]]))

    def test_circuit_size_3_gate_h0_to_matrix(self):
        """xTest a 3-qubit QuantumCircuit.to_matrix after applying the gate H on qubit 0.
        """
        qc = QuantumCircuit(3)
        qc.h(0)
        self.assertNumpyArrayEqual(qc.to_matrix(), np.array([[self._2, self._2, 0, 0, 0, 0, 0, 0],
                                                             [self._2, -self._2, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, self._2, self._2, 0, 0, 0, 0],
                                                             [0, 0, self._2, -self._2, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, self._2, self._2, 0, 0],
                                                             [0, 0, 0, 0, self._2, -self._2, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, self._2, self._2],
                                                             [0, 0, 0, 0, 0, 0, self._2, -self._2]]))

    def test_circuit_size_3_gate_h1_to_matrix(self):
        """xTest a 3-qubit QuantumCircuit.to_matrix after applying the gate H on qubit 1.
        """
        qc = QuantumCircuit(3)
        qc.h(1)
        self.assertNumpyArrayEqual(qc.to_matrix(), np.array([[self._2, 0, self._2, 0, 0, 0, 0, 0],
                                                             [0, self._2, 0, self._2, 0, 0, 0, 0],
                                                             [self._2, 0, -self._2, 0, 0, 0, 0, 0],
                                                             [0, self._2, 0, -self._2, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, self._2, 0, self._2, 0],
                                                             [0, 0, 0, 0, 0, self._2, 0, self._2],
                                                             [0, 0, 0, 0, self._2, 0, -self._2, 0],
                                                             [0, 0, 0, 0, 0, self._2, 0, -self._2]]))

    def test_circuit_size_3_gate_h2_to_matrix(self):
        """xTest a 3-qubit QuantumCircuit.to_matrix after applying the gate H on qubit 2.
        """
        qc = QuantumCircuit(3)
        qc.h(2)
        self.assertNumpyArrayEqual(qc.to_matrix(), np.array([[self._2, 0, 0, 0, self._2, 0, 0, 0],
                                                             [0, self._2, 0, 0, 0, self._2, 0, 0],
                                                             [0, 0, self._2, 0, 0, 0, self._2, 0],
                                                             [0, 0, 0, self._2, 0, 0, 0, self._2],
                                                             [self._2, 0, 0, 0, -self._2, 0, 0, 0],
                                                             [0, self._2, 0, 0, 0, -self._2, 0, 0],
                                                             [0, 0, self._2, 0, 0, 0, -self._2, 0],
                                                             [0, 0, 0, self._2, 0, 0, 0, -self._2]]))

    def assertNumpyArrayEqual(self, x, y):
        np.testing.assert_array_equal(x, y)
