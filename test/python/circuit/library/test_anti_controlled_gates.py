# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at https://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for anti-controlled gates."""

import unittest
from math import sqrt

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import ACHGate, ACXGate, ACYGate, ACZGate
from qiskit.quantum_info import Operator
from test import QiskitTestCase


class TestACHGate(QiskitTestCase):
    """Tests for the anti-controlled Hadamard gate."""

    def test_ach_circuit_method(self):
        """Test applying ACH via the QuantumCircuit method."""
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.ach(qr[0], qr[1])
        self.assertEqual(circuit[0].operation.name, "ach")
        self.assertEqual(circuit[0].qubits, (qr[0], qr[1]))

    def test_ach_wires(self):
        """Test applying ACH using integer wire indices."""
        circuit = QuantumCircuit(2)
        circuit.ach(0, 1)
        self.assertEqual(circuit[0].operation.name, "ach")

    def test_ach_invalid(self):
        """Test that ACH raises errors for invalid arguments."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, circuit.ach, cr[0], cr[1])
        self.assertRaises(CircuitError, circuit.ach, qr[0], qr[0])
        self.assertRaises(CircuitError, circuit.ach, 0.0, qr[0])
        self.assertRaises(CircuitError, circuit.ach, "a", qr[1])

    def test_ach_gate_matrix(self):
        """Test the unitary matrix of the ACH gate."""
        expected = np.array(
            [
                [1 / sqrt(2), 0, 1 / sqrt(2), 0],
                [0, 1, 0, 0],
                [1 / sqrt(2), 0, -1 / sqrt(2), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )
        gate = ACHGate()
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected)

    def test_ach_decomposition_matches_matrix(self):
        """Test that the gate decomposition produces the correct unitary."""
        gate = ACHGate()
        decomposed_op = Operator(gate.definition)
        gate_op = Operator(gate)
        self.assertTrue(gate_op.equiv(decomposed_op))

    def test_ach_equivalence_x_ch_x(self):
        """Test that ACH is equivalent to X-CH-X on the control qubit."""
        # Build the X-CH-X circuit manually
        expected = QuantumCircuit(2)
        expected.x(0)
        expected.ch(0, 1)
        expected.x(0)

        # Build the ACH circuit
        actual = QuantumCircuit(2)
        actual.ach(0, 1)

        self.assertEqual(Operator(expected), Operator(actual))

    def test_ach_inverse(self):
        """Test that ACH is self-inverse."""
        gate = ACHGate()
        inverse = gate.inverse()
        self.assertIsInstance(inverse, ACHGate)

    def test_ach_label(self):
        """Test that label is passed through."""
        circuit = QuantumCircuit(2)
        circuit.ach(0, 1, label="test_label")
        self.assertEqual(circuit[0].operation.label, "test_label")

    def test_ach_is_unitary(self):
        """Test that the ACH gate matrix is unitary."""
        gate = ACHGate()
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(4))

    def test_ach_broadcast(self):
        """Test that ACH broadcasts across qubit registers."""
        qr1 = QuantumRegister(3, "q1")
        qr2 = QuantumRegister(3, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        instruction_set = circuit.ach(qr1, qr2)
        self.assertEqual(len(circuit), 3)


class TestACXGate(QiskitTestCase):
    """Tests for the anti-controlled X (NOT) gate."""

    def test_acx_circuit_method(self):
        """Test applying ACX via the QuantumCircuit method."""
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.acx(qr[0], qr[1])
        self.assertEqual(circuit[0].operation.name, "acx")
        self.assertEqual(circuit[0].qubits, (qr[0], qr[1]))

    def test_acx_wires(self):
        """Test applying ACX using integer wire indices."""
        circuit = QuantumCircuit(2)
        circuit.acx(0, 1)
        self.assertEqual(circuit[0].operation.name, "acx")

    def test_acx_invalid(self):
        """Test that ACX raises errors for invalid arguments."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, circuit.acx, cr[0], cr[1])
        self.assertRaises(CircuitError, circuit.acx, qr[0], qr[0])
        self.assertRaises(CircuitError, circuit.acx, 0.0, qr[0])
        self.assertRaises(CircuitError, circuit.acx, "a", qr[1])

    def test_acx_gate_matrix(self):
        """Test the unitary matrix of the ACX gate."""
        expected = np.array(
            [
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )
        gate = ACXGate()
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected)

    def test_acx_decomposition_matches_matrix(self):
        """Test that the gate decomposition produces the correct unitary."""
        gate = ACXGate()
        decomposed_op = Operator(gate.definition)
        gate_op = Operator(gate)
        self.assertTrue(gate_op.equiv(decomposed_op))

    def test_acx_equivalence_x_cx_x(self):
        """Test that ACX is equivalent to X-CX-X on the control qubit."""
        expected = QuantumCircuit(2)
        expected.x(0)
        expected.cx(0, 1)
        expected.x(0)

        actual = QuantumCircuit(2)
        actual.acx(0, 1)

        self.assertEqual(Operator(expected), Operator(actual))

    def test_acx_inverse(self):
        """Test that ACX is self-inverse."""
        gate = ACXGate()
        inverse = gate.inverse()
        self.assertIsInstance(inverse, ACXGate)

    def test_acx_label(self):
        """Test that label is passed through."""
        circuit = QuantumCircuit(2)
        circuit.acx(0, 1, label="test_label")
        self.assertEqual(circuit[0].operation.label, "test_label")

    def test_acx_is_unitary(self):
        """Test that the ACX gate matrix is unitary."""
        gate = ACXGate()
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(4))

    def test_acx_broadcast(self):
        """Test that ACX broadcasts across qubit registers."""
        qr1 = QuantumRegister(3, "q1")
        qr2 = QuantumRegister(3, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.acx(qr1, qr2)
        self.assertEqual(len(circuit), 3)


class TestACYGate(QiskitTestCase):
    """Tests for the anti-controlled Y gate."""

    def test_acy_circuit_method(self):
        """Test applying ACY via the QuantumCircuit method."""
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.acy(qr[0], qr[1])
        self.assertEqual(circuit[0].operation.name, "acy")
        self.assertEqual(circuit[0].qubits, (qr[0], qr[1]))

    def test_acy_wires(self):
        """Test applying ACY using integer wire indices."""
        circuit = QuantumCircuit(2)
        circuit.acy(0, 1)
        self.assertEqual(circuit[0].operation.name, "acy")

    def test_acy_invalid(self):
        """Test that ACY raises errors for invalid arguments."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, circuit.acy, cr[0], cr[1])
        self.assertRaises(CircuitError, circuit.acy, qr[0], qr[0])
        self.assertRaises(CircuitError, circuit.acy, 0.0, qr[0])
        self.assertRaises(CircuitError, circuit.acy, "a", qr[1])

    def test_acy_gate_matrix(self):
        """Test the unitary matrix of the ACY gate."""
        expected = np.array(
            [
                [0, 0, -1j, 0],
                [0, 1, 0, 0],
                [1j, 0, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )
        gate = ACYGate()
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected)

    def test_acy_decomposition_matches_matrix(self):
        """Test that the gate decomposition produces the correct unitary."""
        gate = ACYGate()
        decomposed_op = Operator(gate.definition)
        gate_op = Operator(gate)
        self.assertTrue(gate_op.equiv(decomposed_op))

    def test_acy_equivalence_x_cy_x(self):
        """Test that ACY is equivalent to X-CY-X on the control qubit."""
        expected = QuantumCircuit(2)
        expected.x(0)
        expected.cy(0, 1)
        expected.x(0)

        actual = QuantumCircuit(2)
        actual.acy(0, 1)

        self.assertEqual(Operator(expected), Operator(actual))

    def test_acy_inverse(self):
        """Test that ACY is self-inverse."""
        gate = ACYGate()
        inverse = gate.inverse()
        self.assertIsInstance(inverse, ACYGate)

    def test_acy_label(self):
        """Test that label is passed through."""
        circuit = QuantumCircuit(2)
        circuit.acy(0, 1, label="test_label")
        self.assertEqual(circuit[0].operation.label, "test_label")

    def test_acy_is_unitary(self):
        """Test that the ACY gate matrix is unitary."""
        gate = ACYGate()
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(4))

    def test_acy_broadcast(self):
        """Test that ACY broadcasts across qubit registers."""
        qr1 = QuantumRegister(3, "q1")
        qr2 = QuantumRegister(3, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.acy(qr1, qr2)
        self.assertEqual(len(circuit), 3)


class TestACZGate(QiskitTestCase):
    """Tests for the anti-controlled Z gate."""

    def test_acz_circuit_method(self):
        """Test applying ACZ via the QuantumCircuit method."""
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.acz(qr[0], qr[1])
        self.assertEqual(circuit[0].operation.name, "acz")
        self.assertEqual(circuit[0].qubits, (qr[0], qr[1]))

    def test_acz_wires(self):
        """Test applying ACZ using integer wire indices."""
        circuit = QuantumCircuit(2)
        circuit.acz(0, 1)
        self.assertEqual(circuit[0].operation.name, "acz")

    def test_acz_invalid(self):
        """Test that ACZ raises errors for invalid arguments."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, circuit.acz, cr[0], cr[1])
        self.assertRaises(CircuitError, circuit.acz, qr[0], qr[0])
        self.assertRaises(CircuitError, circuit.acz, 0.0, qr[0])
        self.assertRaises(CircuitError, circuit.acz, "a", qr[1])

    def test_acz_gate_matrix(self):
        """Test the unitary matrix of the ACZ gate."""
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )
        gate = ACZGate()
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected)

    def test_acz_decomposition_matches_matrix(self):
        """Test that the gate decomposition produces the correct unitary."""
        gate = ACZGate()
        decomposed_op = Operator(gate.definition)
        gate_op = Operator(gate)
        self.assertTrue(gate_op.equiv(decomposed_op))

    def test_acz_equivalence_x_cz_x(self):
        """Test that ACZ is equivalent to X-CZ-X on the control qubit."""
        expected = QuantumCircuit(2)
        expected.x(0)
        expected.cz(0, 1)
        expected.x(0)

        actual = QuantumCircuit(2)
        actual.acz(0, 1)

        self.assertEqual(Operator(expected), Operator(actual))

    def test_acz_inverse(self):
        """Test that ACZ is self-inverse."""
        gate = ACZGate()
        inverse = gate.inverse()
        self.assertIsInstance(inverse, ACZGate)

    def test_acz_label(self):
        """Test that label is passed through."""
        circuit = QuantumCircuit(2)
        circuit.acz(0, 1, label="test_label")
        self.assertEqual(circuit[0].operation.label, "test_label")

    def test_acz_is_unitary(self):
        """Test that the ACZ gate matrix is unitary."""
        gate = ACZGate()
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(4))

    def test_acz_broadcast(self):
        """Test that ACZ broadcasts across qubit registers."""
        qr1 = QuantumRegister(3, "q1")
        qr2 = QuantumRegister(3, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.acz(qr1, qr2)
        self.assertEqual(len(circuit), 3)


if __name__ == "__main__":
    unittest.main()
