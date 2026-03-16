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
import math
from math import sqrt

import numpy as np

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.library import (
    ACHGate, ACXGate, ACYGate, ACZGate, ACRXGate, ACRYGate, ACRZGate, ACPhaseGate, ACUGate,
    ACSXGate, ACSXdgGate, ACSGate, ACSdgGate, ACU1Gate, ACU3Gate, AMCXGate, AMCPhaseGate,
    ACCXGate,
)
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


class TestACRXGate(QiskitTestCase):
    """Tests for the anti-controlled RX gate."""

    def test_acrx_circuit_method(self):
        """Test applying ACRX via the QuantumCircuit method."""
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.acrx(0.5, qr[0], qr[1])
        self.assertEqual(circuit[0].operation.name, "acrx")
        self.assertEqual(circuit[0].qubits, (qr[0], qr[1]))

    def test_acrx_wires(self):
        """Test applying ACRX using integer wire indices."""
        circuit = QuantumCircuit(2)
        circuit.acrx(0.5, 0, 1)
        self.assertEqual(circuit[0].operation.name, "acrx")

    def test_acrx_invalid(self):
        """Test that ACRX raises errors for invalid arguments."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, circuit.acrx, 0.5, cr[0], cr[1])
        self.assertRaises(CircuitError, circuit.acrx, 0.5, qr[0], qr[0])

    def test_acrx_gate_matrix(self):
        """Test the unitary matrix of the ACRX gate for theta=pi/2."""
        theta = math.pi / 2
        half = theta / 2
        cos = math.cos(half)
        isin = 1j * math.sin(half)
        expected = np.array(
            [[cos, 0, -isin, 0], [0, 1, 0, 0], [-isin, 0, cos, 0], [0, 0, 0, 1]],
            dtype=np.complex128,
        )
        gate = ACRXGate(theta)
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected)

    def test_acrx_decomposition_matches_matrix(self):
        """Test that the gate decomposition produces the correct unitary."""
        gate = ACRXGate(1.23)
        decomposed_op = Operator(gate.definition)
        gate_op = Operator(gate)
        self.assertTrue(gate_op.equiv(decomposed_op))

    def test_acrx_equivalence_x_crx_x(self):
        """Test that ACRX is equivalent to X-CRX-X on the control qubit."""
        theta = 0.7
        expected = QuantumCircuit(2)
        expected.x(0)
        expected.crx(theta, 0, 1)
        expected.x(0)

        actual = QuantumCircuit(2)
        actual.acrx(theta, 0, 1)

        self.assertEqual(Operator(expected), Operator(actual))

    def test_acrx_inverse(self):
        """Test that inverse negates the angle."""
        gate = ACRXGate(1.5)
        inverse = gate.inverse()
        self.assertIsInstance(inverse, ACRXGate)
        self.assertAlmostEqual(float(inverse.params[0]), -1.5)

    def test_acrx_inverse_product_is_identity(self):
        """Test that gate * inverse = identity."""
        theta = 0.8
        qc = QuantumCircuit(2)
        qc.acrx(theta, 0, 1)
        qc.acrx(-theta, 0, 1)
        op = Operator(qc)
        np.testing.assert_array_almost_equal(op.data, np.eye(4))

    def test_acrx_label(self):
        """Test that label is passed through."""
        circuit = QuantumCircuit(2)
        circuit.acrx(0.5, 0, 1, label="test_label")
        self.assertEqual(circuit[0].operation.label, "test_label")

    def test_acrx_is_unitary(self):
        """Test that the ACRX gate matrix is unitary."""
        gate = ACRXGate(2.1)
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(4))

    def test_acrx_zero_angle(self):
        """Test that ACRX(0) is identity."""
        gate = ACRXGate(0)
        np.testing.assert_array_almost_equal(gate.to_matrix(), np.eye(4))


class TestACRYGate(QiskitTestCase):
    """Tests for the anti-controlled RY gate."""

    def test_acry_circuit_method(self):
        """Test applying ACRY via the QuantumCircuit method."""
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.acry(0.5, qr[0], qr[1])
        self.assertEqual(circuit[0].operation.name, "acry")
        self.assertEqual(circuit[0].qubits, (qr[0], qr[1]))

    def test_acry_wires(self):
        """Test applying ACRY using integer wire indices."""
        circuit = QuantumCircuit(2)
        circuit.acry(0.5, 0, 1)
        self.assertEqual(circuit[0].operation.name, "acry")

    def test_acry_invalid(self):
        """Test that ACRY raises errors for invalid arguments."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, circuit.acry, 0.5, cr[0], cr[1])
        self.assertRaises(CircuitError, circuit.acry, 0.5, qr[0], qr[0])

    def test_acry_gate_matrix(self):
        """Test the unitary matrix of the ACRY gate for theta=pi/2."""
        theta = math.pi / 2
        half = theta / 2
        cos = math.cos(half)
        sin = math.sin(half)
        expected = np.array(
            [[cos, 0, -sin, 0], [0, 1, 0, 0], [sin, 0, cos, 0], [0, 0, 0, 1]],
            dtype=np.complex128,
        )
        gate = ACRYGate(theta)
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected)

    def test_acry_decomposition_matches_matrix(self):
        """Test that the gate decomposition produces the correct unitary."""
        gate = ACRYGate(1.23)
        decomposed_op = Operator(gate.definition)
        gate_op = Operator(gate)
        self.assertTrue(gate_op.equiv(decomposed_op))

    def test_acry_equivalence_x_cry_x(self):
        """Test that ACRY is equivalent to X-CRY-X on the control qubit."""
        theta = 0.7
        expected = QuantumCircuit(2)
        expected.x(0)
        expected.cry(theta, 0, 1)
        expected.x(0)

        actual = QuantumCircuit(2)
        actual.acry(theta, 0, 1)

        self.assertEqual(Operator(expected), Operator(actual))

    def test_acry_inverse(self):
        """Test that inverse negates the angle."""
        gate = ACRYGate(1.5)
        inverse = gate.inverse()
        self.assertIsInstance(inverse, ACRYGate)
        self.assertAlmostEqual(float(inverse.params[0]), -1.5)

    def test_acry_inverse_product_is_identity(self):
        """Test that gate * inverse = identity."""
        theta = 0.8
        qc = QuantumCircuit(2)
        qc.acry(theta, 0, 1)
        qc.acry(-theta, 0, 1)
        op = Operator(qc)
        np.testing.assert_array_almost_equal(op.data, np.eye(4))

    def test_acry_label(self):
        """Test that label is passed through."""
        circuit = QuantumCircuit(2)
        circuit.acry(0.5, 0, 1, label="test_label")
        self.assertEqual(circuit[0].operation.label, "test_label")

    def test_acry_is_unitary(self):
        """Test that the ACRY gate matrix is unitary."""
        gate = ACRYGate(2.1)
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(4))

    def test_acry_zero_angle(self):
        """Test that ACRY(0) is identity."""
        gate = ACRYGate(0)
        np.testing.assert_array_almost_equal(gate.to_matrix(), np.eye(4))


class TestACRZGate(QiskitTestCase):
    """Tests for the anti-controlled RZ gate."""

    def test_acrz_circuit_method(self):
        """Test applying ACRZ via the QuantumCircuit method."""
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.acrz(0.5, qr[0], qr[1])
        self.assertEqual(circuit[0].operation.name, "acrz")
        self.assertEqual(circuit[0].qubits, (qr[0], qr[1]))

    def test_acrz_wires(self):
        """Test applying ACRZ using integer wire indices."""
        circuit = QuantumCircuit(2)
        circuit.acrz(0.5, 0, 1)
        self.assertEqual(circuit[0].operation.name, "acrz")

    def test_acrz_invalid(self):
        """Test that ACRZ raises errors for invalid arguments."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, circuit.acrz, 0.5, cr[0], cr[1])
        self.assertRaises(CircuitError, circuit.acrz, 0.5, qr[0], qr[0])

    def test_acrz_gate_matrix(self):
        """Test the unitary matrix of the ACRZ gate for theta=pi/2."""
        import cmath

        theta = math.pi / 2
        arg = 1j * theta / 2
        expected = np.array(
            [
                [cmath.exp(-arg), 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, cmath.exp(arg), 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )
        gate = ACRZGate(theta)
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected)

    def test_acrz_decomposition_matches_matrix(self):
        """Test that the gate decomposition produces the correct unitary."""
        gate = ACRZGate(1.23)
        decomposed_op = Operator(gate.definition)
        gate_op = Operator(gate)
        self.assertTrue(gate_op.equiv(decomposed_op))

    def test_acrz_equivalence_x_crz_x(self):
        """Test that ACRZ is equivalent to X-CRZ-X on the control qubit."""
        theta = 0.7
        expected = QuantumCircuit(2)
        expected.x(0)
        expected.crz(theta, 0, 1)
        expected.x(0)

        actual = QuantumCircuit(2)
        actual.acrz(theta, 0, 1)

        self.assertEqual(Operator(expected), Operator(actual))

    def test_acrz_inverse(self):
        """Test that inverse negates the angle."""
        gate = ACRZGate(1.5)
        inverse = gate.inverse()
        self.assertIsInstance(inverse, ACRZGate)
        self.assertAlmostEqual(float(inverse.params[0]), -1.5)

    def test_acrz_inverse_product_is_identity(self):
        """Test that gate * inverse = identity."""
        theta = 0.8
        qc = QuantumCircuit(2)
        qc.acrz(theta, 0, 1)
        qc.acrz(-theta, 0, 1)
        op = Operator(qc)
        np.testing.assert_array_almost_equal(op.data, np.eye(4))

    def test_acrz_label(self):
        """Test that label is passed through."""
        circuit = QuantumCircuit(2)
        circuit.acrz(0.5, 0, 1, label="test_label")
        self.assertEqual(circuit[0].operation.label, "test_label")

    def test_acrz_is_unitary(self):
        """Test that the ACRZ gate matrix is unitary."""
        gate = ACRZGate(2.1)
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(4))

    def test_acrz_zero_angle(self):
        """Test that ACRZ(0) is identity."""
        gate = ACRZGate(0)
        np.testing.assert_array_almost_equal(gate.to_matrix(), np.eye(4))


class TestACPhaseGate(QiskitTestCase):
    """Tests for the anti-controlled Phase gate."""

    def test_acp_circuit_method(self):
        """Test applying ACPhase via the QuantumCircuit method."""
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.acp(0.5, qr[0], qr[1])
        self.assertEqual(circuit[0].operation.name, "acp")
        self.assertEqual(circuit[0].qubits, (qr[0], qr[1]))

    def test_acp_wires(self):
        """Test applying ACPhase using integer wire indices."""
        circuit = QuantumCircuit(2)
        circuit.acp(0.5, 0, 1)
        self.assertEqual(circuit[0].operation.name, "acp")

    def test_acp_invalid(self):
        """Test that ACPhase raises errors for invalid arguments."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, circuit.acp, 0.5, cr[0], cr[1])
        self.assertRaises(CircuitError, circuit.acp, 0.5, qr[0], qr[0])

    def test_acp_gate_matrix(self):
        """Test the unitary matrix of the ACPhase gate."""
        import cmath

        theta = math.pi / 4
        eith = cmath.exp(1j * theta)
        expected = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, eith, 0], [0, 0, 0, 1]],
            dtype=np.complex128,
        )
        gate = ACPhaseGate(theta)
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected)

    def test_acp_decomposition_matches_matrix(self):
        """Test that the gate decomposition produces the correct unitary."""
        gate = ACPhaseGate(1.23)
        decomposed_op = Operator(gate.definition)
        gate_op = Operator(gate)
        self.assertTrue(gate_op.equiv(decomposed_op))

    def test_acp_equivalence_x_cp_x(self):
        """Test that ACPhase is equivalent to X-CP-X on the control qubit."""
        theta = 0.7
        expected = QuantumCircuit(2)
        expected.x(0)
        expected.cp(theta, 0, 1)
        expected.x(0)

        actual = QuantumCircuit(2)
        actual.acp(theta, 0, 1)

        self.assertEqual(Operator(expected), Operator(actual))

    def test_acp_inverse(self):
        """Test that inverse negates the angle."""
        gate = ACPhaseGate(1.5)
        inverse = gate.inverse()
        self.assertIsInstance(inverse, ACPhaseGate)
        self.assertAlmostEqual(float(inverse.params[0]), -1.5)

    def test_acp_inverse_product_is_identity(self):
        """Test that gate * inverse = identity."""
        theta = 0.8
        qc = QuantumCircuit(2)
        qc.acp(theta, 0, 1)
        qc.acp(-theta, 0, 1)
        op = Operator(qc)
        np.testing.assert_array_almost_equal(op.data, np.eye(4))

    def test_acp_label(self):
        """Test that label is passed through."""
        circuit = QuantumCircuit(2)
        circuit.acp(0.5, 0, 1, label="test_label")
        self.assertEqual(circuit[0].operation.label, "test_label")

    def test_acp_is_unitary(self):
        """Test that the ACPhase gate matrix is unitary."""
        gate = ACPhaseGate(2.1)
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(4))

    def test_acp_zero_angle(self):
        """Test that ACPhase(0) is identity."""
        gate = ACPhaseGate(0)
        np.testing.assert_array_almost_equal(gate.to_matrix(), np.eye(4))


class TestACUGate(QiskitTestCase):
    """Tests for the anti-controlled U gate."""

    def test_acu_circuit_method(self):
        """Test applying ACU via the QuantumCircuit method."""
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.acu(0.5, 0.3, 0.1, 0.2, qr[0], qr[1])
        self.assertEqual(circuit[0].operation.name, "acu")
        self.assertEqual(circuit[0].qubits, (qr[0], qr[1]))

    def test_acu_wires(self):
        """Test applying ACU using integer wire indices."""
        circuit = QuantumCircuit(2)
        circuit.acu(0.5, 0.3, 0.1, 0.2, 0, 1)
        self.assertEqual(circuit[0].operation.name, "acu")

    def test_acu_invalid(self):
        """Test that ACU raises errors for invalid arguments."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, circuit.acu, 0.5, 0.3, 0.1, 0.2, cr[0], cr[1])
        self.assertRaises(CircuitError, circuit.acu, 0.5, 0.3, 0.1, 0.2, qr[0], qr[0])

    def test_acu_gate_matrix(self):
        """Test the unitary matrix of the ACU gate."""
        import cmath

        theta, phi, lam, gamma = 0.5, 0.3, 0.1, 0.2
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        a = cmath.exp(1j * gamma) * cos
        b = -cmath.exp(1j * (gamma + lam)) * sin
        c = cmath.exp(1j * (gamma + phi)) * sin
        d = cmath.exp(1j * (gamma + phi + lam)) * cos
        expected = np.array(
            [[a, 0, b, 0], [0, 1, 0, 0], [c, 0, d, 0], [0, 0, 0, 1]],
            dtype=np.complex128,
        )
        gate = ACUGate(theta, phi, lam, gamma)
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected)

    def test_acu_decomposition_matches_matrix(self):
        """Test that the gate decomposition produces the correct unitary."""
        gate = ACUGate(0.5, 0.3, 0.1, 0.2)
        decomposed_op = Operator(gate.definition)
        gate_op = Operator(gate)
        self.assertTrue(gate_op.equiv(decomposed_op))

    def test_acu_equivalence_x_cu_x(self):
        """Test that ACU is equivalent to X-CU-X on the control qubit."""
        theta, phi, lam, gamma = 0.5, 0.3, 0.1, 0.2
        expected = QuantumCircuit(2)
        expected.x(0)
        expected.cu(theta, phi, lam, gamma, 0, 1)
        expected.x(0)

        actual = QuantumCircuit(2)
        actual.acu(theta, phi, lam, gamma, 0, 1)

        self.assertEqual(Operator(expected), Operator(actual))

    def test_acu_inverse(self):
        """Test that inverse negates all parameters (with phi/lam swapped)."""
        gate = ACUGate(0.5, 0.3, 0.1, 0.2)
        inverse = gate.inverse()
        self.assertIsInstance(inverse, ACUGate)
        # CU inverse swaps phi and lam: CU(-θ, -λ, -φ, -γ)
        self.assertAlmostEqual(float(inverse.params[0]), -0.5)
        self.assertAlmostEqual(float(inverse.params[1]), -0.1)
        self.assertAlmostEqual(float(inverse.params[2]), -0.3)
        self.assertAlmostEqual(float(inverse.params[3]), -0.2)

    def test_acu_inverse_product_is_identity(self):
        """Test that gate * inverse = identity."""
        theta, phi, lam, gamma = 0.5, 0.3, 0.1, 0.2
        gate = ACUGate(theta, phi, lam, gamma)
        inv = gate.inverse()
        qc = QuantumCircuit(2)
        qc.append(gate, [0, 1])
        qc.append(inv, [0, 1])
        op = Operator(qc)
        np.testing.assert_array_almost_equal(op.data, np.eye(4))

    def test_acu_label(self):
        """Test that label is passed through."""
        circuit = QuantumCircuit(2)
        circuit.acu(0.5, 0.3, 0.1, 0.2, 0, 1, label="test_label")
        self.assertEqual(circuit[0].operation.label, "test_label")

    def test_acu_is_unitary(self):
        """Test that the ACU gate matrix is unitary."""
        gate = ACUGate(0.5, 0.3, 0.1, 0.2)
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(4))

    def test_acu_zero_params(self):
        """Test that ACU(0,0,0,0) is identity."""
        gate = ACUGate(0, 0, 0, 0)
        np.testing.assert_array_almost_equal(gate.to_matrix(), np.eye(4))


class TestACSXGate(QiskitTestCase):
    """Tests for the anti-controlled sqrt(X) gate."""

    def test_acsx_circuit_method(self):
        """Test applying ACSX via the QuantumCircuit method."""
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.acsx(qr[0], qr[1])
        self.assertEqual(circuit[0].operation.name, "acsx")
        self.assertEqual(circuit[0].qubits, (qr[0], qr[1]))

    def test_acsx_wires(self):
        """Test applying ACSX using integer wire indices."""
        circuit = QuantumCircuit(2)
        circuit.acsx(0, 1)
        self.assertEqual(circuit[0].operation.name, "acsx")

    def test_acsx_invalid(self):
        """Test that ACSX raises errors for invalid arguments."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, circuit.acsx, cr[0], cr[1])
        self.assertRaises(CircuitError, circuit.acsx, qr[0], qr[0])
        self.assertRaises(CircuitError, circuit.acsx, 0.0, qr[0])
        self.assertRaises(CircuitError, circuit.acsx, "a", qr[1])

    def test_acsx_gate_matrix(self):
        """Test the unitary matrix of the ACSX gate."""
        expected = np.array(
            [
                [0.5 + 0.5j, 0, 0.5 - 0.5j, 0],
                [0, 1, 0, 0],
                [0.5 - 0.5j, 0, 0.5 + 0.5j, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )
        gate = ACSXGate()
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected)

    def test_acsx_decomposition_matches_matrix(self):
        """Test that the gate decomposition produces the correct unitary."""
        gate = ACSXGate()
        decomposed_op = Operator(gate.definition)
        gate_op = Operator(gate)
        self.assertTrue(gate_op.equiv(decomposed_op))

    def test_acsx_equivalence_x_csx_x(self):
        """Test that ACSX is equivalent to X-CSX-X on the control qubit."""
        expected = QuantumCircuit(2)
        expected.x(0)
        expected.csx(0, 1)
        expected.x(0)

        actual = QuantumCircuit(2)
        actual.acsx(0, 1)

        self.assertEqual(Operator(expected), Operator(actual))

    def test_acsx_inverse(self):
        """Test that ACSX inverse is ACSXdg."""
        gate = ACSXGate()
        inverse = gate.inverse()
        self.assertIsInstance(inverse, ACSXdgGate)

    def test_acsx_inverse_product_is_identity(self):
        """Test that gate * inverse = identity."""
        qc = QuantumCircuit(2)
        qc.acsx(0, 1)
        qc.acsxdg(0, 1)
        op = Operator(qc)
        np.testing.assert_array_almost_equal(op.data, np.eye(4))

    def test_acsx_label(self):
        """Test that label is passed through."""
        circuit = QuantumCircuit(2)
        circuit.acsx(0, 1, label="test_label")
        self.assertEqual(circuit[0].operation.label, "test_label")

    def test_acsx_is_unitary(self):
        """Test that the ACSX gate matrix is unitary."""
        gate = ACSXGate()
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(4))

    def test_acsx_broadcast(self):
        """Test that ACSX broadcasts across qubit registers."""
        qr1 = QuantumRegister(3, "q1")
        qr2 = QuantumRegister(3, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.acsx(qr1, qr2)
        self.assertEqual(len(circuit), 3)


class TestACSXdgGate(QiskitTestCase):
    """Tests for the anti-controlled sqrt(X)-dagger gate."""

    def test_acsxdg_circuit_method(self):
        """Test applying ACSXdg via the QuantumCircuit method."""
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.acsxdg(qr[0], qr[1])
        self.assertEqual(circuit[0].operation.name, "acsxdg")
        self.assertEqual(circuit[0].qubits, (qr[0], qr[1]))

    def test_acsxdg_wires(self):
        """Test applying ACSXdg using integer wire indices."""
        circuit = QuantumCircuit(2)
        circuit.acsxdg(0, 1)
        self.assertEqual(circuit[0].operation.name, "acsxdg")

    def test_acsxdg_invalid(self):
        """Test that ACSXdg raises errors for invalid arguments."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, circuit.acsxdg, cr[0], cr[1])
        self.assertRaises(CircuitError, circuit.acsxdg, qr[0], qr[0])
        self.assertRaises(CircuitError, circuit.acsxdg, 0.0, qr[0])
        self.assertRaises(CircuitError, circuit.acsxdg, "a", qr[1])

    def test_acsxdg_gate_matrix(self):
        """Test the unitary matrix of the ACSXdg gate."""
        expected = np.array(
            [
                [0.5 - 0.5j, 0, 0.5 + 0.5j, 0],
                [0, 1, 0, 0],
                [0.5 + 0.5j, 0, 0.5 - 0.5j, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )
        gate = ACSXdgGate()
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected)

    def test_acsxdg_decomposition_matches_matrix(self):
        """Test that the gate decomposition produces the correct unitary."""
        gate = ACSXdgGate()
        decomposed_op = Operator(gate.definition)
        gate_op = Operator(gate)
        self.assertTrue(gate_op.equiv(decomposed_op))

    def test_acsxdg_inverse(self):
        """Test that ACSXdg inverse is ACSX."""
        gate = ACSXdgGate()
        inverse = gate.inverse()
        self.assertIsInstance(inverse, ACSXGate)

    def test_acsxdg_inverse_product_is_identity(self):
        """Test that gate * inverse = identity."""
        qc = QuantumCircuit(2)
        qc.acsxdg(0, 1)
        qc.acsx(0, 1)
        op = Operator(qc)
        np.testing.assert_array_almost_equal(op.data, np.eye(4))

    def test_acsxdg_label(self):
        """Test that label is passed through."""
        circuit = QuantumCircuit(2)
        circuit.acsxdg(0, 1, label="test_label")
        self.assertEqual(circuit[0].operation.label, "test_label")

    def test_acsxdg_is_unitary(self):
        """Test that the ACSXdg gate matrix is unitary."""
        gate = ACSXdgGate()
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(4))

    def test_acsxdg_broadcast(self):
        """Test that ACSXdg broadcasts across qubit registers."""
        qr1 = QuantumRegister(3, "q1")
        qr2 = QuantumRegister(3, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.acsxdg(qr1, qr2)
        self.assertEqual(len(circuit), 3)


class TestACSGate(QiskitTestCase):
    """Tests for the anti-controlled S gate."""

    def test_acs_circuit_method(self):
        """Test applying ACS via the QuantumCircuit method."""
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.acs(qr[0], qr[1])
        self.assertEqual(circuit[0].operation.name, "acs")
        self.assertEqual(circuit[0].qubits, (qr[0], qr[1]))

    def test_acs_wires(self):
        """Test applying ACS using integer wire indices."""
        circuit = QuantumCircuit(2)
        circuit.acs(0, 1)
        self.assertEqual(circuit[0].operation.name, "acs")

    def test_acs_invalid(self):
        """Test that ACS raises errors for invalid arguments."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, circuit.acs, cr[0], cr[1])
        self.assertRaises(CircuitError, circuit.acs, qr[0], qr[0])
        self.assertRaises(CircuitError, circuit.acs, 0.0, qr[0])
        self.assertRaises(CircuitError, circuit.acs, "a", qr[1])

    def test_acs_gate_matrix(self):
        """Test the unitary matrix of the ACS gate."""
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1j, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )
        gate = ACSGate()
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected)

    def test_acs_decomposition_matches_matrix(self):
        """Test that the gate decomposition produces the correct unitary."""
        gate = ACSGate()
        decomposed_op = Operator(gate.definition)
        gate_op = Operator(gate)
        self.assertTrue(gate_op.equiv(decomposed_op))

    def test_acs_equivalence_x_cs_x(self):
        """Test that ACS is equivalent to X-CS-X on the control qubit."""
        expected = QuantumCircuit(2)
        expected.x(0)
        expected.cs(0, 1)
        expected.x(0)

        actual = QuantumCircuit(2)
        actual.acs(0, 1)

        self.assertEqual(Operator(expected), Operator(actual))

    def test_acs_inverse(self):
        """Test that ACS inverse is ACSdg."""
        gate = ACSGate()
        inverse = gate.inverse()
        self.assertIsInstance(inverse, ACSdgGate)

    def test_acs_inverse_product_is_identity(self):
        """Test that gate * inverse = identity."""
        qc = QuantumCircuit(2)
        qc.acs(0, 1)
        qc.acsdg(0, 1)
        op = Operator(qc)
        np.testing.assert_array_almost_equal(op.data, np.eye(4))

    def test_acs_label(self):
        """Test that label is passed through."""
        circuit = QuantumCircuit(2)
        circuit.acs(0, 1, label="test_label")
        self.assertEqual(circuit[0].operation.label, "test_label")

    def test_acs_is_unitary(self):
        """Test that the ACS gate matrix is unitary."""
        gate = ACSGate()
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(4))

    def test_acs_broadcast(self):
        """Test that ACS broadcasts across qubit registers."""
        qr1 = QuantumRegister(3, "q1")
        qr2 = QuantumRegister(3, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.acs(qr1, qr2)
        self.assertEqual(len(circuit), 3)


class TestACSdgGate(QiskitTestCase):
    """Tests for the anti-controlled S-dagger gate."""

    def test_acsdg_circuit_method(self):
        """Test applying ACSdg via the QuantumCircuit method."""
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.acsdg(qr[0], qr[1])
        self.assertEqual(circuit[0].operation.name, "acsdg")
        self.assertEqual(circuit[0].qubits, (qr[0], qr[1]))

    def test_acsdg_wires(self):
        """Test applying ACSdg using integer wire indices."""
        circuit = QuantumCircuit(2)
        circuit.acsdg(0, 1)
        self.assertEqual(circuit[0].operation.name, "acsdg")

    def test_acsdg_invalid(self):
        """Test that ACSdg raises errors for invalid arguments."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, circuit.acsdg, cr[0], cr[1])
        self.assertRaises(CircuitError, circuit.acsdg, qr[0], qr[0])
        self.assertRaises(CircuitError, circuit.acsdg, 0.0, qr[0])
        self.assertRaises(CircuitError, circuit.acsdg, "a", qr[1])

    def test_acsdg_gate_matrix(self):
        """Test the unitary matrix of the ACSdg gate."""
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, -1j, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )
        gate = ACSdgGate()
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected)

    def test_acsdg_decomposition_matches_matrix(self):
        """Test that the gate decomposition produces the correct unitary."""
        gate = ACSdgGate()
        decomposed_op = Operator(gate.definition)
        gate_op = Operator(gate)
        self.assertTrue(gate_op.equiv(decomposed_op))

    def test_acsdg_equivalence_x_csdg_x(self):
        """Test that ACSdg is equivalent to X-CSdg-X on the control qubit."""
        expected = QuantumCircuit(2)
        expected.x(0)
        expected.csdg(0, 1)
        expected.x(0)

        actual = QuantumCircuit(2)
        actual.acsdg(0, 1)

        self.assertEqual(Operator(expected), Operator(actual))

    def test_acsdg_inverse(self):
        """Test that ACSdg inverse is ACS."""
        gate = ACSdgGate()
        inverse = gate.inverse()
        self.assertIsInstance(inverse, ACSGate)

    def test_acsdg_inverse_product_is_identity(self):
        """Test that gate * inverse = identity."""
        qc = QuantumCircuit(2)
        qc.acsdg(0, 1)
        qc.acs(0, 1)
        op = Operator(qc)
        np.testing.assert_array_almost_equal(op.data, np.eye(4))

    def test_acsdg_label(self):
        """Test that label is passed through."""
        circuit = QuantumCircuit(2)
        circuit.acsdg(0, 1, label="test_label")
        self.assertEqual(circuit[0].operation.label, "test_label")

    def test_acsdg_is_unitary(self):
        """Test that the ACSdg gate matrix is unitary."""
        gate = ACSdgGate()
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(4))

    def test_acsdg_broadcast(self):
        """Test that ACSdg broadcasts across qubit registers."""
        qr1 = QuantumRegister(3, "q1")
        qr2 = QuantumRegister(3, "q2")
        circuit = QuantumCircuit(qr1, qr2)
        circuit.acsdg(qr1, qr2)
        self.assertEqual(len(circuit), 3)


class TestACU1Gate(QiskitTestCase):
    """Tests for the anti-controlled U1 gate."""

    def test_acu1_circuit_method(self):
        """Test applying ACU1 via the QuantumCircuit method."""
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.acu1(0.5, qr[0], qr[1])
        self.assertEqual(circuit[0].operation.name, "acu1")
        self.assertEqual(circuit[0].qubits, (qr[0], qr[1]))

    def test_acu1_wires(self):
        """Test applying ACU1 using integer wire indices."""
        circuit = QuantumCircuit(2)
        circuit.acu1(0.5, 0, 1)
        self.assertEqual(circuit[0].operation.name, "acu1")

    def test_acu1_invalid(self):
        """Test that ACU1 raises errors for invalid arguments."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, circuit.acu1, 0.5, cr[0], cr[1])
        self.assertRaises(CircuitError, circuit.acu1, 0.5, qr[0], qr[0])

    def test_acu1_gate_matrix(self):
        """Test the unitary matrix of the ACU1 gate."""
        import cmath

        lam = math.pi / 4
        eith = cmath.exp(1j * lam)
        expected = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, eith, 0], [0, 0, 0, 1]],
            dtype=np.complex128,
        )
        gate = ACU1Gate(lam)
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected)

    def test_acu1_decomposition_matches_matrix(self):
        """Test that the gate decomposition produces the correct unitary."""
        gate = ACU1Gate(1.23)
        decomposed_op = Operator(gate.definition)
        gate_op = Operator(gate)
        self.assertTrue(gate_op.equiv(decomposed_op))

    def test_acu1_equivalence_x_cu1_x(self):
        """Test that ACU1 is equivalent to X-CU1-X on the control qubit."""
        from qiskit.circuit.library.standard_gates.u1 import CU1Gate

        lam = 0.7
        expected = QuantumCircuit(2)
        expected.x(0)
        expected.append(CU1Gate(lam), [0, 1])
        expected.x(0)

        actual = QuantumCircuit(2)
        actual.acu1(lam, 0, 1)

        self.assertEqual(Operator(expected), Operator(actual))

    def test_acu1_inverse(self):
        """Test that inverse negates the angle."""
        gate = ACU1Gate(1.5)
        inverse = gate.inverse()
        self.assertIsInstance(inverse, ACU1Gate)
        self.assertAlmostEqual(float(inverse.params[0]), -1.5)

    def test_acu1_inverse_product_is_identity(self):
        """Test that gate * inverse = identity."""
        lam = 0.8
        qc = QuantumCircuit(2)
        qc.acu1(lam, 0, 1)
        qc.acu1(-lam, 0, 1)
        op = Operator(qc)
        np.testing.assert_array_almost_equal(op.data, np.eye(4))

    def test_acu1_label(self):
        """Test that label is passed through."""
        circuit = QuantumCircuit(2)
        circuit.acu1(0.5, 0, 1, label="test_label")
        self.assertEqual(circuit[0].operation.label, "test_label")

    def test_acu1_is_unitary(self):
        """Test that the ACU1 gate matrix is unitary."""
        gate = ACU1Gate(2.1)
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(4))

    def test_acu1_zero_angle(self):
        """Test that ACU1(0) is identity."""
        gate = ACU1Gate(0)
        np.testing.assert_array_almost_equal(gate.to_matrix(), np.eye(4))


class TestACU3Gate(QiskitTestCase):
    """Tests for the anti-controlled U3 gate."""

    def test_acu3_circuit_method(self):
        """Test applying ACU3 via the QuantumCircuit method."""
        qr = QuantumRegister(2, "q")
        circuit = QuantumCircuit(qr)
        circuit.acu3(0.5, 0.3, 0.1, qr[0], qr[1])
        self.assertEqual(circuit[0].operation.name, "acu3")
        self.assertEqual(circuit[0].qubits, (qr[0], qr[1]))

    def test_acu3_wires(self):
        """Test applying ACU3 using integer wire indices."""
        circuit = QuantumCircuit(2)
        circuit.acu3(0.5, 0.3, 0.1, 0, 1)
        self.assertEqual(circuit[0].operation.name, "acu3")

    def test_acu3_invalid(self):
        """Test that ACU3 raises errors for invalid arguments."""
        qr = QuantumRegister(2, "q")
        cr = ClassicalRegister(2, "c")
        circuit = QuantumCircuit(qr, cr)
        self.assertRaises(CircuitError, circuit.acu3, 0.5, 0.3, 0.1, cr[0], cr[1])
        self.assertRaises(CircuitError, circuit.acu3, 0.5, 0.3, 0.1, qr[0], qr[0])

    def test_acu3_gate_matrix(self):
        """Test the unitary matrix of the ACU3 gate."""
        import cmath

        theta, phi, lam = 0.5, 0.3, 0.1
        cos = math.cos(theta / 2)
        sin = math.sin(theta / 2)
        expected = np.array(
            [
                [cos, 0, -cmath.exp(1j * lam) * sin, 0],
                [0, 1, 0, 0],
                [cmath.exp(1j * phi) * sin, 0, cmath.exp(1j * (phi + lam)) * cos, 0],
                [0, 0, 0, 1],
            ],
            dtype=np.complex128,
        )
        gate = ACU3Gate(theta, phi, lam)
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected)

    def test_acu3_decomposition_matches_matrix(self):
        """Test that the gate decomposition produces the correct unitary."""
        gate = ACU3Gate(0.5, 0.3, 0.1)
        decomposed_op = Operator(gate.definition)
        gate_op = Operator(gate)
        self.assertTrue(gate_op.equiv(decomposed_op))

    def test_acu3_equivalence_x_cu3_x(self):
        """Test that ACU3 is equivalent to X-CU3-X on the control qubit."""
        from qiskit.circuit.library.standard_gates.u3 import CU3Gate

        theta, phi, lam = 0.5, 0.3, 0.1
        expected = QuantumCircuit(2)
        expected.x(0)
        expected.append(CU3Gate(theta, phi, lam), [0, 1])
        expected.x(0)

        actual = QuantumCircuit(2)
        actual.acu3(theta, phi, lam, 0, 1)

        self.assertEqual(Operator(expected), Operator(actual))

    def test_acu3_inverse(self):
        """Test that inverse swaps and negates phi/lam."""
        gate = ACU3Gate(0.5, 0.3, 0.1)
        inverse = gate.inverse()
        self.assertIsInstance(inverse, ACU3Gate)
        # U3 inverse: U3(-θ, -λ, -φ)
        self.assertAlmostEqual(float(inverse.params[0]), -0.5)
        self.assertAlmostEqual(float(inverse.params[1]), -0.1)
        self.assertAlmostEqual(float(inverse.params[2]), -0.3)

    def test_acu3_inverse_product_is_identity(self):
        """Test that gate * inverse = identity."""
        theta, phi, lam = 0.5, 0.3, 0.1
        gate = ACU3Gate(theta, phi, lam)
        inv = gate.inverse()
        qc = QuantumCircuit(2)
        qc.append(gate, [0, 1])
        qc.append(inv, [0, 1])
        op = Operator(qc)
        np.testing.assert_array_almost_equal(op.data, np.eye(4))

    def test_acu3_label(self):
        """Test that label is passed through."""
        circuit = QuantumCircuit(2)
        circuit.acu3(0.5, 0.3, 0.1, 0, 1, label="test_label")
        self.assertEqual(circuit[0].operation.label, "test_label")

    def test_acu3_is_unitary(self):
        """Test that the ACU3 gate matrix is unitary."""
        gate = ACU3Gate(0.5, 0.3, 0.1)
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(4))

    def test_acu3_zero_params(self):
        """Test that ACU3(0,0,0) is identity."""
        gate = ACU3Gate(0, 0, 0)
        np.testing.assert_array_almost_equal(gate.to_matrix(), np.eye(4))


class TestAMCXGate(QiskitTestCase):
    """Tests for AMCXGate (anti-controlled/controlled multi-qubit X gate)."""

    def test_amcx_circuit_method(self):
        """Test the QuantumCircuit.amcx() method."""
        qc = QuantumCircuit(5)
        qc.amcx([0, 2], [1, 3], 4)
        self.assertEqual(qc[0].operation.name, "amcx")
        self.assertEqual(qc[0].operation.num_qubits, 5)

    def test_amcx_wires(self):
        """Test that the wires are correct."""
        qc = QuantumCircuit(5)
        qc.amcx([0, 2], [1, 3], 4)
        qubits = [qc.qubits.index(q) for q in qc[0].qubits]
        self.assertEqual(qubits, [0, 2, 1, 3, 4])

    def test_amcx_invalid_zero_controls(self):
        """Test that AMCXGate raises with 0 total controls."""
        with self.assertRaises(ValueError):
            AMCXGate(0, 0)

    def test_amcx_invalid_negative(self):
        """Test that AMCXGate raises with negative control counts."""
        with self.assertRaises(ValueError):
            AMCXGate(-1, 2)
        with self.assertRaises(ValueError):
            AMCXGate(2, -1)

    def test_amcx_properties(self):
        """Test num_anti_ctrl_qubits and num_ctrl_qubits properties."""
        gate = AMCXGate(2, 3)
        self.assertEqual(gate.num_anti_ctrl_qubits, 2)
        self.assertEqual(gate.num_ctrl_qubits, 3)
        self.assertEqual(gate.num_qubits, 6)

    def test_amcx_1_anti_0_ctrl_equals_acx(self):
        """Test that AMCX(1, 0) is equivalent to ACX."""
        gate = AMCXGate(1, 0)
        acx = ACXGate()
        np.testing.assert_array_almost_equal(gate.to_matrix(), acx.to_matrix())

    def test_amcx_0_anti_1_ctrl_equals_cx(self):
        """Test that AMCX(0, 1) is equivalent to CX."""
        from qiskit.circuit.library import CXGate

        gate = AMCXGate(0, 1)
        cx = CXGate()
        np.testing.assert_array_almost_equal(gate.to_matrix(), cx.to_matrix())

    def test_amcx_0_anti_2_ctrl_equals_ccx(self):
        """Test that AMCX(0, 2) is equivalent to CCX (Toffoli)."""
        from qiskit.circuit.library import CCXGate

        gate = AMCXGate(0, 2)
        ccx = CCXGate()
        np.testing.assert_array_almost_equal(gate.to_matrix(), ccx.to_matrix())

    def test_amcx_matrix_2anti_1ctrl(self):
        """Test AMCX(2, 1) matrix: flips target when anti-ctrls=|00⟩, ctrl=|1⟩."""
        gate = AMCXGate(2, 1)
        mat = gate.to_matrix()
        dim = 8
        # The activation pattern: qubits 0,1 = 0 (anti), qubit 2 = 1 (ctrl)
        # In little-endian: binary = target(q3) ctrl(q2) anti1(q1) anti0(q0)
        # ctrl_pattern = bit 2 set = 0b100 = 4
        # state_0 = 4 (target=0), state_1 = 4 + 8 = not right... let me think
        # 3 qubits total (2 anti + 1 ctrl), target is qubit index 3
        # Wait, num_qubits = 2 + 1 + 1 = 4, so dim = 16
        self.assertEqual(mat.shape, (16, 16))

        # ctrl_pattern: bits 0,1 = 0 (anti), bit 2 = 1 (ctrl) → 0b100 = 4
        # state_0 = 4 (target bit 3 = 0), state_1 = 4 + 8 = 12 (target bit 3 = 1)
        # These two should be swapped
        expected = np.eye(16)
        expected[4, 4] = 0
        expected[12, 12] = 0
        expected[4, 12] = 1
        expected[12, 4] = 1
        np.testing.assert_array_almost_equal(mat, expected)

    def test_amcx_matrix_1anti_1ctrl(self):
        """Test AMCX(1, 1) matrix: flips target when anti-ctrl=|0⟩, ctrl=|1⟩."""
        gate = AMCXGate(1, 1)
        mat = gate.to_matrix()
        self.assertEqual(mat.shape, (8, 8))

        # ctrl_pattern: bit 0 = 0 (anti), bit 1 = 1 (ctrl) → 0b10 = 2
        # state_0 = 2 (target bit 2 = 0), state_1 = 2 + 4 = 6 (target bit 2 = 1)
        expected = np.eye(8)
        expected[2, 2] = 0
        expected[6, 6] = 0
        expected[2, 6] = 1
        expected[6, 2] = 1
        np.testing.assert_array_almost_equal(mat, expected)

    def test_amcx_decomposition_matches_matrix(self):
        """Test that the decomposition gives the same unitary as the matrix."""
        for na, nc in [(1, 1), (2, 1), (1, 2), (2, 2)]:
            gate = AMCXGate(na, nc)
            mat = gate.to_matrix()
            decomp_op = Operator(gate.definition)
            np.testing.assert_array_almost_equal(
                decomp_op.data, mat,
                err_msg=f"AMCX({na},{nc}) decomposition doesn't match matrix",
            )

    def test_amcx_self_inverse(self):
        """Test that AMCX is self-inverse."""
        gate = AMCXGate(2, 1)
        inv = gate.inverse()
        self.assertIsInstance(inv, AMCXGate)
        self.assertEqual(inv.num_anti_ctrl_qubits, 2)
        self.assertEqual(inv.num_ctrl_qubits, 1)

        # gate * inverse = identity
        n = gate.num_qubits
        qc = QuantumCircuit(n)
        qc.append(gate, list(range(n)))
        qc.append(inv, list(range(n)))
        op = Operator(qc)
        np.testing.assert_array_almost_equal(op.data, np.eye(2**n))

    def test_amcx_is_unitary(self):
        """Test that the AMCX gate matrix is unitary."""
        for na, nc in [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)]:
            gate = AMCXGate(na, nc)
            mat = gate.to_matrix()
            dim = 2 ** gate.num_qubits
            np.testing.assert_array_almost_equal(
                mat @ mat.conj().T, np.eye(dim),
                err_msg=f"AMCX({na},{nc}) is not unitary",
            )

    def test_amcx_label(self):
        """Test that label is passed through."""
        qc = QuantumCircuit(3)
        qc.amcx([0], [1], 2, label="test_label")
        self.assertEqual(qc[0].operation.label, "test_label")

    def test_amcx_equality(self):
        """Test equality comparison."""
        g1 = AMCXGate(2, 1)
        g2 = AMCXGate(2, 1)
        g3 = AMCXGate(1, 2)
        self.assertEqual(g1, g2)
        self.assertNotEqual(g1, g3)

    def test_amcx_only_anti_controls(self):
        """Test AMCX with only anti-controls (no regular controls)."""
        gate = AMCXGate(2, 0)
        mat = gate.to_matrix()
        # Should flip target when both anti-controls are |0⟩
        # ctrl_pattern = 0 (no regular ctrl bits set)
        # state_0 = 0, state_1 = 0 + 4 = 4 (target at bit 2)
        expected = np.eye(8)
        expected[0, 0] = 0
        expected[4, 4] = 0
        expected[0, 4] = 1
        expected[4, 0] = 1
        np.testing.assert_array_almost_equal(mat, expected)

    def test_amcx_only_regular_controls(self):
        """Test AMCX with only regular controls (no anti-controls)."""
        from qiskit.circuit.library import CCXGate

        gate = AMCXGate(0, 2)
        mat = gate.to_matrix()
        ccx = CCXGate()
        np.testing.assert_array_almost_equal(mat, ccx.to_matrix())


class TestAMCPhaseGate(QiskitTestCase):
    """Tests for AMCPhaseGate (anti-controlled/controlled multi-qubit Phase gate)."""

    def test_amcp_circuit_method(self):
        """Test the QuantumCircuit.amcp() method."""
        qc = QuantumCircuit(5)
        qc.amcp(0.5, [0, 2], [1, 3], 4)
        self.assertEqual(qc[0].operation.name, "amcp")
        self.assertEqual(qc[0].operation.num_qubits, 5)

    def test_amcp_wires(self):
        """Test that the wires are correct."""
        qc = QuantumCircuit(5)
        qc.amcp(0.5, [0, 2], [1, 3], 4)
        qubits = [qc.qubits.index(q) for q in qc[0].qubits]
        self.assertEqual(qubits, [0, 2, 1, 3, 4])

    def test_amcp_invalid_zero_controls(self):
        """Test that AMCPhaseGate raises with 0 total controls."""
        with self.assertRaises(ValueError):
            AMCPhaseGate(0.5, 0, 0)

    def test_amcp_invalid_negative(self):
        """Test that AMCPhaseGate raises with negative control counts."""
        with self.assertRaises(ValueError):
            AMCPhaseGate(0.5, -1, 2)
        with self.assertRaises(ValueError):
            AMCPhaseGate(0.5, 2, -1)

    def test_amcp_properties(self):
        """Test num_anti_ctrl_qubits and num_ctrl_qubits properties."""
        gate = AMCPhaseGate(0.5, 2, 3)
        self.assertEqual(gate.num_anti_ctrl_qubits, 2)
        self.assertEqual(gate.num_ctrl_qubits, 3)
        self.assertEqual(gate.num_qubits, 6)

    def test_amcp_1_anti_0_ctrl_equals_acp(self):
        """Test that AMCPhase(lam, 1, 0) is equivalent to ACPhase."""
        lam = 0.7
        gate = AMCPhaseGate(lam, 1, 0)
        acp = ACPhaseGate(lam)
        np.testing.assert_array_almost_equal(gate.to_matrix(), acp.to_matrix())

    def test_amcp_0_anti_1_ctrl_equals_cp(self):
        """Test that AMCPhase(lam, 0, 1) is equivalent to CPhase."""
        from qiskit.circuit.library import CPhaseGate

        lam = 0.7
        gate = AMCPhaseGate(lam, 0, 1)
        cp = CPhaseGate(lam)
        np.testing.assert_array_almost_equal(gate.to_matrix(), cp.to_matrix())

    def test_amcp_0_anti_2_ctrl_equals_mcphase(self):
        """Test that AMCPhase(lam, 0, 2) is equivalent to MCPhase with 2 controls."""
        from qiskit.circuit.library import MCPhaseGate

        lam = 0.7
        gate = AMCPhaseGate(lam, 0, 2)
        mcp = MCPhaseGate(lam, 2)
        # MCPhaseGate doesn't have to_matrix(), so compare via Operator
        np.testing.assert_array_almost_equal(gate.to_matrix(), Operator(mcp).data)

    def test_amcp_matrix_1anti_1ctrl(self):
        """Test AMCPhase(lam, 1, 1) matrix: applies phase when anti-ctrl=|0⟩, ctrl=|1⟩."""
        lam = 0.7
        gate = AMCPhaseGate(lam, 1, 1)
        mat = gate.to_matrix()
        self.assertEqual(mat.shape, (8, 8))

        # ctrl_pattern: bit 0 = 0 (anti), bit 1 = 1 (ctrl) → 0b10 = 2
        # Phase applied at state: ctrl_pattern | (1 << target_bit) = 2 | 4 = 6
        expected = np.eye(8, dtype=complex)
        expected[6, 6] = np.exp(1j * lam)
        np.testing.assert_array_almost_equal(mat, expected)

    def test_amcp_decomposition_matches_matrix(self):
        """Test that the decomposition gives the same unitary as the matrix."""
        lam = 0.7
        for na, nc in [(1, 1), (2, 1), (1, 2), (2, 2)]:
            gate = AMCPhaseGate(lam, na, nc)
            mat = gate.to_matrix()
            decomp_op = Operator(gate.definition)
            np.testing.assert_array_almost_equal(
                decomp_op.data, mat,
                err_msg=f"AMCPhase({na},{nc}) decomposition doesn't match matrix",
            )

    def test_amcp_inverse(self):
        """Test that the inverse has negated parameter."""
        lam = 0.7
        gate = AMCPhaseGate(lam, 2, 1)
        inv = gate.inverse()
        self.assertIsInstance(inv, AMCPhaseGate)
        self.assertEqual(inv.num_anti_ctrl_qubits, 2)
        self.assertEqual(inv.num_ctrl_qubits, 1)
        self.assertAlmostEqual(float(inv.params[0]), -lam)

    def test_amcp_inverse_product_is_identity(self):
        """Test that gate * inverse = identity."""
        lam = 0.7
        gate = AMCPhaseGate(lam, 1, 1)
        inv = gate.inverse()
        n = gate.num_qubits
        qc = QuantumCircuit(n)
        qc.append(gate, list(range(n)))
        qc.append(inv, list(range(n)))
        op = Operator(qc)
        np.testing.assert_array_almost_equal(op.data, np.eye(2**n))

    def test_amcp_is_unitary(self):
        """Test that the AMCPhase gate matrix is unitary."""
        lam = 0.7
        for na, nc in [(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)]:
            gate = AMCPhaseGate(lam, na, nc)
            mat = gate.to_matrix()
            dim = 2 ** gate.num_qubits
            np.testing.assert_array_almost_equal(
                mat @ mat.conj().T, np.eye(dim),
                err_msg=f"AMCPhase({na},{nc}) is not unitary",
            )

    def test_amcp_label(self):
        """Test that label is passed through."""
        qc = QuantumCircuit(3)
        qc.amcp(0.5, [0], [1], 2, label="test_label")
        self.assertEqual(qc[0].operation.label, "test_label")

    def test_amcp_equality(self):
        """Test equality comparison."""
        g1 = AMCPhaseGate(0.5, 2, 1)
        g2 = AMCPhaseGate(0.5, 2, 1)
        g3 = AMCPhaseGate(0.5, 1, 2)
        g4 = AMCPhaseGate(0.7, 2, 1)
        self.assertEqual(g1, g2)
        self.assertNotEqual(g1, g3)
        self.assertNotEqual(g1, g4)

    def test_amcp_zero_angle_is_identity(self):
        """Test that AMCPhase(0, ...) is identity."""
        gate = AMCPhaseGate(0, 1, 1)
        np.testing.assert_array_almost_equal(gate.to_matrix(), np.eye(8))

    def test_amcp_only_anti_controls(self):
        """Test AMCPhase with only anti-controls (no regular controls)."""
        lam = 0.7
        gate = AMCPhaseGate(lam, 2, 0)
        mat = gate.to_matrix()
        # Phase at state: ctrl_pattern=0 | (1 << 2) = 4
        expected = np.eye(8, dtype=complex)
        expected[4, 4] = np.exp(1j * lam)
        np.testing.assert_array_almost_equal(mat, expected)


class TestACCXGate(QiskitTestCase):
    """Tests for ACCXGate (anti-controlled Toffoli gate)."""

    def test_accx_circuit_method(self):
        """Test the QuantumCircuit.accx() method."""
        qc = QuantumCircuit(3)
        qc.accx(0, 1, 2)
        self.assertEqual(qc[0].operation.name, "accx")
        self.assertEqual(qc[0].operation.num_qubits, 3)

    def test_accx_wires(self):
        """Test that the wires are correct."""
        qc = QuantumCircuit(3)
        qc.accx(0, 1, 2)
        qubits = [qc.qubits.index(q) for q in qc[0].qubits]
        self.assertEqual(qubits, [0, 1, 2])

    def test_accx_invalid(self):
        """Test that ACCX raises when applied to fewer than 3 qubits."""
        qc = QuantumCircuit(2)
        with self.assertRaises(CircuitError):
            qc.accx(0, 1, 1)

    def test_accx_gate_matrix(self):
        """Test the ACCX gate matrix."""
        gate = ACCXGate()
        mat = gate.to_matrix()
        # When both anti-controls are |0⟩, target flips: |000⟩ ↔ |100⟩
        expected = np.eye(8, dtype=complex)
        expected[0, 0] = 0
        expected[4, 4] = 0
        expected[0, 4] = 1
        expected[4, 0] = 1
        np.testing.assert_array_almost_equal(mat, expected)

    def test_accx_decomposition_matches_matrix(self):
        """Test that the decomposition gives the same unitary as the matrix."""
        gate = ACCXGate()
        mat = gate.to_matrix()
        decomp_op = Operator(gate.definition)
        np.testing.assert_array_almost_equal(decomp_op.data, mat)

    def test_accx_equivalence_x_ccx_x(self):
        """Test ACCX = X⊗X, CCX, X⊗X."""
        from qiskit.circuit.library import CCXGate

        qc = QuantumCircuit(3)
        qc.x(0)
        qc.x(1)
        qc.ccx(0, 1, 2)
        qc.x(0)
        qc.x(1)
        expected = Operator(qc)
        gate = ACCXGate()
        np.testing.assert_array_almost_equal(gate.to_matrix(), expected.data)

    def test_accx_equals_amcx_2_0(self):
        """Test that ACCX is equivalent to AMCX(2, 0)."""
        accx = ACCXGate()
        amcx = AMCXGate(2, 0)
        np.testing.assert_array_almost_equal(accx.to_matrix(), amcx.to_matrix())

    def test_accx_inverse(self):
        """Test that ACCX is self-inverse."""
        gate = ACCXGate()
        inv = gate.inverse()
        self.assertIsInstance(inv, ACCXGate)
        qc = QuantumCircuit(3)
        qc.append(gate, [0, 1, 2])
        qc.append(inv, [0, 1, 2])
        op = Operator(qc)
        np.testing.assert_array_almost_equal(op.data, np.eye(8))

    def test_accx_is_unitary(self):
        """Test that the ACCX gate matrix is unitary."""
        gate = ACCXGate()
        mat = gate.to_matrix()
        np.testing.assert_array_almost_equal(mat @ mat.conj().T, np.eye(8))

    def test_accx_label(self):
        """Test that label is passed through."""
        qc = QuantumCircuit(3)
        qc.accx(0, 1, 2, label="test_label")
        self.assertEqual(qc[0].operation.label, "test_label")

    def test_accx_broadcast(self):
        """Test broadcasting with multiple qubits."""
        qr1 = QuantumRegister(2, "ctrl")
        qr2 = QuantumRegister(1, "tgt")
        qc = QuantumCircuit(qr1, qr2)
        qc.accx(qr1[0], qr1[1], qr2[0])
        self.assertEqual(len(qc), 1)


if __name__ == "__main__":
    unittest.main()
