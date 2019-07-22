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


"""Test Qiskit's power instruction operation."""

import unittest
from numpy import pi, array
from numpy.testing import assert_allclose
from numpy.linalg import matrix_power

from qiskit.transpiler import PassManager
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.extensions import SGate, SdgGate, U3Gate, UnitaryGate
from qiskit.transpiler.passes import Unroller

ATOL = 1e-8
RTOL = 1e-7


class TestPowerInt(QiskitTestCase):
    """Test Instruction.power() with integer"""

    def test_standard_1Q_two(self):
        """Test standard gate.power(2) method.
        """
        qr = QuantumRegister(1, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(SGate(), qr[:])
        expected_circ.append(SGate(), qr[:])
        expected = expected_circ.to_instruction()

        result = SGate().power(2)

        self.assertEqual(result.name, 's^2')
        self.assertEqual(result.definition, expected.definition)

    def test_standard_1Q_one(self):
        """Test standard gate.power(1) method.
        """
        qr = QuantumRegister(1, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(SGate(), qr[:])
        expected = expected_circ.to_instruction()

        result = SGate().power(1)

        self.assertEqual(result.name, 's^1')
        self.assertEqual(result.definition, expected.definition)

    def test_standard_1Q_zero(self):
        """Test standard gate.power(0) method.
        """
        qr = QuantumRegister(1, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(U3Gate(0, 0, 0), qr[:])
        expected = expected_circ.to_instruction()

        result = SGate().power(0)

        self.assertEqual(result.name, 's^0')
        self.assertEqual(result.definition, expected.definition)

    def test_standard_1Q_minus_one(self):
        """Test standard gate.power(-1) method.
        """
        qr = QuantumRegister(1, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(SdgGate(), qr[:])
        expected = expected_circ.to_instruction()

        result = SGate().power(-1)

        self.assertEqual(result.name, 's^-1')
        self.assertEqual(result.definition, expected.definition)

    def test_standard_1Q_minus_two(self):
        """Test standard gate.power(-2) method.
        """
        qr = QuantumRegister(1, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(SdgGate(), qr[:])
        expected_circ.append(SdgGate(), qr[:])
        expected = expected_circ.to_instruction()

        result = SGate().power(-2)

        self.assertEqual(result.name, 's^-2')
        self.assertEqual(result.definition, expected.definition)


class TestPowerUnroller(QiskitTestCase):
    """Test unrolling Gate.power"""

    def test_unroller_two(self):
        """Test unrolling gate.power(2).
        """
        qr = QuantumRegister(1, 'qr')

        circuit = QuantumCircuit(qr)
        circuit.append(SGate().power(2), qr[:])
        result = PassManager(Unroller('u3')).run(circuit)

        expected = QuantumCircuit(qr)
        expected.append(U3Gate(0, 0, pi / 2), qr[:])
        expected.append(U3Gate(0, 0, pi / 2), qr[:])

        self.assertEqual(result, expected)

    def test_unroller_one(self):
        """Test unrolling gate.power(1).
        """
        qr = QuantumRegister(1, 'qr')

        circuit = QuantumCircuit(qr)
        circuit.append(SGate().power(1), qr[:])
        result = PassManager(Unroller('u3')).run(circuit)

        expected = QuantumCircuit(qr)
        expected.append(U3Gate(0, 0, pi / 2), qr[:])

        self.assertEqual(result, expected)

    def test_unroller_zero(self):
        """Test unrolling gate.power(0).
        """
        qr = QuantumRegister(1, 'qr')

        circuit = QuantumCircuit(qr)
        circuit.append(SGate().power(0), qr[:])
        result = PassManager(Unroller('u3')).run(circuit)

        expected = QuantumCircuit(qr)
        expected.append(U3Gate(0, 0, 0), qr[:])

        self.assertEqual(result, expected)

    def test_unroller_minus_one(self):
        """Test unrolling gate.power(-1).
        """
        qr = QuantumRegister(1, 'qr')

        circuit = QuantumCircuit(qr)
        circuit.append(SGate().power(-1), qr[:])
        result = PassManager(Unroller('u3')).run(circuit)

        expected = QuantumCircuit(qr)
        expected.append(U3Gate(0, 0, -pi / 2), qr[:])

        self.assertEqual(result, expected)

    def test_unroller_minus_two(self):
        """Test unrolling gate.power(-2).
        """
        qr = QuantumRegister(1, 'qr')

        circuit = QuantumCircuit(qr)
        circuit.append(SGate().power(-2), qr[:])
        result = PassManager(Unroller('u3')).run(circuit)

        expected = QuantumCircuit(qr)
        expected.append(U3Gate(0, 0, -pi / 2), qr[:])
        expected.append(U3Gate(0, 0, -pi / 2), qr[:])

        self.assertEqual(result, expected)


class TestGateFloat(QiskitTestCase):
    """Test Gate.power() with float"""

    def test_unitary_sqrt(self):
        """Test UnitaryGate.power(1/2) method.
        """
        expected = array([[0.5 + 0.5j, 0.5 + 0.5j],
                          [-0.5 - 0.5j, 0.5 + 0.5j]])
        result = UnitaryGate([[0, 1j], [-1j, 0]]).power(1 / 2)
        self.assertEqual(result.name, 'unitary')
        assert_allclose(result.to_matrix(), expected)

    def test_starndard_sqrt(self):
        """Test standard Gate.sqrt() method.
        """
        expected = array([[1 + 0.j, 0 + 0.j],
                          [0 + 0.j, 0.70710678 + 0.70710678j]])
        result = SGate().power(1 / 2)
        self.assertEqual(result.name, 'unitary')
        assert_allclose(result.to_matrix(), expected)


class TestGateSqrt(QiskitTestCase):
    def test_sqrt(self):
        for degree in range(2, 10):
            result = SGate().power(1 / degree)
            self.assertEqual(result.name, 'unitary')
            assert_allclose(matrix_power(result.to_matrix(), degree), SGate().to_matrix(), rtol=RTOL,
                            atol=ATOL)


if __name__ == '__main__':
    unittest.main()
