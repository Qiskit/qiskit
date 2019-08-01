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
from ddt import ddt, data
from numpy import pi, array
from numpy.testing import assert_allclose
from numpy.linalg import matrix_power

from qiskit.transpiler import PassManager
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister
from qiskit.test import QiskitTestCase
from qiskit.extensions import SGate, SdgGate, U3Gate, UnitaryGate, CnotGate
from qiskit.circuit import Instruction, Measure
from qiskit.transpiler.passes import Unroller
from qiskit.exceptions import QiskitError


class TestPowerInt1Q(QiskitTestCase):
    """Test gate_q1.power() with integer"""

    def test_standard_1Q_two(self):
        """Test standard gate.power(2) method.
        """
        qr = QuantumRegister(1, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(SGate(), [qr[0]])
        expected_circ.append(SGate(), [qr[0]])
        expected = expected_circ.to_instruction()

        result = SGate().power(2)

        self.assertEqual(result.name, 's^2')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)

    def test_standard_1Q_one(self):
        """Test standard gate.power(1) method.
        """
        qr = QuantumRegister(1, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(SGate(), [qr[0]])
        expected = expected_circ.to_instruction()

        result = SGate().power(1)

        self.assertEqual(result.name, 's^1')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)

    def test_standard_1Q_zero(self):
        """Test standard gate.power(0) method.
        """
        qr = QuantumRegister(1, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(U3Gate(0, 0, 0), [qr[0]])
        expected = expected_circ.to_instruction()

        result = SGate().power(0)

        self.assertEqual(result.name, 's^0')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)

    def test_standard_1Q_minus_one(self):
        """Test standard gate.power(-1) method.
        """
        qr = QuantumRegister(1, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(SdgGate(), [qr[0]])
        expected = expected_circ.to_instruction()

        result = SGate().power(-1)

        self.assertEqual(result.name, 's^-1')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)

    def test_standard_1Q_minus_two(self):
        """Test standard gate.power(-2) method.
        """
        qr = QuantumRegister(1, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(SdgGate(), [qr[0]])
        expected_circ.append(SdgGate(), [qr[0]])
        expected = expected_circ.to_instruction()

        result = SGate().power(-2)

        self.assertEqual(result.name, 's^-2')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)


class TestPowerInt2Q(QiskitTestCase):
    """Test gate_q2.power() with integer"""

    def test_standard_2Q_two(self):
        """Test standard 2Q gate.power(2) method.
        """
        qr = QuantumRegister(2, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(CnotGate(), [qr[0], qr[1]])
        expected_circ.append(CnotGate(), [qr[0], qr[1]])
        expected = expected_circ.to_instruction()

        result = CnotGate().power(2)

        self.assertEqual(result.name, 'cx^2')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)

    def test_standard_2Q_one(self):
        """Test standard 2Q gate.power(1) method.
        """
        qr = QuantumRegister(2, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(CnotGate(), [qr[0], qr[1]])
        expected = expected_circ.to_instruction()

        result = CnotGate().power(1)

        self.assertEqual(result.name, 'cx^1')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)

    def test_standard_2Q_zero(self):
        """Test standard 2Q gate.power(0) method.
        """
        qr = QuantumRegister(2, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(U3Gate(0, 0, 0), [qr[0]])
        expected_circ.append(U3Gate(0, 0, 0), [qr[1]])
        expected = expected_circ.to_instruction()

        result = CnotGate().power(0)

        self.assertEqual(result.name, 'cx^0')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)

    def test_standard_2Q_minus_one(self):
        """Test standard 2Q gate.power(-1) method.
        """
        qr = QuantumRegister(2, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(CnotGate(), [qr[0], qr[1]])
        expected = expected_circ.to_instruction()

        result = CnotGate().power(-1)

        self.assertEqual(result.name, 'cx^-1')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)

    def test_standard_2Q_minus_two(self):
        """Test standard 2Q gate.power(-2) method.
        """
        qr = QuantumRegister(2, 'qr')
        expected_circ = QuantumCircuit(qr)
        expected_circ.append(CnotGate(), [qr[0], qr[1]])
        expected_circ.append(CnotGate(), [qr[0], qr[1]])
        expected = expected_circ.to_instruction()

        result = CnotGate().power(-2)

        self.assertEqual(result.name, 'cx^-2')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)


class TestPowerIntMeasure(QiskitTestCase):
    """Test Measure.power() with integer"""

    def test_measure_two(self):
        """Test Measure.power(2) method.
        """
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        expected_circ = QuantumCircuit(qr, cr)
        expected_circ.append(Measure(), [qr[0]], [cr[0]])
        expected_circ.append(Measure(), [qr[0]], [cr[0]])
        expected = expected_circ.to_instruction()

        result = Measure().power(2)

        self.assertEqual(result.name, 'measure^2')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)

    def test_measure_one(self):
        """Test Measure.power(1) method.
        """
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        expected_circ = QuantumCircuit(qr, cr)
        expected_circ.append(Measure(), [qr[0]], [cr[0]])
        expected = expected_circ.to_instruction()

        result = Measure().power(1)

        self.assertEqual(result.name, 'measure^1')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)

    def test_measure_zero(self):
        """Test Measure.power(0) method.
        """
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        expected_circ = QuantumCircuit(qr, cr)
        expected_circ.append(U3Gate(0, 0, 0), [qr[0]])
        expected = expected_circ.to_instruction()

        result = Measure().power(0)

        self.assertEqual(result.name, 'measure^0')
        self.assertEqual(result.definition, expected.definition)
        self.assertIsInstance(result, Instruction)

    def test_measure_minus_one(self):
        """Test Measure.power(-1) method.
        """
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        expected_circ = QuantumCircuit(qr, cr)
        expected_circ.append(U3Gate(0, 0, 0), [qr[0]])
        expected = expected_circ.to_instruction()

        with self.assertRaises(QiskitError) as cm:
            _ = Measure().power(-1)
        self.assertIn('inverse', str(cm.exception))


class TestPowerUnroller(QiskitTestCase):
    """Test unrolling Gate.power"""

    def test_unroller_two(self):
        """Test unrolling gate.power(2).
        """
        qr = QuantumRegister(1, 'qr')

        circuit = QuantumCircuit(qr)
        circuit.append(SGate().power(2), [qr[0]])
        result = PassManager(Unroller('u3')).run(circuit)

        expected = QuantumCircuit(qr)
        expected.append(U3Gate(0, 0, pi / 2), [qr[0]])
        expected.append(U3Gate(0, 0, pi / 2), [qr[0]])

        self.assertEqual(result, expected)

    def test_unroller_one(self):
        """Test unrolling gate.power(1).
        """
        qr = QuantumRegister(1, 'qr')

        circuit = QuantumCircuit(qr)
        circuit.append(SGate().power(1), [qr[0]])
        result = PassManager(Unroller('u3')).run(circuit)

        expected = QuantumCircuit(qr)
        expected.append(U3Gate(0, 0, pi / 2), [qr[0]])

        self.assertEqual(result, expected)

    def test_unroller_zero(self):
        """Test unrolling gate.power(0).
        """
        qr = QuantumRegister(1, 'qr')

        circuit = QuantumCircuit(qr)
        circuit.append(SGate().power(0), [qr[0]])
        result = PassManager(Unroller('u3')).run(circuit)

        expected = QuantumCircuit(qr)
        expected.append(U3Gate(0, 0, 0), [qr[0]])

        self.assertEqual(result, expected)

    def test_unroller_minus_one(self):
        """Test unrolling gate.power(-1).
        """
        qr = QuantumRegister(1, 'qr')

        circuit = QuantumCircuit(qr)
        circuit.append(SGate().power(-1), [qr[0]])
        result = PassManager(Unroller('u3')).run(circuit)

        expected = QuantumCircuit(qr)
        expected.append(U3Gate(0, 0, -pi / 2), [qr[0]])

        self.assertEqual(result, expected)

    def test_unroller_minus_two(self):
        """Test unrolling gate.power(-2).
        """
        qr = QuantumRegister(1, 'qr')

        circuit = QuantumCircuit(qr)
        circuit.append(SGate().power(-2), [qr[0]])
        result = PassManager(Unroller('u3')).run(circuit)

        expected = QuantumCircuit(qr)
        expected.append(U3Gate(0, 0, -pi / 2), [qr[0]])
        expected.append(U3Gate(0, 0, -pi / 2), [qr[0]])

        self.assertEqual(result, expected)


class TestGateSqrt(QiskitTestCase):
    """Test square root using Gate.power()"""

    def test_unitary_sqrt(self):
        """Test UnitaryGate.power(1/2) method.
        """
        expected = array([[0.5 + 0.5j, 0.5 + 0.5j],
                          [-0.5 - 0.5j, 0.5 + 0.5j]])

        result = UnitaryGate([[0, 1j], [-1j, 0]]).power(1 / 2)

        self.assertEqual(result.name, 'unitary^0.5')
        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Instruction)
        assert_allclose(result.definition[0][0].to_matrix(), expected)

    def test_starndard_sqrt(self):
        """Test standard Gate.power(1/2) method.
        """
        expected = array([[1 + 0.j, 0 + 0.j],
                          [0 + 0.j, 0.70710678 + 0.70710678j]])

        result = SGate().power(1 / 2)

        self.assertEqual(result.name, 's^0.5')
        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Instruction)
        assert_allclose(result.definition[0][0].to_matrix(), expected)


@ddt
class TestGateFloat(QiskitTestCase):
    """Test power generalization to root calculation"""

    @data(2, 3, 4, 5, 6, 7, 8, 9)
    def test_direct_root(self, degree):
        """Test nth root"""
        result = SGate().power(1 / degree)

        self.assertEqual(result.name, 's^' + str(1 / degree))
        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Instruction)
        assert_allclose(matrix_power(result.definition[0][0].to_matrix(), degree),
                        SGate().to_matrix())

    @data(2.1, 3.2, 4.3, 5.4, 6.5, 7.6, 8.7, 9.8)
    def test_float_gt_one(self, exponent):
        """Test greater-than-one exponents """
        result = SGate().power(exponent)

        self.assertEqual(result.name, 's^' + str(exponent))
        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Instruction)
        assert_allclose(SGate().to_matrix() ** exponent, result.definition[0][0].to_matrix())

    def test_zero_two(self, exponent=0.2):
        """Test Sgate^(0.2)"""
        result = SGate().power(exponent)

        self.assertEqual(result.name, 's^' + str(exponent))
        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Instruction)
        assert_allclose(array([[1. + 0.j, 0. + 0.j],
                               [0. + 0.j, 0.95105652 + 0.30901699j]], dtype=complex),
                        result.definition[0][0].to_matrix(), rtol=1e-07, atol=1e-8)

    def test_minus_zero_two(self, exponent=-0.2):
        """Test Sgate^(-0.2)"""
        result = SGate().power(exponent)

        self.assertEqual(result.name, 's^' + str(exponent))
        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Instruction)
        assert_allclose(array([[1. + 0.j, 0. + 0.j],
                               [0. + 0.j, 0.95105652 - 0.30901699j]], dtype=complex),
                        result.definition[0][0].to_matrix(), rtol=1e-07, atol=1e-8)


if __name__ == '__main__':
    unittest.main()
