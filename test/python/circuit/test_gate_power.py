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
from numpy import array, eye
from numpy.testing import assert_allclose, assert_array_almost_equal
from numpy.linalg import matrix_power

from qiskit.transpiler import PassManager
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.extensions import SGate, UnitaryGate, CnotGate
from qiskit.circuit import Gate
from qiskit.transpiler.passes import Unroller, Optimize1qGates
from qiskit.quantum_info.operators import Operator


@ddt
class TestPowerSgate(QiskitTestCase):
    """Test Sgate.power() with integer"""

    @data(2, 1, 0, -1, -2)
    def test_sgate_int(self, n):
        """Test Sgate.power(n) method with n as integer.
        """
        result = SGate().power(n)

        self.assertEqual(result.label, 's^%s' % n)
        self.assertIsInstance(result, UnitaryGate)
        assert_array_almost_equal(result.to_matrix(), matrix_power(SGate().to_matrix(), n))

    def test_sgate_float_1_5(self, n=1.5):
        """Test Sgate.power(1.5) method.
        """
        result = SGate().power(n)

        expected = array([[1, 0],
                          [0, -0.70710678 + 0.70710678j]], dtype=complex)
        self.assertEqual(result.label, 's^%s' % n)
        self.assertIsInstance(result, UnitaryGate)
        assert_array_almost_equal(result.to_matrix(), expected)

    def test_sgate_float_0_1(self, n=0.1):
        """Test Sgate.power(0.1) method.
        """
        result = SGate().power(n)

        expected = array([[1, 0],
                          [0, 0.98768834 + 0.15643447j]], dtype=complex)
        self.assertEqual(result.label, 's^%s' % n)
        self.assertIsInstance(result, UnitaryGate)
        assert_array_almost_equal(result.to_matrix(), expected)

    def test_sgate_float_minus_1_5(self, n=-1.5):
        """Test Sgate.power(-1.5) method.
        """
        result = SGate().power(n)

        expected = array([[1, 0],
                          [0, -0.70710678 - 0.70710678j]], dtype=complex)
        self.assertEqual(result.label, 's^%s' % n)
        self.assertIsInstance(result, UnitaryGate)
        assert_array_almost_equal(result.to_matrix(), expected)

    def test_sgate_float_minus_0_1(self, n=-0.1):
        """Test Sgate.power(-0.1) method.
        """
        result = SGate().power(n)

        expected = array([[1, 0],
                          [0, 0.98768834 - 0.15643447j]], dtype=complex)
        self.assertEqual(result.label, 's^%s' % n)
        self.assertIsInstance(result, UnitaryGate)
        assert_array_almost_equal(result.to_matrix(), expected)


class TestPowerIntCX(QiskitTestCase):
    """Test CX.power() with integer"""

    def setUp(self) -> None:
        self.identity = array([[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [0, 0, 0, 1]], dtype=complex)

    def test_cx_two(self):
        """Test CX.power(2) method.
        """
        result = CnotGate().power(2)

        self.assertEqual(result.label, 'cx^2')
        self.assertIsInstance(result, UnitaryGate)
        assert_array_almost_equal(result.to_matrix(), self.identity)

    def test_cx_one(self):
        """Test CX.power(1) method.
        """
        result = CnotGate().power(1)

        self.assertEqual(result.label, 'cx^1')
        self.assertIsInstance(result, UnitaryGate)
        assert_array_almost_equal(result.to_matrix(), CnotGate().to_matrix())

    def test_cx_zero(self):
        """Test CX.power(0) method.
        """
        result = CnotGate().power(0)

        self.assertEqual(result.label, 'cx^0')
        self.assertIsInstance(result, UnitaryGate)
        assert_array_almost_equal(result.to_matrix(), self.identity)

    def test_cx_minus_one(self):
        """Test CX.power(-1) method.
        """
        result = CnotGate().power(-1)

        self.assertEqual(result.label, 'cx^-1')
        self.assertIsInstance(result, UnitaryGate)
        assert_array_almost_equal(result.to_matrix(), CnotGate().to_matrix())

    def test_cx_minus_two(self):
        """Test CX.power(-2) method.
        """
        result = CnotGate().power(-2)

        self.assertEqual(result.label, 'cx^-2')
        self.assertIsInstance(result, UnitaryGate)
        assert_array_almost_equal(result.to_matrix(), self.identity)


class TestGateSqrt(QiskitTestCase):
    """Test square root using Gate.power()"""

    def test_unitary_sqrt(self):
        """Test UnitaryGate.power(1/2) method.
        """
        expected = array([[0.70710678118, 0.70710678118],
                          [-0.70710678118, 0.70710678118]], dtype=complex)

        result = UnitaryGate([[0, 1j], [-1j, 0]]).power(1 / 2)

        self.assertEqual(result.label, 'unitary^0.5')
        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Gate)
        assert_allclose(result.definition[0][0].to_matrix(), expected)

    def test_starndard_sqrt(self):
        """Test standard Gate.power(1/2) method.
        """
        expected = array([[1, 0],
                          [0, 0.70710678118 + 0.70710678118j]], dtype=complex)

        result = SGate().power(1 / 2)

        self.assertEqual(result.label, 's^0.5')
        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Gate)
        assert_allclose(result.definition[0][0].to_matrix(), expected)


@ddt
class TestGateFloat(QiskitTestCase):
    """Test power generalization to root calculation"""

    @data(2, 3, 4, 5, 6, 7, 8, 9)
    def test_direct_root(self, degree):
        """Test nth root"""
        result = SGate().power(1 / degree)

        self.assertEqual(result.label, 's^' + str(1 / degree))
        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Gate)
        assert_allclose(matrix_power(result.definition[0][0].to_matrix(), degree),
                        SGate().to_matrix())

    @data(2.1, 3.2, 4.3, 5.4, 6.5, 7.6, 8.7, 9.8)
    def test_float_gt_one(self, exponent):
        """Test greater-than-one exponents """
        result = SGate().power(exponent)

        self.assertEqual(result.label, 's^' + str(exponent))
        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Gate)
        # SGate().to_matrix() is diagonal so `**` is equivalent.
        assert_allclose(SGate().to_matrix() ** exponent, result.definition[0][0].to_matrix())

    def test_zero_two(self, exponent=0.2):
        """Test Sgate^(0.2)"""
        result = SGate().power(exponent)

        self.assertEqual(result.label, 's^' + str(exponent))
        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Gate)
        assert_allclose(array([[1, 0],
                               [0, 0.95105652 + 0.30901699j]], dtype=complex),
                        result.definition[0][0].to_matrix())

    def test_minus_zero_two(self, exponent=-0.2):
        """Test Sgate^(-0.2)"""
        result = SGate().power(exponent)

        self.assertEqual(result.label, 's^' + str(exponent))
        self.assertEqual(len(result.definition), 1)
        self.assertIsInstance(result, Gate)
        assert_allclose(array([[1, 0],
                               [0, 0.95105652 - 0.30901699j]], dtype=complex),
                        result.definition[0][0].to_matrix())


@ddt
class TestPowerInvariant(QiskitTestCase):
    """Test Gate.power invariants"""

    @data(-3, -2, -1, 1, 2, 3)
    def test_invariant1_int(self, n):
        """Test (op^(1/n))^(n) == op, integer n
        """
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.append(SGate().power(1 / n).power(n), [qr[0]])
        result = PassManager([Unroller('u3'), Optimize1qGates()]).run(circuit)

        expected_circuit = QuantumCircuit(qr)
        expected_circuit.append(SGate(), [qr[0]])
        expected = PassManager([Unroller('u3'), Optimize1qGates()]).run(expected_circuit)

        self.assertEqual(result, expected)

    @data(-3, -2, -1, 1, 2, 3)
    def test_invariant2(self, n):
        """Test op^(n) * op^(-n) == I
        """
        result = Operator(SGate().power(n)) @ Operator(SGate().power(-n))
        expected = Operator(eye(2))

        self.assertEqual(len(result.data), len(expected.data))
        assert_array_almost_equal(result.data, expected.data)


if __name__ == '__main__':
    unittest.main()
