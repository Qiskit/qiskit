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

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.extensions.standard import SGate, SdgGate


class TestGatePower(QiskitTestCase):
    """Test Gate.power"""

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


if __name__ == '__main__':
    unittest.main()
