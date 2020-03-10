# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test library of quantum circuits."""

from unittest import mock

from qiskit.test import QiskitTestCase
from qiskit.circuit import bit
from qiskit.circuit import quantumregister
from qiskit.circuit import classicalregister


class TestBitClass(QiskitTestCase):
    """Test library of boolean logic quantum circuits."""

    def test_bit_hash_update_reg(self):
        orig_reg = mock.MagicMock()
        orig_reg.size = 3
        new_reg = mock.MagicMock()
        orig_reg.size = 4
        test_bit = bit.Bit(orig_reg, 0)
        orig_hash = hash(test_bit)
        test_bit.register = new_reg
        new_hash = hash(test_bit)
        self.assertNotEqual(orig_hash, new_hash)

    def test_bit_hash_update_index(self):
        orig_reg = mock.MagicMock()
        orig_reg.size = 4
        test_bit = bit.Bit(orig_reg, 0)
        orig_hash = hash(test_bit)
        test_bit.index = 2
        new_hash = hash(test_bit)
        self.assertNotEqual(orig_hash, new_hash)

    def test_qubit_hash_update_reg(self):
        orig_reg = mock.MagicMock(spec=quantumregister.QuantumRegister)
        orig_reg.size = 3
        new_reg = mock.MagicMock(spec=quantumregister.QuantumRegister)
        new_reg.size = 6
        test_bit = quantumregister.Qubit(orig_reg, 0)
        orig_hash = hash(test_bit)
        test_bit.register = new_reg
        new_hash = hash(test_bit)
        self.assertNotEqual(orig_hash, new_hash)

    def test_qubit_hash_update_index(self):
        orig_reg = mock.MagicMock(spec=quantumregister.QuantumRegister)
        orig_reg.size = 67
        test_bit = quantumregister.Qubit(orig_reg, 0)
        orig_hash = hash(test_bit)
        test_bit.index = 2
        new_hash = hash(test_bit)
        self.assertNotEqual(orig_hash, new_hash)

    def test_clbit_hash_update_reg(self):
        orig_reg = mock.MagicMock(spec=classicalregister.ClassicalRegister)
        orig_reg.size = 5
        new_reg = mock.MagicMock(spec=classicalregister.ClassicalRegister)
        new_reg.size = 53
        test_bit = classicalregister.Clbit(orig_reg, 0)
        orig_hash = hash(test_bit)
        test_bit.register = new_reg
        new_hash = hash(test_bit)
        self.assertNotEqual(orig_hash, new_hash)

    def test_clbit_hash_update_index(self):
        orig_reg = mock.MagicMock(spec=classicalregister.ClassicalRegister)
        orig_reg.size = 42
        test_bit = classicalregister.Clbit(orig_reg, 0)
        orig_hash = hash(test_bit)
        test_bit.index = 2
        new_hash = hash(test_bit)
        self.assertNotEqual(orig_hash, new_hash)
