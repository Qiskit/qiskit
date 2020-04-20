# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Randomized tests of Clifford operator class."""
import unittest
from hypothesis import given, strategies

from qiskit.quantum_info.random import random_clifford
from qiskit.quantum_info import Clifford


class TestClifford(unittest.TestCase):
    """Test random_clifford"""

    def assertValidClifford(self, value, num_qubits):
        """Assertion from test/python/quantum_info/operators/test_random.py:
        TestRandomClifford:test_valid"""
        with self.subTest(msg='Test type'):
            self.assertIsInstance(value, Clifford)
        with self.subTest(msg='Test num_qubits'):
            self.assertEqual(value.num_qubits, num_qubits)

    @given(strategies.integers(min_value=0, max_value=2 ** 32 - 1))
    def test_1q_random_clifford_valid(self, seed):
        """Test random_clifford 1-qubits."""
        value = random_clifford(1, seed=seed)
        self.assertValidClifford(value, num_qubits=1)

    @given(strategies.integers(min_value=0, max_value=2 ** 32 - 1))
    def test_2q_random_clifford_valid(self, seed):
        """Test random_clifford 2-qubits."""
        value = random_clifford(2, seed=seed)
        self.assertValidClifford(value, num_qubits=2)

    @given(strategies.integers(min_value=0, max_value=2 ** 32 - 1))
    def test_3q_random_clifford_valid(self, seed):
        """Test random_clifford 3-qubits."""
        value = random_clifford(3, seed=seed)
        self.assertValidClifford(value, num_qubits=3)

    @given(strategies.integers(min_value=0, max_value=2 ** 32 - 1))
    def test_4q_random_clifford_valid(self, seed):
        """Test random_clifford 4-qubits."""
        value = random_clifford(4, seed=seed)
        self.assertValidClifford(value, num_qubits=4)

    @given(strategies.integers(min_value=0, max_value=2 ** 32 - 1))
    def test_5q_random_clifford_valid(self, seed):
        """Test random_clifford 5-qubits."""
        value = random_clifford(5, seed=seed)
        self.assertValidClifford(value, num_qubits=5)

    @given(strategies.integers(min_value=0, max_value=2 ** 32 - 1))
    def test_10q_random_clifford_valid(self, seed):
        """Test random_clifford 10-qubits."""
        value = random_clifford(10, seed=seed)
        self.assertValidClifford(value, num_qubits=10)

    @given(strategies.integers(min_value=0, max_value=2 ** 32 - 1))
    def test_50q_random_clifford_valid(self, seed):
        """Test random_clifford 50-qubits."""
        value = random_clifford(50, seed=seed)
        self.assertValidClifford(value, num_qubits=50)

    @given(strategies.integers(min_value=0, max_value=2 ** 32 - 1))
    def test_100q_random_clifford_valid(self, seed):
        """Test random_clifford 100-qubits."""
        value = random_clifford(100, seed=seed)
        self.assertValidClifford(value, num_qubits=100)

    @given(strategies.integers(min_value=0, max_value=2 ** 32 - 1))
    def test_150q_random_clifford_valid(self, seed):
        """Test random_clifford 150-qubits."""
        value = random_clifford(150, seed=seed)
        self.assertValidClifford(value, num_qubits=150)

    @given(strategies.integers(min_value=0, max_value=2 ** 32 - 1))
    def test_211q_random_clifford_valid(self, seed):
        """Test random_clifford 211-qubits."""
        value = random_clifford(211, seed=seed)
        self.assertValidClifford(value, num_qubits=211)


if __name__ == '__main__':
    unittest.main()
