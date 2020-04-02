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

"""Test random operator functions."""

import unittest
from test import combine
from ddt import ddt
import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.quantum_info import Operator, Stinespring, Choi
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info.random import random_hermitian
from qiskit.quantum_info.random import random_quantum_channel
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix


@ddt
class TestRandomUnitary(QiskitTestCase):
    """Testing random_unitary function."""

    @combine(dims=[(2,), (3,), (2, 2), (2, 3)])
    def test_tuple_dims(self, dims):
        """Test random unitary is valid with dims {dims}."""
        value = random_unitary(dims)
        self.assertIsInstance(value, Operator)
        self.assertTrue(value.is_unitary())
        self.assertEqual(value.input_dims(), dims)
        self.assertEqual(value.output_dims(), dims)

    @combine(dim=[2, 3, 4, 5])
    def test_int_dims(self, dim):
        """Test random unitary is valid with dims {dim}."""
        value = random_unitary(dim)
        self.assertIsInstance(value, Operator)
        self.assertTrue(value.is_unitary())
        self.assertEqual(np.product(value.input_dims()), dim)
        self.assertEqual(np.product(value.output_dims()), dim)

    def test_fixed_seed(self):
        """Test fixing seed fixes output"""
        seed = 1532
        value1 = random_unitary(4, seed=seed)
        value2 = random_unitary(4, seed=seed)
        self.assertEqual(value1, value2)

    def test_not_global_seed(self):
        """Test fixing random_unitary seed is locally scoped."""
        seed = 314159
        test_cases = 100
        random_unitary(2, seed=seed)
        rng_before = np.random.randint(1000, size=test_cases)
        random_unitary(2, seed=seed)
        rng_after = np.random.randint(1000, size=test_cases)
        self.assertFalse(np.all(rng_before == rng_after))


@ddt
class TestRandomHermitian(QiskitTestCase):
    """Testing random_hermitian function."""

    @combine(dims=[(2,), (3,), (2, 2), (2, 3)])
    def test_tuple_dims(self, dims):
        """Test random_hermitian is valid with dims {dims}."""
        value = random_hermitian(dims)
        self.assertIsInstance(value, Operator)
        self.assertTrue(is_hermitian_matrix(value.data))
        self.assertEqual(value.input_dims(), dims)
        self.assertEqual(value.output_dims(), dims)

    @combine(dim=[2, 3, 4, 5])
    def test_int_dims(self, dim):
        """Test random_hermitian is valid with dims {dim}."""
        value = random_hermitian(dim)
        self.assertIsInstance(value, Operator)
        self.assertTrue(is_hermitian_matrix(value.data))
        self.assertEqual(np.product(value.input_dims()), dim)
        self.assertEqual(np.product(value.output_dims()), dim)

    def test_fixed_seed(self):
        """Test fixing seed fixes output"""
        seed = 1532
        value1 = random_hermitian(4, seed=seed)
        value2 = random_hermitian(4, seed=seed)
        self.assertEqual(value1, value2)

    def test_not_global_seed(self):
        """Test fixing random_hermitian seed is locally scoped."""
        seed = 314159
        test_cases = 100
        random_hermitian(2, seed=seed)
        rng_before = np.random.randint(1000, size=test_cases)
        random_hermitian(2, seed=seed)
        rng_after = np.random.randint(1000, size=test_cases)
        self.assertFalse(np.all(rng_before == rng_after))


@ddt
class TestRandomQuantumChannel(QiskitTestCase):
    """Testing random_quantum_channel function."""

    @combine(dims=[(2,), (3,), (2, 2), (2, 3)])
    def test_tuple_dims(self, dims):
        """Test random_quantum_channel is valid with dims {dims}."""
        value = random_quantum_channel(dims)
        self.assertIsInstance(value, Stinespring)
        self.assertTrue(value.is_cptp())
        self.assertEqual(value.input_dims(), dims)
        self.assertEqual(value.output_dims(), dims)

    @combine(dim=[2, 3, 4, 5])
    def test_int_dims(self, dim):
        """Test random_quantum_channel is valid with dims {dim}."""
        value = random_quantum_channel(dim)
        self.assertIsInstance(value, Stinespring)
        self.assertTrue(value.is_cptp())
        self.assertEqual(np.product(value.input_dims()), dim)
        self.assertEqual(np.product(value.output_dims()), dim)

    @combine(rank=[1, 2, 3, 4])
    def test_rank(self, rank):
        """Test random_quantum_channel with fixed rank {rank}"""
        choi = Choi(random_quantum_channel(2, rank=rank))
        # Get number of non-zero eigenvalues
        evals = np.linalg.eigvals(choi.data).round(8)
        value = len(evals[evals > 0])
        self.assertEqual(value, rank)

    def test_fixed_seed(self):
        """Test fixing seed fixes output"""
        seed = 1532
        value1 = random_quantum_channel(4, seed=seed)
        value2 = random_quantum_channel(4, seed=seed)
        self.assertEqual(value1, value2)

    def test_not_global_seed(self):
        """Test fixing random_unitary seed is locally scoped."""
        seed = 314159
        test_cases = 100
        random_unitary(2, seed=seed)
        rng_before = np.random.randint(1000, size=test_cases)
        random_unitary(2, seed=seed)
        rng_after = np.random.randint(1000, size=test_cases)
        self.assertFalse(np.all(rng_before == rng_after))


if __name__ == '__main__':
    unittest.main()
