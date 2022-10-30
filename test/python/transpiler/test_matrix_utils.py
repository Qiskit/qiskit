"""Test matrix utils"""

import unittest
import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.matrix_utils import (
    build_random_parity_matrix,
    switch_random_rows,
    add_random_rows,
)


class TestMatrixUtils(QiskitTestCase):
    """Test matrix utils"""

    def setUp(self):
        super().setUp()
        self.seed = 1234
        self.rng = np.random.default_rng(self.seed)
        self.n = np.random.randint(3, 21)
        self.m = self.n * 10
        self.matrix = build_random_parity_matrix(self.seed, self.n, self.m)
        self.id_matrix = np.identity(self.n)

    def test_build_random_parity_matrix_returns_np_ndarray(self):
        """Test the output type of build_random_parity_matrix"""

        self.assertIsInstance(self.matrix, np.ndarray)

    def test_build_random_parity_matrix_returns_np_ndarray_of_given_dimensions(self):
        """Test the output type of build_random_parity_matrix"""

        self.assertEqual(self.matrix.shape, (self.n, self.n))

    def test_build_random_parity_matrix_returns_an_invertible_matrix(self):
        """Test build_random_parity_matrix for correctness"""

        inv_matrix = np.linalg.inv(self.matrix)

        self.assertIsInstance(inv_matrix, np.ndarray)

    def test_build_random_parity_matrix_does_not_return_an_identity_matrix_when_row_operations_are_executed(
        self,
    ):
        """Test build_random_parity_matrix for correctness"""

        self.assertEqual(np.array_equal(self.matrix, self.id_matrix), False)

    def test_switch_random_rows_returns_np_nd_array(self):
        """Test the output type of switch_random_rows"""
        instance = switch_random_rows(self.id_matrix, self.rng)

        self.assertIsInstance(instance, np.ndarray)

    def test_switch_random_rows_changes_array(self):
        """Test switch_random_rows for correctness"""
        instance = switch_random_rows(self.id_matrix.copy(), self.rng)

        self.assertEqual(np.array_equal(instance, self.id_matrix), False)

    def test_add_random_rows_returns_np_nd_array(self):
        """Test the output type of add_random_rows"""
        instance = add_random_rows(self.id_matrix, self.rng)

        self.assertIsInstance(instance, np.ndarray)

    def test_add_random_rows_changes_array(self):
        """Test add_random_rows for correctness"""
        instance = add_random_rows(self.id_matrix.copy(), self.rng)

        self.assertEqual(np.array_equal(instance, self.id_matrix), False)

    def test_add_random_rows_will_perform_only_binary_additions(self):
        """Test add_random_rows for correctness"""
        instance = add_random_rows(np.identity(2), self.rng)
        instance = add_random_rows(instance, self.rng)

        self.assertEqual(np.all(instance <= 1), True)
