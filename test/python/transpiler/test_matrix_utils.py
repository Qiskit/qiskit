"""Test matrix utils"""

import unittest
import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.matrix_utils import build_random_parity_matrix, switch_random_rows


class TestMatrixUtils(QiskitTestCase):
    """Test matrix utils"""

    def test_build_random_parity_matrix_returns_np_ndarray(self):
        """Test the output type of build_random_parity_matrix"""
        n = np.random.randint(3, 21)
        instance = build_random_parity_matrix(n)

        self.assertIsInstance(instance, np.ndarray)

    def test_build_random_parity_matrix_returns_np_ndarray_of_given_dimensions(self):
        """Test the output type of build_random_parity_matrix"""
        n = np.random.randint(3, 21)
        instance = build_random_parity_matrix(n)

        self.assertEqual(instance.shape, (n, n))

    def test_build_random_parity_matrix_returns_an_invertible_matrix(self):
        """Test build_random_parity_matrix for correctness"""
        n = np.random.randint(3, 21)
        matrix = build_random_parity_matrix(n)
        instance = np.linalg.inv(matrix)

        self.assertIsInstance(matrix, np.ndarray)

    def test_build_random_parity_matrix_does_not_return_an_identity_matrix(self):
        """Test build_random_parity_matrix for correctness"""
        n = np.random.randint(3, 21)
        instance = build_random_parity_matrix(n)
        identity = np.identity(n)

        self.assertEqual(np.array_equal(instance, identity), False)

    def test_switch_random_rows_returns_np_nd_array(self):
        """Test the output type of switch_random_rows"""
        n = np.random.randint(3, 21)
        identity = np.identity(n)
        instance = switch_random_rows(identity)

        self.assertIsInstance(instance, np.ndarray)

    def test_switch_random_rows_returns_np_nd_array(self):
        """Test switch_random_rows for correctness"""
        n = np.random.randint(3, 21)
        instance = switch_random_rows(np.identity(n))
        identity = np.identity(n)

        self.assertEqual(np.array_equal(instance, identity), False)
