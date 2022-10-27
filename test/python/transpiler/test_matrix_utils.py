"""Test matrix utils"""

import unittest
import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.matrix_utils import (
    build_random_parity_matrix
)

class TestMatrixUtils(QiskitTestCase):
    """Test matrix utils"""

    def test_build_random_parity_matrix_returns_np_ndarray(self):
        """Test the output type of build_random_parity_matrix"""
        instance = build_random_parity_matrix(9)

        self.assertIsInstance(instance, np.ndarray)

    def test_build_random_parity_matrix_returns_np_ndarray_of_given_dimensions(self):
        """Test the output type of build_random_parity_matrix"""
        n = np.random.randint(3, 21)
        instance = build_random_parity_matrix(n)

        self.assertEqual(instance.shape, (n, n))
