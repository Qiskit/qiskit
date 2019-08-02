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

import unittest

import numpy as np

from test.aqua.common import QiskitAquaTestCase
from qiskit.aqua import aqua_globals
from qiskit.aqua.operators import MatrixOperator


class TestMatrixOperator(QiskitAquaTestCase):
    """MatrixOperator tests."""

    def setUp(self):
        super().setUp()
        seed = 0
        np.random.seed(seed)
        aqua_globals.random_seed = seed

        self.num_qubits = 3
        m_size = np.power(2, self.num_qubits)
        matrix = np.random.rand(m_size, m_size)
        self.qubit_op = MatrixOperator(matrix=matrix)

    def test_num_qubits(self):
        op = MatrixOperator(matrix=np.zeros((2, 2)))
        self.assertEqual(op.num_qubits, 0)
        self.assertEqual(self.qubit_op.num_qubits, self.num_qubits)

    def test_is_empty(self):
        op = MatrixOperator(matrix=np.zeros((2, 2)))
        self.assertTrue(op.is_empty())
        self.assertFalse(self.qubit_op.is_empty())


if __name__ == '__main__':
    unittest.main()
