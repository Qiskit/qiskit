# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Op Converter """

import unittest
import itertools
from test.aqua import QiskitAquaTestCase

import numpy as np
from qiskit.quantum_info import Pauli

from qiskit.aqua import aqua_globals
from qiskit.aqua.operators import WeightedPauliOperator, MatrixOperator
from qiskit.aqua.operators.legacy import op_converter


class TestOpConverter(QiskitAquaTestCase):
    """OpConverter tests."""

    def setUp(self):
        super().setUp()
        seed = 0
        aqua_globals.random_seed = seed

        self.num_qubits = 2
        m_size = np.power(2, self.num_qubits)
        matrix = aqua_globals.random.random((m_size, m_size))
        self.mat_op = MatrixOperator(matrix=matrix)
        paulis = [Pauli.from_label(pauli_label)
                  for pauli_label in itertools.product('IXYZ', repeat=self.num_qubits)]
        weights = aqua_globals.random.random(len(paulis))
        self.pauli_op = WeightedPauliOperator.from_list(paulis, weights)

    def test_to_weighted_pauli_operator(self):
        """ to weighted pauli operator """
        mat_op = op_converter.to_matrix_operator(self.pauli_op)
        pauli_op = op_converter.to_weighted_pauli_operator(mat_op)
        pauli_op.rounding(8)
        self.pauli_op.rounding(8)
        self.assertEqual(pauli_op, self.pauli_op)

    def test_to_matrix_operator(self):
        """ to matrix operator """
        pauli_op = op_converter.to_weighted_pauli_operator(self.mat_op)
        mat_op = op_converter.to_matrix_operator(pauli_op)
        diff = float(np.sum(np.abs(self.mat_op.matrix - mat_op.matrix)))
        self.assertAlmostEqual(0, diff, places=8)


if __name__ == '__main__':
    unittest.main()
