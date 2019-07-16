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
import itertools

import numpy as np
from qiskit.quantum_info import Pauli

from test.aqua.common import QiskitAquaTestCase
from qiskit.aqua import aqua_globals
from qiskit.aqua.operators import WeightedPauliOperator, TPBGroupedWeightedPauliOperator


class TestTPBGroupedWeightedPauliOperator(QiskitAquaTestCase):
    """TPBGroupedWeightedPauliOperator tests."""

    def setUp(self):
        super().setUp()
        np.random.seed(0)
        aqua_globals.random_seed = 0

    def test_sorted_grouping(self):
        """Test with color grouping approach."""
        num_qubits = 2
        paulis = [Pauli.from_label(pauli_label) for pauli_label in itertools.product('IXYZ', repeat=num_qubits)]
        weights = np.random.random(len(paulis))
        op = WeightedPauliOperator.from_list(paulis, weights)
        grouped_op = op.to_grouped_weighted_pauli_operator(TPBGroupedWeightedPauliOperator.sorted_grouping)

        # check all paulis are still existed.
        for gp in grouped_op.paulis:
            passed = False
            for p in op.paulis:
                if p[1] == gp[1]:
                    passed = p[0] == gp[0]
                    break
            self.assertTrue(passed, "non-existed paulis in grouped_paulis: {}".format(gp[1].to_label()))

        # check the number of basis of grouped one should be less than and equal to the original one.
        self.assertGreaterEqual(len(op.basis), len(grouped_op.basis))

    def test_unsorted_grouping(self):
        """Test with normal grouping approach."""

        num_qubits = 4
        paulis = [Pauli.from_label(pauli_label) for pauli_label in itertools.product('IXYZ', repeat=num_qubits)]
        weights = np.random.random(len(paulis))
        op = WeightedPauliOperator.from_list(paulis, weights)
        grouped_op = op.to_grouped_weighted_pauli_operator(TPBGroupedWeightedPauliOperator.unsorted_grouping)

        for gp in grouped_op.paulis:
            passed = False
            for p in op.paulis:
                if p[1] == gp[1]:
                    passed = p[0] == gp[0]
                    break
            self.assertTrue(passed, "non-existed paulis in grouped_paulis: {}".format(gp[1].to_label()))

        self.assertGreaterEqual(len(op.basis), len(grouped_op.basis))

    def test_chop(self):
        paulis = [Pauli.from_label(x) for x in ['IIXX', 'ZZXX', 'ZZZZ', 'XXZZ', 'XXXX', 'IXXX']]
        coeffs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        op = WeightedPauliOperator.from_list(paulis, coeffs)
        grouped_op = op.to_grouped_weighted_pauli_operator(TPBGroupedWeightedPauliOperator.sorted_grouping)

        original_num_basis = len(grouped_op.basis)
        chopped_grouped_op = grouped_op.chop(0.35, copy=True)
        self.assertLessEqual(len(chopped_grouped_op.basis), 3)
        self.assertLessEqual(len(chopped_grouped_op.basis), original_num_basis)
        # ZZXX group is remove
        for b, _ in chopped_grouped_op.basis:
            self.assertFalse(b.to_label() == 'ZZXX')

        chopped_grouped_op = grouped_op.chop(0.55, copy=True)
        self.assertLessEqual(len(chopped_grouped_op.basis), 1)
        self.assertLessEqual(len(chopped_grouped_op.basis), original_num_basis)

        for b, _ in chopped_grouped_op.basis:
            self.assertFalse(b.to_label() == 'ZZXX')
            self.assertFalse(b.to_label() == 'ZZZZ')
            self.assertFalse(b.to_label() == 'XXZZ')


if __name__ == '__main__':
    unittest.main()
