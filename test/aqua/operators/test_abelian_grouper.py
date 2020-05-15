# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test Abelian Grouper """

import random
import unittest
from itertools import combinations
from test.aqua import QiskitAquaTestCase

from qiskit.aqua.operators import (X, Y, Z, I, AbelianGrouper)


class TestAbelianGrouper(QiskitAquaTestCase):
    """Abelian Grouper tests."""

    def test_abelian_grouper(self):
        """ abelian grouper test """
        paulis = (-1.052373245772859 * I ^ I) + \
                 (0.39793742484318045 * I ^ Z) + \
                 (-0.39793742484318045 * Z ^ I) + \
                 (-0.01128010425623538 * Z ^ Z) + \
                 (0.18093119978423156 * X ^ X)
        grouped_sum = AbelianGrouper().convert(paulis)
        self.assertEqual(len(grouped_sum.oplist), 2)
        for group in grouped_sum:
            for op_1, op_2 in combinations(group, 2):
                self.assertTrue(op_1.commutes(op_2))

    def test_abelian_grouper2(self):
        """ abelian grouper test 2 """
        paulis = (I ^ I ^ X ^ X * 0.2) + \
                 (Z ^ Z ^ X ^ X * 0.3) + \
                 (Z ^ Z ^ Z ^ Z * 0.4) + \
                 (X ^ X ^ Z ^ Z * 0.5) + \
                 (X ^ X ^ X ^ X * 0.6) + \
                 (I ^ X ^ X ^ X * 0.7)
        grouped_sum = AbelianGrouper().convert(paulis)
        self.assertEqual(len(grouped_sum.oplist), 4)
        for group in grouped_sum:
            for op_1, op_2 in combinations(group, 2):
                self.assertTrue(op_1.commutes(op_2))

    def test_abelian_grouper3(self):
        """ abelian grouper test 3 """
        paulis = (I ^ X) + (2 * X ^ X) + (3 * Z ^ Y)
        for fast in [True, False]:
            for use_nx in [True, False]:
                grouped_sum = AbelianGrouper.group_subops(paulis, fast=fast, use_nx=use_nx)
                self.assertEqual(len(grouped_sum), 2)
                self.assertEqual(len(grouped_sum[0]), 2)
                self.assertEqual(str(grouped_sum[0][0].primitive), 'IX')
                self.assertEqual(grouped_sum[0][0].coeff, 1)
                self.assertEqual(str(grouped_sum[0][1].primitive), 'XX')
                self.assertEqual(grouped_sum[0][1].coeff, 2)
                self.assertEqual(len(grouped_sum[1]), 1)
                self.assertEqual(str(grouped_sum[1][0].primitive), 'ZY')
                self.assertEqual(grouped_sum[1][0].coeff, 3)

    def test_abelian_grouper4(self):
        """ abelian grouper test 4 """
        paulis = X + (2 * Y) + (3 * Z)
        grouped_sum = AbelianGrouper.group_subops(paulis)
        self.assertEqual(len(grouped_sum), 3)
        self.assertEqual(str(grouped_sum[0][0].primitive), 'X')
        self.assertEqual(grouped_sum[0][0].coeff, 1)
        self.assertEqual(str(grouped_sum[1][0].primitive), 'Y')
        self.assertEqual(grouped_sum[1][0].coeff, 2)
        self.assertEqual(str(grouped_sum[2][0].primitive), 'Z')
        self.assertEqual(grouped_sum[2][0].coeff, 3)

    def test_abelian_grouper_random(self):
        """ abelian grouper test with random paulis """
        random.seed(1234)
        k = 10  # size of pauli operators
        n = 100  # number of pauli operators
        num_tests = 20  # number of tests
        for _ in range(num_tests):
            paulis = []
            for _ in range(n):
                pauliop = 1
                for eachop in random.choices([I] * 5 + [X, Y, Z], k=k):
                    pauliop ^= eachop
                paulis.append(pauliop)
            grouped_sum = AbelianGrouper().convert(sum(paulis))
            for group in grouped_sum:
                for op_1, op_2 in combinations(group, 2):
                    self.assertTrue(op_1.commutes(op_2))


if __name__ == '__main__':
    unittest.main()
