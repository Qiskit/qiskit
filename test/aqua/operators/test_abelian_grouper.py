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

"""Test Abelian Grouper"""

import random
import unittest
from itertools import combinations
from test.aqua import QiskitAquaTestCase

from ddt import ddt, data

from qiskit.aqua import AquaError
from qiskit.aqua.operators import (X, Y, Z, I, Zero, Plus, AbelianGrouper)


@ddt
class TestAbelianGrouper(QiskitAquaTestCase):
    """Abelian Grouper tests."""

    @data('h2_op', 'generic')
    def test_abelian_grouper(self, pauli_op):
        """Abelian grouper test"""
        if pauli_op == 'h2_op':
            paulis = (-1.052373245772859 * I ^ I) + \
                     (0.39793742484318045 * I ^ Z) + \
                     (-0.39793742484318045 * Z ^ I) + \
                     (-0.01128010425623538 * Z ^ Z) + \
                     (0.18093119978423156 * X ^ X)
            num_groups = 2
        else:
            paulis = (I ^ I ^ X ^ X * 0.2) + \
                     (Z ^ Z ^ X ^ X * 0.3) + \
                     (Z ^ Z ^ Z ^ Z * 0.4) + \
                     (X ^ X ^ Z ^ Z * 0.5) + \
                     (X ^ X ^ X ^ X * 0.6) + \
                     (I ^ X ^ X ^ X * 0.7)
            num_groups = 4
        grouped_sum = AbelianGrouper().convert(paulis)
        self.assertEqual(len(grouped_sum.oplist), num_groups)
        for group in grouped_sum:
            for op_1, op_2 in combinations(group, 2):
                self.assertTrue(op_1.commutes(op_2))

    def test_ablian_grouper_no_commute(self):
        """Abelian grouper test when non-PauliOp is given"""
        ops = Zero ^ Plus + X ^ Y
        with self.assertRaises(AquaError):
            _ = AbelianGrouper.group_subops(ops)

    def test_group_subops(self):
        """grouper subroutine test"""
        paulis = (I ^ X) + (2 * X ^ X) + (3 * Z ^ Y)
        grouped_sum = AbelianGrouper.group_subops(paulis)
        with self.subTest('test group subops 1'):
            self.assertEqual(len(grouped_sum), 2)
            self.assertListEqual([str(op.primitive) for op in grouped_sum[0]], ['IX', 'XX'])
            self.assertListEqual([op.coeff for op in grouped_sum[0]], [1, 2])
            self.assertListEqual([str(op.primitive) for op in grouped_sum[1]], ['ZY'])
            self.assertListEqual([op.coeff for op in grouped_sum[1]], [3])

        paulis = X + (2 * Y) + (3 * Z)
        grouped_sum = AbelianGrouper.group_subops(paulis)
        with self.subTest('test group subops 2'):
            self.assertEqual(len(grouped_sum), 3)
            self.assertListEqual([str(op[0].primitive) for op in grouped_sum], ['X', 'Y', 'Z'])
            self.assertListEqual([op[0].coeff for op in grouped_sum], [1, 2, 3])

    def test_abelian_grouper_random(self):
        """Abelian grouper test with random paulis"""
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
