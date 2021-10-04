# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test NumPy Eigen solver """

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

import numpy as np
from ddt import data, ddt

from qiskit.algorithms import NumPyEigensolver
from qiskit.opflow import PauliSumOp, X, Y, Z


@ddt
class TestNumPyEigensolver(QiskitAlgorithmsTestCase):
    """Test NumPy Eigen solver"""

    def setUp(self):
        super().setUp()
        self.qubit_op = PauliSumOp.from_list(
            [
                ("II", -1.052373245772859),
                ("ZI", 0.39793742484318045),
                ("IZ", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )

    def test_ce(self):
        """Test basics"""
        algo = NumPyEigensolver()
        result = algo.compute_eigenvalues(operator=self.qubit_op, aux_operators=[])
        self.assertEqual(len(result.eigenvalues), 1)
        self.assertEqual(len(result.eigenstates), 1)
        self.assertAlmostEqual(result.eigenvalues[0], -1.85727503 + 0j)

    def test_ce_k4(self):
        """Test for k=4 eigenvalues"""
        algo = NumPyEigensolver(k=4)
        result = algo.compute_eigenvalues(operator=self.qubit_op, aux_operators=[])
        self.assertEqual(len(result.eigenvalues), 4)
        self.assertEqual(len(result.eigenstates), 4)
        np.testing.assert_array_almost_equal(
            result.eigenvalues.real, [-1.85727503, -1.24458455, -0.88272215, -0.22491125]
        )

    def test_ce_k4_filtered(self):
        """Test for k=4 eigenvalues with filter"""

        # define filter criterion
        # pylint: disable=unused-argument
        def criterion(x, v, a_v):
            return v >= -1

        algo = NumPyEigensolver(k=4, filter_criterion=criterion)
        result = algo.compute_eigenvalues(operator=self.qubit_op, aux_operators=[])
        self.assertEqual(len(result.eigenvalues), 2)
        self.assertEqual(len(result.eigenstates), 2)
        np.testing.assert_array_almost_equal(result.eigenvalues.real, [-0.88272215, -0.22491125])

    def test_ce_k4_filtered_empty(self):
        """Test for k=4 eigenvalues with filter always returning False"""

        # define filter criterion
        # pylint: disable=unused-argument
        def criterion(x, v, a_v):
            return False

        algo = NumPyEigensolver(k=4, filter_criterion=criterion)
        result = algo.compute_eigenvalues(operator=self.qubit_op, aux_operators=[])
        self.assertEqual(len(result.eigenvalues), 0)
        self.assertEqual(len(result.eigenstates), 0)

    @data(X, Y, Z)
    def test_ce_k1_1q(self, op):
        """Test for 1 qubit operator"""
        algo = NumPyEigensolver(k=1)
        result = algo.compute_eigenvalues(operator=op)
        np.testing.assert_array_almost_equal(result.eigenvalues, [-1])

    @data(X, Y, Z)
    def test_ce_k2_1q(self, op):
        """Test for 1 qubit operator"""
        algo = NumPyEigensolver(k=2)
        result = algo.compute_eigenvalues(operator=op)
        np.testing.assert_array_almost_equal(result.eigenvalues, [-1, 1])


if __name__ == "__main__":
    unittest.main()
