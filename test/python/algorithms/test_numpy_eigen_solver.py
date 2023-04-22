# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test NumPy Eigen solver"""

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
        with self.assertWarns(DeprecationWarning):
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
        with self.assertWarns(DeprecationWarning):
            algo = NumPyEigensolver()
            result = algo.compute_eigenvalues(operator=self.qubit_op, aux_operators=[])

        self.assertEqual(len(result.eigenvalues), 1)
        self.assertEqual(len(result.eigenstates), 1)
        self.assertEqual(result.eigenvalues.dtype, np.float64)
        self.assertAlmostEqual(result.eigenvalues[0], -1.85727503)

    def test_ce_k4(self):
        """Test for k=4 eigenvalues"""
        with self.assertWarns(DeprecationWarning):
            algo = NumPyEigensolver(k=4)
            result = algo.compute_eigenvalues(operator=self.qubit_op, aux_operators=[])

        self.assertEqual(len(result.eigenvalues), 4)
        self.assertEqual(len(result.eigenstates), 4)
        self.assertEqual(result.eigenvalues.dtype, np.float64)
        np.testing.assert_array_almost_equal(
            result.eigenvalues, [-1.85727503, -1.24458455, -0.88272215, -0.22491125]
        )

    def test_ce_k4_filtered(self):
        """Test for k=4 eigenvalues with filter"""

        # define filter criterion
        # pylint: disable=unused-argument
        def criterion(x, v, a_v):
            return v >= -1

        with self.assertWarns(DeprecationWarning):
            algo = NumPyEigensolver(k=4, filter_criterion=criterion)
            result = algo.compute_eigenvalues(operator=self.qubit_op, aux_operators=[])

        self.assertEqual(len(result.eigenvalues), 2)
        self.assertEqual(len(result.eigenstates), 2)
        self.assertEqual(result.eigenvalues.dtype, np.float64)
        np.testing.assert_array_almost_equal(result.eigenvalues, [-0.88272215, -0.22491125])

    def test_ce_k4_filtered_empty(self):
        """Test for k=4 eigenvalues with filter always returning False"""

        # define filter criterion
        # pylint: disable=unused-argument
        def criterion(x, v, a_v):
            return False

        with self.assertWarns(DeprecationWarning):
            algo = NumPyEigensolver(k=4, filter_criterion=criterion)
            result = algo.compute_eigenvalues(operator=self.qubit_op, aux_operators=[])
        self.assertEqual(len(result.eigenvalues), 0)
        self.assertEqual(len(result.eigenstates), 0)

    @data(X, Y, Z)
    def test_ce_k1_1q(self, op):
        """Test for 1 qubit operator"""

        with self.assertWarns(DeprecationWarning):
            algo = NumPyEigensolver(k=1)
            result = algo.compute_eigenvalues(operator=op)
        np.testing.assert_array_almost_equal(result.eigenvalues, [-1])

    @data(X, Y, Z)
    def test_ce_k2_1q(self, op):
        """Test for 1 qubit operator"""

        with self.assertWarns(DeprecationWarning):
            algo = NumPyEigensolver(k=2)
            result = algo.compute_eigenvalues(operator=op)
        np.testing.assert_array_almost_equal(result.eigenvalues, [-1, 1])

    def test_aux_operators_list(self):
        """Test list-based aux_operators."""

        with self.assertWarns(DeprecationWarning):
            aux_op1 = PauliSumOp.from_list([("II", 2.0)])
            aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
        aux_ops = [aux_op1, aux_op2]

        with self.assertWarns(DeprecationWarning):
            algo = NumPyEigensolver()
            result = algo.compute_eigenvalues(operator=self.qubit_op, aux_operators=aux_ops)

        self.assertEqual(len(result.eigenvalues), 1)
        self.assertEqual(len(result.eigenstates), 1)
        self.assertEqual(result.eigenvalues.dtype, np.float64)
        self.assertAlmostEqual(result.eigenvalues[0], -1.85727503)
        self.assertEqual(len(result.aux_operator_eigenvalues), 1)
        self.assertEqual(len(result.aux_operator_eigenvalues[0]), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1][0], 0, places=6)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1][1], 0.0)

        # Go again with additional None and zero operators
        extra_ops = [*aux_ops, None, 0]

        with self.assertWarns(DeprecationWarning):
            result = algo.compute_eigenvalues(operator=self.qubit_op, aux_operators=extra_ops)

        self.assertEqual(len(result.eigenvalues), 1)
        self.assertEqual(len(result.eigenstates), 1)
        self.assertEqual(result.eigenvalues.dtype, np.float64)
        self.assertAlmostEqual(result.eigenvalues[0], -1.85727503)
        self.assertEqual(len(result.aux_operator_eigenvalues), 1)
        self.assertEqual(len(result.aux_operator_eigenvalues[0]), 4)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1][0], 0, places=6)
        self.assertIsNone(result.aux_operator_eigenvalues[0][2], None)
        self.assertEqual(result.aux_operator_eigenvalues[0][3][0], 0.0)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1][1], 0.0)
        self.assertEqual(result.aux_operator_eigenvalues[0][3][1], 0.0)

    def test_aux_operators_dict(self):
        """Test dict-based aux_operators."""

        with self.assertWarns(DeprecationWarning):
            aux_op1 = PauliSumOp.from_list([("II", 2.0)])
            aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
        aux_ops = {"aux_op1": aux_op1, "aux_op2": aux_op2}

        with self.assertWarns(DeprecationWarning):
            algo = NumPyEigensolver()
            result = algo.compute_eigenvalues(operator=self.qubit_op, aux_operators=aux_ops)
        self.assertEqual(len(result.eigenvalues), 1)
        self.assertEqual(len(result.eigenstates), 1)
        self.assertEqual(result.eigenvalues.dtype, np.float64)
        self.assertAlmostEqual(result.eigenvalues[0], -1.85727503)
        self.assertEqual(len(result.aux_operator_eigenvalues), 1)
        self.assertEqual(len(result.aux_operator_eigenvalues[0]), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0]["aux_op1"][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0]["aux_op2"][0], 0, places=6)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0]["aux_op1"][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0]["aux_op2"][1], 0.0)

        # Go again with additional None and zero operators
        extra_ops = {**aux_ops, "None_operator": None, "zero_operator": 0}

        with self.assertWarns(DeprecationWarning):
            result = algo.compute_eigenvalues(operator=self.qubit_op, aux_operators=extra_ops)

        self.assertEqual(len(result.eigenvalues), 1)
        self.assertEqual(len(result.eigenstates), 1)
        self.assertEqual(result.eigenvalues.dtype, np.float64)
        self.assertAlmostEqual(result.eigenvalues[0], -1.85727503)
        self.assertEqual(len(result.aux_operator_eigenvalues), 1)
        self.assertEqual(len(result.aux_operator_eigenvalues[0]), 3)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0]["aux_op1"][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0]["aux_op2"][0], 0, places=6)
        self.assertEqual(result.aux_operator_eigenvalues[0]["zero_operator"][0], 0.0)
        self.assertTrue("None_operator" not in result.aux_operator_eigenvalues[0].keys())
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0]["aux_op1"][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0]["aux_op2"][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0]["zero_operator"][1], 0.0)


if __name__ == "__main__":
    unittest.main()
