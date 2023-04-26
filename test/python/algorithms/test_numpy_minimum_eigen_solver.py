# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test NumPy Minimum Eigensolver"""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

import numpy as np
from ddt import ddt, data

from qiskit.algorithms import NumPyMinimumEigensolver
from qiskit.opflow import PauliSumOp, X, Y, Z


@ddt
class TestNumPyMinimumEigensolver(QiskitAlgorithmsTestCase):
    """Test NumPy Minimum Eigensolver"""

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
            aux_op1 = PauliSumOp.from_list([("II", 2.0)])
            aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])

        self.aux_ops_list = [aux_op1, aux_op2]
        self.aux_ops_dict = {"aux_op1": aux_op1, "aux_op2": aux_op2}

    def test_cme(self):
        """Basic test"""

        with self.assertWarns(DeprecationWarning):
            algo = NumPyMinimumEigensolver()
            result = algo.compute_minimum_eigenvalue(
                operator=self.qubit_op, aux_operators=self.aux_ops_list
            )

        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[0], [2, 0])
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[1], [0, 0])

    def test_cme_reuse(self):
        """Test reuse"""
        # Start with no operator or aux_operators, give via compute method
        with self.assertWarns(DeprecationWarning):
            algo = NumPyMinimumEigensolver()
            result = algo.compute_minimum_eigenvalue(operator=self.qubit_op)

        self.assertEqual(result.eigenvalue.dtype, np.float64)
        self.assertAlmostEqual(result.eigenvalue, -1.85727503)
        self.assertIsNone(result.aux_operator_eigenvalues)

        # Add aux_operators and go again
        with self.assertWarns(DeprecationWarning):
            result = algo.compute_minimum_eigenvalue(
                operator=self.qubit_op, aux_operators=self.aux_ops_list
            )

        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[0], [2, 0])
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[1], [0, 0])

        # "Remove" aux_operators and go again
        with self.assertWarns(DeprecationWarning):
            result = algo.compute_minimum_eigenvalue(operator=self.qubit_op, aux_operators=[])

        self.assertEqual(result.eigenvalue.dtype, np.float64)
        self.assertAlmostEqual(result.eigenvalue, -1.85727503)
        self.assertIsNone(result.aux_operator_eigenvalues)

        # Set aux_operators and go again
        with self.assertWarns(DeprecationWarning):
            result = algo.compute_minimum_eigenvalue(
                operator=self.qubit_op, aux_operators=self.aux_ops_list
            )

        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[0], [2, 0])
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[1], [0, 0])

        # Finally just set one of aux_operators and main operator, remove aux_operators

        with self.assertWarns(DeprecationWarning):
            result = algo.compute_minimum_eigenvalue(
                operator=self.aux_ops_list[0], aux_operators=[]
            )

        self.assertAlmostEqual(result.eigenvalue, 2 + 0j)
        self.assertIsNone(result.aux_operator_eigenvalues)

    def test_cme_filter(self):
        """Basic test"""

        # define filter criterion
        # pylint: disable=unused-argument
        def criterion(x, v, a_v):
            return v >= -0.5

        with self.assertWarns(DeprecationWarning):
            algo = NumPyMinimumEigensolver(filter_criterion=criterion)
            result = algo.compute_minimum_eigenvalue(
                operator=self.qubit_op, aux_operators=self.aux_ops_list
            )

        self.assertAlmostEqual(result.eigenvalue, -0.22491125 + 0j)
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[0], [2, 0])
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[1], [0, 0])

    def test_cme_filter_empty(self):
        """Test with filter always returning False"""

        # define filter criterion
        # pylint: disable=unused-argument
        def criterion(x, v, a_v):
            return False

        with self.assertWarns(DeprecationWarning):
            algo = NumPyMinimumEigensolver(filter_criterion=criterion)
            result = algo.compute_minimum_eigenvalue(
                operator=self.qubit_op, aux_operators=self.aux_ops_list
            )

        self.assertEqual(result.eigenvalue, None)
        self.assertEqual(result.eigenstate, None)
        self.assertEqual(result.aux_operator_eigenvalues, None)

    @data(X, Y, Z)
    def test_cme_1q(self, op):
        """Test for 1 qubit operator"""

        with self.assertWarns(DeprecationWarning):
            algo = NumPyMinimumEigensolver()
            result = algo.compute_minimum_eigenvalue(operator=op)

        self.assertAlmostEqual(result.eigenvalue, -1)

    def test_cme_aux_ops_dict(self):
        """Test dictionary compatibility of aux_operators"""
        # Start with an empty dictionary
        with self.assertWarns(DeprecationWarning):
            algo = NumPyMinimumEigensolver()
            result = algo.compute_minimum_eigenvalue(operator=self.qubit_op, aux_operators={})

        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertIsNone(result.aux_operator_eigenvalues)

        # Add aux_operators dictionary and go again
        with self.assertWarns(DeprecationWarning):
            result = algo.compute_minimum_eigenvalue(
                operator=self.qubit_op, aux_operators=self.aux_ops_dict
            )

        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues["aux_op1"], [2, 0])
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues["aux_op2"], [0, 0])

        # Add None and zero operators and go again
        extra_ops = {"None_op": None, "zero_op": 0, **self.aux_ops_dict}
        with self.assertWarns(DeprecationWarning):
            result = algo.compute_minimum_eigenvalue(
                operator=self.qubit_op, aux_operators=extra_ops
            )

        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertEqual(len(result.aux_operator_eigenvalues), 3)
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues["aux_op1"], [2, 0])
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues["aux_op2"], [0, 0])
        self.assertEqual(result.aux_operator_eigenvalues["zero_op"], (0.0, 0))

    def test_aux_operators_list(self):
        """Test list-based aux_operators."""

        with self.assertWarns(DeprecationWarning):
            aux_op1 = PauliSumOp.from_list([("II", 2.0)])
            aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
        aux_ops = [aux_op1, aux_op2]

        with self.assertWarns(DeprecationWarning):
            algo = NumPyMinimumEigensolver()
            result = algo.compute_minimum_eigenvalue(operator=self.qubit_op, aux_operators=aux_ops)

        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][0], 0, places=6)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][1], 0.0)

        # Go again with additional None and zero operators
        extra_ops = [*aux_ops, None, 0]

        with self.assertWarns(DeprecationWarning):
            result = algo.compute_minimum_eigenvalue(
                operator=self.qubit_op, aux_operators=extra_ops
            )

        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertEqual(len(result.aux_operator_eigenvalues), 4)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][0], 0, places=6)
        self.assertIsNone(result.aux_operator_eigenvalues[2], None)
        self.assertEqual(result.aux_operator_eigenvalues[3][0], 0.0)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues[0][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues[1][1], 0.0)
        self.assertEqual(result.aux_operator_eigenvalues[3][1], 0.0)

    def test_aux_operators_dict(self):
        """Test dict-based aux_operators."""

        with self.assertWarns(DeprecationWarning):
            aux_op1 = PauliSumOp.from_list([("II", 2.0)])
            aux_op2 = PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)])
        aux_ops = {"aux_op1": aux_op1, "aux_op2": aux_op2}

        with self.assertWarns(DeprecationWarning):
            algo = NumPyMinimumEigensolver()
            result = algo.compute_minimum_eigenvalue(operator=self.qubit_op, aux_operators=aux_ops)

        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op1"][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op2"][0], 0, places=6)
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op1"][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op2"][1], 0.0)

        # Go again with additional None and zero operators
        extra_ops = {**aux_ops, "None_operator": None, "zero_operator": 0}

        with self.assertWarns(DeprecationWarning):
            result = algo.compute_minimum_eigenvalue(
                operator=self.qubit_op, aux_operators=extra_ops
            )

        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertEqual(len(result.aux_operator_eigenvalues), 3)
        # expectation values
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op1"][0], 2, places=6)
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op2"][0], 0, places=6)
        self.assertEqual(result.aux_operator_eigenvalues["zero_operator"][0], 0.0)
        self.assertTrue("None_operator" not in result.aux_operator_eigenvalues.keys())
        # standard deviations
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op1"][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues["aux_op2"][1], 0.0)
        self.assertAlmostEqual(result.aux_operator_eigenvalues["zero_operator"][1], 0.0)


if __name__ == "__main__":
    unittest.main()
