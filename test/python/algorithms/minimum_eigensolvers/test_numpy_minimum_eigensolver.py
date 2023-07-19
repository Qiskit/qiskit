# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test NumPy minimum eigensolver"""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

import numpy as np
from ddt import ddt, data

from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info import Operator, SparsePauliOp

H2_SPARSE_PAULI = SparsePauliOp(
    ["II", "ZI", "IZ", "ZZ", "XX"],
    coeffs=[
        -1.052373245772859,
        0.39793742484318045,
        -0.39793742484318045,
        -0.01128010425623538,
        0.18093119978423156,
    ],
)

H2_OP = Operator(H2_SPARSE_PAULI.to_matrix())

H2_PAULI = PauliSumOp(H2_SPARSE_PAULI)


@ddt
class TestNumPyMinimumEigensolver(QiskitAlgorithmsTestCase):
    """Test NumPy minimum eigensolver"""

    def setUp(self):
        super().setUp()
        aux_op1 = Operator(SparsePauliOp(["II"], coeffs=[2.0]).to_matrix())
        aux_op2 = SparsePauliOp(["II", "ZZ", "YY", "XX"], coeffs=[0.5, 0.5, 0.5, -0.5])
        self.aux_ops_list = [aux_op1, aux_op2]
        self.aux_ops_dict = {"aux_op1": aux_op1, "aux_op2": aux_op2}

    @data(H2_SPARSE_PAULI, H2_PAULI, H2_OP)
    def test_cme(self, op):
        """Basic test"""
        algo = NumPyMinimumEigensolver()
        result = algo.compute_minimum_eigenvalue(operator=op, aux_operators=self.aux_ops_list)
        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertEqual(len(result.aux_operators_evaluated), 2)
        self.assertAlmostEqual(result.aux_operators_evaluated[0][0], 2)
        self.assertAlmostEqual(result.aux_operators_evaluated[1][0], 0)

    @data(H2_SPARSE_PAULI, H2_PAULI, H2_OP)
    def test_cme_reuse(self, op):
        """Test reuse"""
        algo = NumPyMinimumEigensolver()

        with self.subTest("Test with no operator or aux_operators, give via compute method"):
            result = algo.compute_minimum_eigenvalue(operator=op)
            self.assertEqual(result.eigenvalue.dtype, np.float64)
            self.assertAlmostEqual(result.eigenvalue, -1.85727503)
            self.assertIsNone(result.aux_operators_evaluated)

        with self.subTest("Test with added aux_operators"):
            result = algo.compute_minimum_eigenvalue(operator=op, aux_operators=self.aux_ops_list)
            self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
            self.assertEqual(len(result.aux_operators_evaluated), 2)
            self.assertAlmostEqual(result.aux_operators_evaluated[0][0], 2)
            self.assertAlmostEqual(result.aux_operators_evaluated[1][0], 0)

        with self.subTest("Test with aux_operators removed"):
            result = algo.compute_minimum_eigenvalue(operator=op, aux_operators=[])
            self.assertEqual(result.eigenvalue.dtype, np.float64)
            self.assertAlmostEqual(result.eigenvalue, -1.85727503)
            self.assertIsNone(result.aux_operators_evaluated)

        with self.subTest("Test with aux_operators set again"):
            result = algo.compute_minimum_eigenvalue(operator=op, aux_operators=self.aux_ops_list)
            self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
            self.assertEqual(len(result.aux_operators_evaluated), 2)
            self.assertAlmostEqual(result.aux_operators_evaluated[0][0], 2)
            self.assertAlmostEqual(result.aux_operators_evaluated[1][0], 0)

        with self.subTest("Test after setting first aux_operators as main operator"):
            result = algo.compute_minimum_eigenvalue(
                operator=self.aux_ops_list[0], aux_operators=[]
            )
            self.assertAlmostEqual(result.eigenvalue, 2 + 0j)
            self.assertIsNone(result.aux_operators_evaluated)

    @data(H2_SPARSE_PAULI, H2_PAULI, H2_OP)
    def test_cme_filter(self, op):
        """Basic test"""

        # define filter criterion
        # pylint: disable=unused-argument
        def criterion(x, v, a_v):
            return v >= -0.5

        algo = NumPyMinimumEigensolver(filter_criterion=criterion)
        result = algo.compute_minimum_eigenvalue(operator=op, aux_operators=self.aux_ops_list)
        self.assertAlmostEqual(result.eigenvalue, -0.22491125 + 0j)
        self.assertEqual(len(result.aux_operators_evaluated), 2)
        self.assertAlmostEqual(result.aux_operators_evaluated[0][0], 2)
        self.assertAlmostEqual(result.aux_operators_evaluated[1][0], 0)

    @data(H2_SPARSE_PAULI, H2_PAULI, H2_OP)
    def test_cme_filter_empty(self, op):
        """Test with filter always returning False"""

        # define filter criterion
        # pylint: disable=unused-argument
        def criterion(x, v, a_v):
            return False

        algo = NumPyMinimumEigensolver(filter_criterion=criterion)
        result = algo.compute_minimum_eigenvalue(operator=op, aux_operators=self.aux_ops_list)
        self.assertEqual(result.eigenvalue, None)
        self.assertEqual(result.eigenstate, None)
        self.assertEqual(result.aux_operators_evaluated, None)

    @data("X", "Y", "Z")
    def test_cme_1q(self, op):
        """Test for 1 qubit operator"""
        algo = NumPyMinimumEigensolver()
        operator = SparsePauliOp([op], coeffs=1.0)
        result = algo.compute_minimum_eigenvalue(operator=operator)
        self.assertAlmostEqual(result.eigenvalue, -1)

    @data(H2_SPARSE_PAULI, H2_PAULI, H2_OP)
    def test_cme_aux_ops_dict(self, op):
        """Test dictionary compatibility of aux_operators"""
        # Start with an empty dictionary
        algo = NumPyMinimumEigensolver()

        with self.subTest("Test with an empty dictionary."):
            result = algo.compute_minimum_eigenvalue(operator=op, aux_operators={})
            self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
            self.assertIsNone(result.aux_operators_evaluated)

        with self.subTest("Test with two auxiliary operators."):
            result = algo.compute_minimum_eigenvalue(operator=op, aux_operators=self.aux_ops_dict)
            self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
            self.assertEqual(len(result.aux_operators_evaluated), 2)
            self.assertAlmostEqual(result.aux_operators_evaluated["aux_op1"][0], 2)
            self.assertAlmostEqual(result.aux_operators_evaluated["aux_op2"][0], 0)

        with self.subTest("Test with additional zero and None operators."):
            extra_ops = {"None_op": None, "zero_op": 0, **self.aux_ops_dict}
            result = algo.compute_minimum_eigenvalue(operator=op, aux_operators=extra_ops)
            self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
            self.assertEqual(len(result.aux_operators_evaluated), 3)
            self.assertAlmostEqual(result.aux_operators_evaluated["aux_op1"][0], 2)
            self.assertAlmostEqual(result.aux_operators_evaluated["aux_op2"][0], 0)
            self.assertEqual(result.aux_operators_evaluated["zero_op"], (0.0, {"variance": 0}))

    @data(H2_SPARSE_PAULI, H2_PAULI, H2_OP)
    def test_aux_operators_list(self, op):
        """Test list-based aux_operators."""
        algo = NumPyMinimumEigensolver()

        with self.subTest("Test with two auxiliary operators."):
            result = algo.compute_minimum_eigenvalue(operator=op, aux_operators=self.aux_ops_list)
            self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
            self.assertEqual(len(result.aux_operators_evaluated), 2)
            # expectation values
            self.assertAlmostEqual(result.aux_operators_evaluated[0][0], 2, places=6)
            self.assertAlmostEqual(result.aux_operators_evaluated[1][0], 0, places=6)
            # standard deviations
            self.assertAlmostEqual(result.aux_operators_evaluated[0][1].pop("variance"), 0.0)
            self.assertAlmostEqual(result.aux_operators_evaluated[1][1].pop("variance"), 0.0)

        with self.subTest("Test with additional zero and None operators."):
            extra_ops = [*self.aux_ops_list, None, 0]
            result = algo.compute_minimum_eigenvalue(operator=op, aux_operators=extra_ops)
            self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
            self.assertEqual(len(result.aux_operators_evaluated), 4)
            # expectation values
            self.assertAlmostEqual(result.aux_operators_evaluated[0][0], 2, places=6)
            self.assertAlmostEqual(result.aux_operators_evaluated[1][0], 0, places=6)
            self.assertIsNone(result.aux_operators_evaluated[2], None)
            self.assertEqual(result.aux_operators_evaluated[3][0], 0.0)
            # standard deviations
            self.assertAlmostEqual(result.aux_operators_evaluated[0][1].pop("variance"), 0.0)
            self.assertAlmostEqual(result.aux_operators_evaluated[1][1].pop("variance"), 0.0)
            self.assertEqual(result.aux_operators_evaluated[3][1].pop("variance"), 0.0)

    @data(H2_SPARSE_PAULI, H2_PAULI, H2_OP)
    def test_aux_operators_dict(self, op):
        """Test dict-based aux_operators."""
        algo = NumPyMinimumEigensolver()

        with self.subTest("Test with two auxiliary operators."):
            result = algo.compute_minimum_eigenvalue(operator=op, aux_operators=self.aux_ops_dict)
            self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
            self.assertEqual(len(result.aux_operators_evaluated), 2)
            # expectation values
            self.assertAlmostEqual(result.aux_operators_evaluated["aux_op1"][0], 2, places=6)
            self.assertAlmostEqual(result.aux_operators_evaluated["aux_op2"][0], 0, places=6)
            # standard deviations
            self.assertAlmostEqual(
                result.aux_operators_evaluated["aux_op1"][1].pop("variance"), 0.0
            )
            self.assertAlmostEqual(
                result.aux_operators_evaluated["aux_op2"][1].pop("variance"), 0.0
            )

        with self.subTest("Test with additional zero and None operators."):
            extra_ops = {**self.aux_ops_dict, "None_operator": None, "zero_operator": 0}
            result = algo.compute_minimum_eigenvalue(operator=op, aux_operators=extra_ops)
            self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
            self.assertEqual(len(result.aux_operators_evaluated), 3)
            # expectation values
            self.assertAlmostEqual(result.aux_operators_evaluated["aux_op1"][0], 2, places=6)
            self.assertAlmostEqual(result.aux_operators_evaluated["aux_op2"][0], 0, places=6)
            self.assertEqual(result.aux_operators_evaluated["zero_operator"][0], 0.0)
            self.assertTrue("None_operator" not in result.aux_operators_evaluated.keys())
            # standard deviations
            self.assertAlmostEqual(
                result.aux_operators_evaluated["aux_op1"][1].pop("variance"), 0.0
            )
            self.assertAlmostEqual(
                result.aux_operators_evaluated["aux_op2"][1].pop("variance"), 0.0
            )
            self.assertAlmostEqual(
                result.aux_operators_evaluated["zero_operator"][1].pop("variance"), 0.0
            )


if __name__ == "__main__":
    unittest.main()
