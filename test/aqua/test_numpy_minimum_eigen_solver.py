# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test NumPy Minimum Eigensolver """

import unittest
from test.aqua import QiskitAquaTestCase
import numpy as np
from qiskit.aqua import AquaError
from qiskit.aqua.algorithms import NumPyMinimumEigensolver
from qiskit.aqua.operators import WeightedPauliOperator


class TestNumPyMinimumEigensolver(QiskitAquaTestCase):
    """ Test NumPy Minimum Eigensolver """

    def setUp(self):
        super().setUp()
        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }
        self.qubit_op = WeightedPauliOperator.from_dict(pauli_dict)

        aux_dict_1 = {
            'paulis': [{'coeff': {'imag': 0.0, 'real': 2.0}, 'label': 'II'}]
        }
        aux_dict_2 = {
            'paulis': [{'coeff': {'imag': 0.0, 'real': 0.5}, 'label': 'II'},
                       {'coeff': {'imag': 0.0, 'real': 0.5}, 'label': 'ZZ'},
                       {'coeff': {'imag': 0.0, 'real': 0.5}, 'label': 'YY'},
                       {'coeff': {'imag': 0.0, 'real': -0.5}, 'label': 'XX'}
                       ]
        }
        self.aux_ops = [WeightedPauliOperator.from_dict(aux_dict_1),
                        WeightedPauliOperator.from_dict(aux_dict_2)]

    def test_cme(self):
        """ Basic test """
        algo = NumPyMinimumEigensolver(self.qubit_op, aux_operators=self.aux_ops)
        result = algo.run()
        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[0], [2, 0])
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[1], [0, 0])

    def test_cme_fail(self):
        """ Test no operator """
        algo = NumPyMinimumEigensolver()
        with self.assertRaises(AquaError):
            _ = algo.run()

    def test_cme_reuse(self):
        """ Test reuse """
        # Start with no operator or aux_operators, give via compute method
        algo = NumPyMinimumEigensolver()
        result = algo.compute_minimum_eigenvalue(self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertEqual(self.qubit_op.to_opflow(), algo.operator)
        self.assertIsNone(result.aux_operator_eigenvalues)

        # Set operator to None and go again
        algo.operator = None
        with self.assertRaises(AquaError):
            _ = algo.run()

        # Set operator back as it was and go again
        algo.operator = self.qubit_op
        result = algo.compute_minimum_eigenvalue()
        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertIsNone(result.aux_operator_eigenvalues)

        # Add aux_operators and go again
        result = algo.compute_minimum_eigenvalue(aux_operators=self.aux_ops)
        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[0], [2, 0])
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[1], [0, 0])

        # "Remove" aux_operators and go again
        result = algo.compute_minimum_eigenvalue(aux_operators=[])
        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertIsNone(result.aux_operator_eigenvalues)

        # Set aux_operators and go again
        algo.aux_operators = self.aux_ops
        result = algo.compute_minimum_eigenvalue()
        self.assertAlmostEqual(result.eigenvalue, -1.85727503 + 0j)
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[0], [2, 0])
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[1], [0, 0])

        # Finally just set one of aux_operators and main operator, remove aux_operators
        result = algo.compute_minimum_eigenvalue(self.aux_ops[0], [])
        self.assertAlmostEqual(result.eigenvalue, 2 + 0j)
        self.assertIsNone(result.aux_operator_eigenvalues)

    def test_cme_filter(self):
        """ Basic test """

        # define filter criterion
        # pylint: disable=unused-argument
        def criterion(x, v, a_v):
            return v >= -0.5

        algo = NumPyMinimumEigensolver(
            self.qubit_op, aux_operators=self.aux_ops, filter_criterion=criterion)

        result = algo.run()
        self.assertAlmostEqual(result.eigenvalue, -0.22491125 + 0j)
        self.assertEqual(len(result.aux_operator_eigenvalues), 2)
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[0], [2, 0])
        np.testing.assert_array_almost_equal(result.aux_operator_eigenvalues[1], [0, 0])

    def test_cme_filter_empty(self):
        """ Test with filter always returning False """

        # define filter criterion
        # pylint: disable=unused-argument
        def criterion(x, v, a_v):
            return False

        algo = NumPyMinimumEigensolver(
            self.qubit_op, aux_operators=self.aux_ops, filter_criterion=criterion)
        result = algo.run()
        self.assertEqual(result.eigenvalue, None)
        self.assertEqual(result.eigenstate, None)
        self.assertEqual(result.aux_operator_eigenvalues, None)


if __name__ == '__main__':
    unittest.main()
