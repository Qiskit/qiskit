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

""" Test NumPy Eigen solver """

import unittest
from test.aqua import QiskitAquaTestCase
import numpy as np
from qiskit.aqua import AquaError
from qiskit.aqua.algorithms import NumPyEigensolver
from qiskit.aqua.operators import WeightedPauliOperator


class TestNumPyEigensolver(QiskitAquaTestCase):
    """ Test NumPy Eigen solver """

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

    def test_ce(self):
        """ Test basics """
        algo = NumPyEigensolver(self.qubit_op, aux_operators=[])
        result = algo.run()
        self.assertEqual(len(result.eigenvalues), 1)
        self.assertEqual(len(result.eigenstates), 1)
        self.assertAlmostEqual(result.eigenvalues[0], -1.85727503 + 0j)

    def test_ce_k4(self):
        """ Test for k=4 eigenvalues """
        algo = NumPyEigensolver(self.qubit_op, k=4, aux_operators=[])
        result = algo.run()
        self.assertEqual(len(result.eigenvalues), 4)
        self.assertEqual(len(result.eigenstates), 4)
        np.testing.assert_array_almost_equal(result.eigenvalues.real,
                                             [-1.85727503, -1.24458455, -0.88272215, -0.22491125])

    def test_ce_fail(self):
        """ Test no operator """
        algo = NumPyEigensolver()
        with self.assertRaises(AquaError):
            _ = algo.run()

    def test_ce_k4_filtered(self):
        """ Test for k=4 eigenvalues with filter """

        # define filter criterion
        # pylint: disable=unused-argument
        def criterion(x, v, a_v):
            return v >= -1

        algo = NumPyEigensolver(self.qubit_op, k=4, aux_operators=[], filter_criterion=criterion)
        result = algo.run()
        self.assertEqual(len(result.eigenvalues), 2)
        self.assertEqual(len(result.eigenstates), 2)
        np.testing.assert_array_almost_equal(result.eigenvalues.real, [-0.88272215, -0.22491125])

    def test_ce_k4_filtered_empty(self):
        """ Test for k=4 eigenvalues with filter always returning False """

        # define filter criterion
        # pylint: disable=unused-argument
        def criterion(x, v, a_v):
            return False

        algo = NumPyEigensolver(self.qubit_op, k=4, aux_operators=[], filter_criterion=criterion)
        result = algo.run()
        self.assertEqual(len(result.eigenvalues), 0)
        self.assertEqual(len(result.eigenstates), 0)


if __name__ == '__main__':
    unittest.main()
