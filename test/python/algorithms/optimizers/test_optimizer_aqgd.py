# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of AQGD optimizer """

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit.circuit.library import RealAmplitudes
from qiskit.utils import algorithm_globals
from qiskit.opflow import PauliSumOp
from qiskit.algorithms.optimizers import AQGD
from qiskit.algorithms import AlgorithmError
from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.primitives import Estimator
from qiskit.algorithms.gradients import LinCombEstimatorGradient


class TestOptimizerAQGD(QiskitAlgorithmsTestCase):
    """Test AQGD optimizer using RY for analytic gradient with VQE"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 50
        self.qubit_op = PauliSumOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )

    def test_simple(self):
        """test AQGD optimizer with the parameters as single values."""

        est = Estimator()
        aqgd = AQGD(momentum=0.0)
        vqe = VQE(
            est,
            ansatz=RealAmplitudes(),
            optimizer=aqgd,
            gradient=LinCombEstimatorGradient(est),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=3)

    def test_list(self):
        """test AQGD optimizer with the parameters as lists."""

        aqgd = AQGD(maxiter=[1000, 1000, 1000], eta=[1.0, 0.5, 0.3], momentum=[0.0, 0.5, 0.75])
        vqe = VQE(Estimator(), ansatz=RealAmplitudes(), optimizer=aqgd)
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=3)

    def test_raises_exception(self):
        """tests that AQGD raises an exception when incorrect values are passed."""
        self.assertRaises(AlgorithmError, AQGD, maxiter=[1000], eta=[1.0, 0.5], momentum=[0.0, 0.5])

    def test_int_values(self):
        """test AQGD with int values passed as eta and momentum."""

        est = Estimator()
        aqgd = AQGD(maxiter=1000, eta=1, momentum=0)
        vqe = VQE(
            est,
            ansatz=RealAmplitudes(),
            optimizer=aqgd,
            gradient=LinCombEstimatorGradient(est),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=3)


if __name__ == "__main__":
    unittest.main()
