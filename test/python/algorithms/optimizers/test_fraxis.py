# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of Fraxis optimizer """

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

from qiskit.algorithms.minimum_eigensolvers import VQE
from qiskit.algorithms.optimizers import FraxisOptimizer
from qiskit.circuit.library import FraxisCircuit
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.utils import algorithm_globals


class TestOptimizerNFT(QiskitAlgorithmsTestCase):
    """Test Fraxis optimizer with VQE"""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        self.qubit_op = SparsePauliOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )

    def test_nft(self):
        """Test Fraxis optimizer by using it"""

        vqe = VQE(
            estimator=Estimator(),
            ansatz=FraxisCircuit(),
            optimizer=FraxisOptimizer(),
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, -1.857275, places=6)


if __name__ == "__main__":
    unittest.main()
