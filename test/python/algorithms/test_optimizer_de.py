
# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test of differential evolution optimizer."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from scipy.optimize import rosen
import numpy as np

from qiskit import BasicAer
from qiskit.algorithms import VQE
from qiskit.circuit.library import RealAmplitudes
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import PauliSumOp
from qiskit.algorithms.optimizers import DE


class TestOptimizerDE(QiskitAlgorithmsTestCase):
    """ Test Differential Evolution (DE) optimizer"""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        self.qubit_op = PauliSumOp.from_list([
            ("II", -1.052373245772859),
            ("IZ", 0.39793742484318045),
            ("ZI", -0.39793742484318045),
            ("ZZ", -0.01128010425623538),
            ("XX", 0.18093119978423156),
        ])

    def _optimize(self, optimizer):
        x_0 = [1.3, 0.7, 0.8, 1.9, 1.2]
        bounds = [(-6, 6)] * len(x_0)
        res = optimizer.optimize(len(x_0), rosen, initial_point=x_0, variable_bounds=bounds)
        np.testing.assert_array_almost_equal(res[0], [1.0] * len(x_0), decimal=2)
        return res

    def test_de(self):
        """ Test DE optimizer by using it """

        vqe = VQE(var_form=RealAmplitudes(),
                  optimizer=DE(),
                  quantum_instance=QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                                   seed_simulator=algorithm_globals.random_seed,
                                                   seed_transpiler=algorithm_globals.random_seed))
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, -1.857275, places=6)


if __name__ == '__main__':
    unittest.main()
