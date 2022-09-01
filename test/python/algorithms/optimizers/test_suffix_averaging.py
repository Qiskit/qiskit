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

""" Test of Suffix averaging """

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

import numpy as np

from qiskit import BasicAer
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import ADAM
from qiskit.algorithms.optimizers.suffix_averaging import SuffixAveragingOptimizer
from qiskit.circuit.library import RealAmplitudes
from qiskit.opflow import PauliSumOp
from qiskit.utils import QuantumInstance, algorithm_globals


class TestOptimizerSA(QiskitAlgorithmsTestCase):
    """Test suffix averaging technique using RY with VQE"""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        self.qubit_op = PauliSumOp.from_list(
            [
                ("II", -1.052373245772859),
                ("IZ", 0.39793742484318045),
                ("ZI", -0.39793742484318045),
                ("ZZ", -0.01128010425623538),
                ("XX", 0.18093119978423156),
            ]
        )

    def test_suffix_averaging(self):
        """Test suffix averaging."""
        circ_params = []
        n_params_suffix = 5
        maxiter = 7

        # pylint: disable=unused-argument
        def store_intermediate_result(eval_count, parameters, mean, std):
            circ_params.append(parameters)

        suffix_optimizer = SuffixAveragingOptimizer(
            ADAM(maxiter=maxiter, tol=0.0), n_params_suffix=n_params_suffix
        )

        vqe = VQE(
            ansatz=RealAmplitudes(),
            optimizer=suffix_optimizer,
            callback=store_intermediate_result,
            quantum_instance=QuantumInstance(
                BasicAer.get_backend("statevector_simulator"),
                seed_simulator=algorithm_globals.random_seed,
                seed_transpiler=algorithm_globals.random_seed,
            ),
        )

        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)

        n_param = len(circ_params[0])
        average_params = np.mean(circ_params[-2:-n_params_suffix*(n_param+1):-(n_param+1)], axis=0)

        np.testing.assert_array_almost_equal(result.optimal_point, average_params)


if __name__ == "__main__":
    unittest.main()
