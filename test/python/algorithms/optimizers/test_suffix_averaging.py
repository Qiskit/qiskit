""" Test of Suffix averaging """

import unittest
from qiskit.algorithms.optimizers.suffix_averaging import SuffixAveragingOptimizer
from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit import BasicAer
from qiskit.circuit.library import RealAmplitudes
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import PauliSumOp
from qiskit.algorithms.optimizers import ADAM
from qiskit.algorithms import VQE
import numpy as np


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
        maxiter = 5

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

        average_params = np.zeros_like(circ_params[0])
        for i in range(n_params_suffix):
            average_params += circ_params[maxiter - i - 2]
        average_params /= n_params_suffix

        self.assertListEqual(result.optimal_point, average_params)


if __name__ == "__main__":
    unittest.main()
