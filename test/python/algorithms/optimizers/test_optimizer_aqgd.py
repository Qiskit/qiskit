# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2022
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
from qiskit.utils import QuantumInstance, algorithm_globals, optionals
from qiskit.opflow import PauliSumOp
from qiskit.algorithms.optimizers import AQGD
from qiskit.algorithms import VQE, AlgorithmError
from qiskit.opflow.gradients import Gradient
from qiskit.test import slow_test


class TestOptimizerAQGD(QiskitAlgorithmsTestCase):
    """Test AQGD optimizer using RY for analytic gradient with VQE"""

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

    @slow_test
    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_simple(self):
        """test AQGD optimizer with the parameters as single values."""
        from qiskit_aer import Aer

        q_instance = QuantumInstance(
            Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        aqgd = AQGD(momentum=0.0)
        vqe = VQE(
            ansatz=RealAmplitudes(),
            optimizer=aqgd,
            gradient=Gradient("lin_comb"),
            quantum_instance=q_instance,
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=3)

    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_list(self):
        """test AQGD optimizer with the parameters as lists."""
        from qiskit_aer import Aer

        q_instance = QuantumInstance(
            Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        aqgd = AQGD(maxiter=[1000, 1000, 1000], eta=[1.0, 0.5, 0.3], momentum=[0.0, 0.5, 0.75])
        vqe = VQE(ansatz=RealAmplitudes(), optimizer=aqgd, quantum_instance=q_instance)
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=3)

    def test_raises_exception(self):
        """tests that AQGD raises an exception when incorrect values are passed."""
        self.assertRaises(AlgorithmError, AQGD, maxiter=[1000], eta=[1.0, 0.5], momentum=[0.0, 0.5])

    @slow_test
    @unittest.skipUnless(optionals.HAS_AER, "qiskit-aer is required to run this test")
    def test_int_values(self):
        """test AQGD with int values passed as eta and momentum."""
        from qiskit_aer import Aer

        q_instance = QuantumInstance(
            Aer.get_backend("aer_simulator_statevector"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )

        aqgd = AQGD(maxiter=1000, eta=1, momentum=0)
        vqe = VQE(
            ansatz=RealAmplitudes(),
            optimizer=aqgd,
            gradient=Gradient("lin_comb"),
            quantum_instance=q_instance,
        )
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=3)


if __name__ == "__main__":
    unittest.main()
