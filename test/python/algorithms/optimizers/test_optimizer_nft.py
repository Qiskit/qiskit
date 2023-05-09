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

"""Test of NFT optimizer"""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit import BasicAer
from qiskit.circuit.library import RealAmplitudes
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.opflow import PauliSumOp
from qiskit.algorithms.optimizers import NFT
from qiskit.algorithms import VQE


class TestOptimizerNFT(QiskitAlgorithmsTestCase):
    """Test NFT optimizer using RY with VQE"""

    def setUp(self):
        super().setUp()
        algorithm_globals.random_seed = 50
        with self.assertWarns(DeprecationWarning):
            self.qubit_op = PauliSumOp.from_list(
                [
                    ("II", -1.052373245772859),
                    ("IZ", 0.39793742484318045),
                    ("ZI", -0.39793742484318045),
                    ("ZZ", -0.01128010425623538),
                    ("XX", 0.18093119978423156),
                ]
            )

    def test_nft(self):
        """Test NFT optimizer by using it"""
        with self.assertWarns(DeprecationWarning):
            vqe = VQE(
                ansatz=RealAmplitudes(),
                optimizer=NFT(),
                quantum_instance=QuantumInstance(
                    BasicAer.get_backend("statevector_simulator"),
                    seed_simulator=algorithm_globals.random_seed,
                    seed_transpiler=algorithm_globals.random_seed,
                ),
            )
            result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)

        self.assertAlmostEqual(result.eigenvalue.real, -1.857275, places=6)


if __name__ == "__main__":
    unittest.main()
