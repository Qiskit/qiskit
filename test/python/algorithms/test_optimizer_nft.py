# This code is part of Qiskit.
#
# (C) Copyright IBM 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of NFT optimizer """

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit import BasicAer
from qiskit.circuit.library import RealAmplitudes
from qiskit.utils import QuantumInstance, aqua_globals
from qiskit.opflow import WeightedPauliOperator
from qiskit.algorithms.optimizers import NFT
from qiskit.algorithms import VQE


class TestOptimizerNFT(QiskitAlgorithmsTestCase):
    """ Test NFT optimizer using RY with VQE """

    def setUp(self):
        super().setUp()
        self.seed = 50
        aqua_globals.random_seed = self.seed
        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }
        self.qubit_op = WeightedPauliOperator.from_dict(pauli_dict)

    def test_nft(self):
        """ Test NFT optimizer by using it """

        result = VQE(self.qubit_op,
                     RealAmplitudes(),
                     NFT()).run(
                         QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                         seed_simulator=aqua_globals.random_seed,
                                         seed_transpiler=aqua_globals.random_seed))
        self.assertAlmostEqual(result.eigenvalue.real, -1.857275, places=6)


if __name__ == '__main__':
    unittest.main()
