# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Test of AQGD optimizer """

from test.aqua import QiskitAquaTestCase
from qiskit import BasicAer

from qiskit.circuit.library import RealAmplitudes
from qiskit.aqua import QuantumInstance, aqua_globals, AquaError
from qiskit.aqua.operators import WeightedPauliOperator
from qiskit.aqua.components.optimizers import AQGD
from qiskit.aqua.algorithms import VQE


class TestOptimizerAQGD(QiskitAquaTestCase):
    """ Test AQGD optimizer using RY for analytic gradient with VQE """

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

    def test_simple(self):
        """ test AQGD optimizer with the parameters as single values."""
        q_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                     seed_simulator=aqua_globals.random_seed,
                                     seed_transpiler=aqua_globals.random_seed)

        aqgd = AQGD(momentum=0.0)
        result = VQE(self.qubit_op, RealAmplitudes(), aqgd).run(q_instance)
        self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=3)

    def test_list(self):
        """ test AQGD optimizer with the parameters as lists. """
        q_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                     seed_simulator=aqua_globals.random_seed,
                                     seed_transpiler=aqua_globals.random_seed)

        aqgd = AQGD(maxiter=[1000, 1000, 1000], eta=[1.0, 0.5, 0.3], momentum=[0.0, 0.5, 0.75])
        result = VQE(self.qubit_op, RealAmplitudes(), aqgd).run(q_instance)
        self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=3)

    def test_raises_exception(self):
        """ tests that AQGD raises an exception when incorrect values are passed. """
        self.assertRaises(AquaError, AQGD, maxiter=[1000], eta=[1.0, 0.5], momentum=[0.0, 0.5])

    def test_int_values(self):
        """ test AQGD with int values passed as eta and momentum. """
        q_instance = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                                     seed_simulator=aqua_globals.random_seed,
                                     seed_transpiler=aqua_globals.random_seed)

        aqgd = AQGD(maxiter=1000, eta=1, momentum=0)
        result = VQE(self.qubit_op, RealAmplitudes(), aqgd).run(q_instance)
        self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=3)
