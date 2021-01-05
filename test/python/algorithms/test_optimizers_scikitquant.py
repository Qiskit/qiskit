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

""" Test of scikit-quant optimizers. """

import unittest

from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit import BasicAer
from qiskit.circuit.library import RealAmplitudes
from qiskit.utils import QuantumInstance, aqua_globals
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.opflow import WeightedPauliOperator
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import BOBYQA, SNOBFIT, IMFIL


class TestOptimizers(QiskitAlgorithmsTestCase):
    """ Test scikit-quant optimizers. """

    def setUp(self):
        """ Set the problem. """
        super().setUp()
        aqua_globals.random_seed = 50
        pauli_dict = {
            'paulis': [{"coeff": {"imag": 0.0, "real": -1.052373245772859}, "label": "II"},
                       {"coeff": {"imag": 0.0, "real": 0.39793742484318045}, "label": "IZ"},
                       {"coeff": {"imag": 0.0, "real": -0.39793742484318045}, "label": "ZI"},
                       {"coeff": {"imag": 0.0, "real": -0.01128010425623538}, "label": "ZZ"},
                       {"coeff": {"imag": 0.0, "real": 0.18093119978423156}, "label": "XX"}
                       ]
        }
        self.qubit_op = WeightedPauliOperator.from_dict(pauli_dict).to_opflow()

    def _optimize(self, optimizer):
        """ launch vqe """
        qe = QuantumInstance(BasicAer.get_backend('statevector_simulator'),
                             seed_simulator=aqua_globals.random_seed,
                             seed_transpiler=aqua_globals.random_seed)
        vqe = VQE(var_form=RealAmplitudes(),
                  optimizer=optimizer,
                  quantum_instance=qe)
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=1)

    def test_bobyqa(self):
        """ BOBYQA optimizer test. """
        try:
            optimizer = BOBYQA(maxiter=150)
            self._optimize(optimizer)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_snobfit(self):
        """ SNOBFIT optimizer test. """
        try:
            optimizer = SNOBFIT(maxiter=100, maxfail=100, maxmp=20)
            self._optimize(optimizer)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_imfil(self):
        """ IMFIL test. """
        try:
            optimizer = IMFIL(maxiter=100)
            self._optimize(optimizer)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))


if __name__ == '__main__':
    unittest.main()
