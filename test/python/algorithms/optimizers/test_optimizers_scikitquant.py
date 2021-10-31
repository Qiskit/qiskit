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

""" Test of scikit-quant optimizers. """

import unittest

from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit import BasicAer
from qiskit.circuit.library import RealAmplitudes
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.opflow import PauliSumOp
from qiskit.algorithms import VQE
from qiskit.algorithms.optimizers import BOBYQA, SNOBFIT, IMFIL


class TestOptimizers(QiskitAlgorithmsTestCase):
    """Test scikit-quant optimizers."""

    def setUp(self):
        """Set the problem."""
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

    def _optimize(self, optimizer):
        """launch vqe"""
        qe = QuantumInstance(
            BasicAer.get_backend("statevector_simulator"),
            seed_simulator=algorithm_globals.random_seed,
            seed_transpiler=algorithm_globals.random_seed,
        )
        vqe = VQE(ansatz=RealAmplitudes(), optimizer=optimizer, quantum_instance=qe)
        result = vqe.compute_minimum_eigenvalue(operator=self.qubit_op)
        self.assertAlmostEqual(result.eigenvalue.real, -1.857, places=1)

    def test_bobyqa(self):
        """BOBYQA optimizer test."""
        try:
            optimizer = BOBYQA(maxiter=150)
            self._optimize(optimizer)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_snobfit(self):
        """SNOBFIT optimizer test."""
        try:
            optimizer = SNOBFIT(maxiter=100, maxfail=100, maxmp=20)
            self._optimize(optimizer)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))

    def test_imfil(self):
        """IMFIL test."""
        try:
            optimizer = IMFIL(maxiter=100)
            self._optimize(optimizer)
        except MissingOptionalLibraryError as ex:
            self.skipTest(str(ex))


if __name__ == "__main__":
    unittest.main()
