# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the SPSA optimizer."""

from test.python.algorithms import QiskitAlgorithmsTestCase

import numpy as np

from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import PauliTwoDesign
from qiskit.opflow import I, Z, StateFn


class TestSPSA(QiskitAlgorithmsTestCase):
    """Tests for the SPSA optimizer."""

    def setUp(self):
        super().setUp()
        np.random.seed(12)

    def test_pauli_two_design(self):
        """Test SPSA on the Pauli two-design example."""
        circuit = PauliTwoDesign(3, reps=3, seed=2)
        parameters = list(circuit.parameters)
        obs = Z ^ Z ^ I
        expr = ~StateFn(obs) @ StateFn(circuit)

        initial_point = np.array([0.1822308, -0.27254251,  0.83684425,  0.86153976, -0.7111668,
                                  0.82766631,  0.97867993,  0.46136964,  2.27079901,  0.13382699,
                                  0.29589915,  0.64883193])

        def objective(x):
            return expr.bind_parameters(dict(zip(parameters, x))).eval().real

        spsa = SPSA(maxiter=100,
                    blocking=True,
                    allowed_increase=0,
                    learning_rate=0.1,
                    perturbation=0.1)

        result = spsa.optimize(circuit.num_parameters, objective, initial_point=initial_point)

        self.assertLess(result[1], -0.95)  # final loss
        self.assertEqual(result[2], 301)  # function evaluations
