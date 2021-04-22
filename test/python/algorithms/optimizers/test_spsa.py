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
from ddt import ddt, data

import numpy as np

from qiskit.algorithms.optimizers import SPSA
from qiskit.circuit.library import PauliTwoDesign
from qiskit.opflow import I, Z, StateFn


@ddt
class TestSPSA(QiskitAlgorithmsTestCase):
    """Tests for the SPSA optimizer."""

    def setUp(self):
        super().setUp()
        np.random.seed(12)

    @data(True, False)
    def test_pauli_two_design(self, second_order):
        """Test SPSA on the Pauli two-design example."""
        circuit = PauliTwoDesign(3, reps=3, seed=2)
        parameters = list(circuit.parameters)
        obs = Z ^ Z ^ I
        expr = ~StateFn(obs) @ StateFn(circuit)

        # starting at around -0.57
        initial_point = np.array([-1.86141546, 0.57531717, 1.81793969, -0.29648091, 1.52771669,
                                  2.10872189, 0.7085359, -0.26967352, 0.31890205, 0.8638752,
                                  -2.28414718, 0.33684998])

        def objective(x):
            return expr.bind_parameters(dict(zip(parameters, x))).eval().real

        settings = {'maxiter': 40,
                    'blocking': True,
                    'allowed_increase': 0,
                    'learning_rate': 0.1,
                    'perturbation': 0.1}
        if second_order:
            settings['second_order'] = True
            settings['regularization'] = 0.01

        spsa = SPSA(**settings)

        result = spsa.optimize(circuit.num_parameters, objective, initial_point=initial_point)

        with self.subTest('check final accuracy'):
            self.assertLess(result[1], -0.8)  # final loss

        with self.subTest('check number of function calls'):
            expected = 201 if second_order else 121
            self.assertEqual(result[2], expected)  # function evaluations

    def test_recalibrate_at_optimize(self):
        """Test SPSA calibrates anew upon each optimization run, if no autocalibration is set."""
        def objective(x):
            return -(x ** 2)

        spsa = SPSA(maxiter=1)
        _ = spsa.optimize(1, objective, initial_point=np.array([0.5]))

        self.assertIsNone(spsa.learning_rate)
        self.assertIsNone(spsa.perturbation)
