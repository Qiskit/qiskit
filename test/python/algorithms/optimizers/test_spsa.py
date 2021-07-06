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

from qiskit.algorithms.optimizers import SPSA, QNSPSA
from qiskit.circuit.library import PauliTwoDesign
from qiskit.opflow import I, Z, StateFn
from qiskit.utils import algorithm_globals


@ddt
class TestSPSA(QiskitAlgorithmsTestCase):
    """Tests for the SPSA optimizer."""

    def setUp(self):
        super().setUp()
        np.random.seed(12)
        algorithm_globals.random_seed = 12

    # @slow_test
    @data("spsa", "2spsa", "qnspsa")
    def test_pauli_two_design(self, method):
        """Test SPSA on the Pauli two-design example."""
        circuit = PauliTwoDesign(3, reps=1, seed=1)
        parameters = list(circuit.parameters)
        obs = Z ^ Z ^ I
        expr = ~StateFn(obs) @ StateFn(circuit)

        initial_point = np.array(
            [0.82311034, 0.02611798, 0.21077064, 0.61842177, 0.09828447, 0.62013131]
        )

        def objective(x):
            return expr.bind_parameters(dict(zip(parameters, x))).eval().real

        settings = {"maxiter": 100, "blocking": True, "allowed_increase": 0}

        if method == "2spsa":
            settings["second_order"] = True
            settings["regularization"] = 0.01
            expected_nfev = settings["maxiter"] * 5 + 1
        elif method == "qnspsa":
            settings["fidelity"] = QNSPSA.get_fidelity(circuit)
            settings["regularization"] = 0.001
            settings["learning_rate"] = 0.05
            settings["perturbation"] = 0.05

            expected_nfev = settings["maxiter"] * 7 + 1
        else:
            expected_nfev = settings["maxiter"] * 3 + 1

        if method == "qnspsa":
            spsa = QNSPSA(**settings)
        else:
            spsa = SPSA(**settings)

        result = spsa.optimize(circuit.num_parameters, objective, initial_point=initial_point)

        with self.subTest("check final accuracy"):
            self.assertLess(result[1], -0.95)  # final loss

        with self.subTest("check number of function calls"):
            self.assertEqual(result[2], expected_nfev)  # function evaluations

    def test_recalibrate_at_optimize(self):
        """Test SPSA calibrates anew upon each optimization run, if no autocalibration is set."""

        def objective(x):
            return -(x ** 2)

        spsa = SPSA(maxiter=1)
        _ = spsa.optimize(1, objective, initial_point=np.array([0.5]))

        self.assertIsNone(spsa.learning_rate)
        self.assertIsNone(spsa.perturbation)

    def test_learning_rate_perturbation_as_iterators(self):
        """Test the learning rate and perturbation can be callables returning iterators."""

        def get_learning_rate():
            def learning_rate():
                x = 0.99
                while True:
                    x *= x
                    yield x

            return learning_rate

        def get_perturbation():
            def perturbation():
                x = 0.99
                while True:
                    x *= x
                    yield max(x, 0.01)

            return perturbation

        def objective(x):
            return (np.linalg.norm(x) - 2) ** 2

        spsa = SPSA(learning_rate=get_learning_rate(), perturbation=get_perturbation())
        result, _, _ = spsa.optimize(1, objective, initial_point=np.array([0.5, 0.5]))

        self.assertAlmostEqual(np.linalg.norm(result), 2, places=2)

    def test_learning_rate_perturbation_as_arrays(self):
        """Test the learning rate and perturbation can be arrays."""

        learning_rate = np.linspace(1, 0, num=100, endpoint=False) ** 2
        perturbation = 0.01 * np.ones(100)

        def objective(x):
            return (np.linalg.norm(x) - 2) ** 2

        spsa = SPSA(learning_rate=learning_rate, perturbation=perturbation)
        result, _, _ = spsa.optimize(1, objective, initial_point=np.array([0.5, 0.5]))

        self.assertAlmostEqual(np.linalg.norm(result), 2, places=2)

    def test_callback(self):
        """Test using the callback."""

        def objective(x):
            return (np.linalg.norm(x) - 2) ** 2

        history = {"nfevs": [], "points": [], "fvals": [], "updates": [], "accepted": []}

        def callback(nfev, point, fval, update, accepted):
            history["nfevs"].append(nfev)
            history["points"].append(point)
            history["fvals"].append(fval)
            history["updates"].append(update)
            history["accepted"].append(accepted)

        maxiter = 10
        spsa = SPSA(maxiter=maxiter, learning_rate=0.01, perturbation=0.01, callback=callback)
        _ = spsa.optimize(1, objective, initial_point=np.array([0.5, 0.5]))

        expected_types = [int, np.ndarray, float, float, bool]
        for i, (key, values) in enumerate(history.items()):
            self.assertTrue(all(isinstance(value, expected_types[i]) for value in values))
            self.assertEqual(len(history[key]), maxiter)
