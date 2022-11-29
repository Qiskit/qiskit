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
from qiskit.primitives import Estimator, Sampler
from qiskit.providers.basicaer import StatevectorSimulatorPy
from qiskit.opflow import I, Z, StateFn, MatrixExpectation
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

        with self.assertWarns(DeprecationWarning):
            result = spsa.optimize(circuit.num_parameters, objective, initial_point=initial_point)

        with self.subTest("check final accuracy"):
            self.assertLess(result[1], -0.95)  # final loss

        with self.subTest("check number of function calls"):
            self.assertEqual(result[2], expected_nfev)  # function evaluations

    def test_recalibrate_at_optimize(self):
        """Test SPSA calibrates anew upon each optimization run, if no autocalibration is set."""

        def objective(x):
            return -(x**2)

        spsa = SPSA(maxiter=1)
        _ = spsa.minimize(objective, x0=np.array([0.5]))

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
        result = spsa.minimize(objective, np.array([0.5, 0.5]))

        self.assertAlmostEqual(np.linalg.norm(result.x), 2, places=2)

    def test_learning_rate_perturbation_as_arrays(self):
        """Test the learning rate and perturbation can be arrays."""

        learning_rate = np.linspace(1, 0, num=100, endpoint=False) ** 2
        perturbation = 0.01 * np.ones(100)

        def objective(x):
            return (np.linalg.norm(x) - 2) ** 2

        spsa = SPSA(learning_rate=learning_rate, perturbation=perturbation)
        result = spsa.minimize(objective, x0=np.array([0.5, 0.5]))

        self.assertAlmostEqual(np.linalg.norm(result.x), 2, places=2)

    def test_termination_checker(self):
        """Test the termination_callback"""

        def objective(x):
            return np.linalg.norm(x) + np.random.rand(1)

        class TerminationChecker:
            """Example termination checker"""

            def __init__(self):
                self.values = []

            def __call__(self, nfev, point, fvalue, stepsize, accepted) -> bool:
                self.values.append(fvalue)

                if len(self.values) > 10:
                    return True
                return False

        maxiter = 400
        spsa = SPSA(maxiter=maxiter, termination_checker=TerminationChecker())
        result = spsa.minimize(objective, x0=[0.5, 0.5])

        self.assertLess(result.nit, maxiter)

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
        _ = spsa.minimize(objective, x0=np.array([0.5, 0.5]))

        expected_types = [int, np.ndarray, float, float, bool]
        for i, (key, values) in enumerate(history.items()):
            self.assertTrue(all(isinstance(value, expected_types[i]) for value in values))
            self.assertEqual(len(history[key]), maxiter)

    @data(1, 2, 3, 4)
    def test_estimate_stddev(self, max_evals_grouped):
        """Test the estimate_stddev
        See https://github.com/Qiskit/qiskit-nature/issues/797"""

        def objective(x):
            if len(x.shape) == 2:
                return np.array([sum(x_i) for x_i in x])
            return sum(x)

        point = np.ones(5)
        result = SPSA.estimate_stddev(objective, point, avg=10, max_evals_grouped=max_evals_grouped)
        self.assertAlmostEqual(result, 0)

    def test_qnspsa_fidelity_deprecation(self):
        """Test using a backend and expectation converter in get_fidelity warns."""
        ansatz = PauliTwoDesign(2, reps=1, seed=2)

        with self.assertWarns(PendingDeprecationWarning):
            _ = QNSPSA.get_fidelity(ansatz, StatevectorSimulatorPy(), MatrixExpectation())

    def test_qnspsa_fidelity_primitives(self):
        """Test the primitives can be used in get_fidelity."""
        ansatz = PauliTwoDesign(2, reps=1, seed=2)
        initial_point = np.random.random(ansatz.num_parameters)

        with self.subTest(msg="pass as kwarg"):
            fidelity = QNSPSA.get_fidelity(ansatz, sampler=Sampler())
            result = fidelity(initial_point, initial_point)

            self.assertAlmostEqual(result[0], 1)

        # this test can be removed once backend and expectation are removed
        with self.subTest(msg="pass positionally"):
            fidelity = QNSPSA.get_fidelity(ansatz, Sampler())
            result = fidelity(initial_point, initial_point)

            self.assertAlmostEqual(result[0], 1)

    def test_qnspsa_max_evals_grouped(self):
        """Test using max_evals_grouped with QNSPSA."""
        circuit = PauliTwoDesign(3, reps=1, seed=1)
        num_parameters = circuit.num_parameters
        obs = Z ^ Z ^ I

        estimator = Estimator(options={"seed": 12})

        initial_point = np.array(
            [0.82311034, 0.02611798, 0.21077064, 0.61842177, 0.09828447, 0.62013131]
        )

        def objective(x):
            x = np.reshape(x, (-1, num_parameters)).tolist()
            n = len(x)
            return estimator.run(n * [circuit], n * [obs.primitive], x).result().values.real

        fidelity = QNSPSA.get_fidelity(circuit)
        optimizer = QNSPSA(fidelity)
        optimizer.maxiter = 1
        optimizer.learning_rate = 0.05
        optimizer.perturbation = 0.05
        optimizer.set_max_evals_grouped(50)  # greater than 1

        result = optimizer.minimize(objective, initial_point)

        with self.subTest("check final accuracy"):
            self.assertAlmostEqual(result.fun[0], 0.473, places=3)

        with self.subTest("check number of function calls"):
            expected_nfev = 8  # 7 * maxiter + 1
            self.assertEqual(result.nfev, expected_nfev)
