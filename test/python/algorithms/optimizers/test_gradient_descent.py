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

"""Tests for the Gradient Descent optimizer."""

from test.python.algorithms import QiskitAlgorithmsTestCase

import numpy as np
import random

from qiskit.algorithms.optimizers import GradientDescent
from qiskit.algorithms.optimizers.steppable_optimizer import TellObject
from qiskit.circuit.library import PauliTwoDesign
from qiskit.opflow import I, Z, StateFn
from qiskit.test.decorators import slow_test


class TestGradientDescent(QiskitAlgorithmsTestCase):
    """Tests for the gradient descent optimizer."""

    def setUp(self):
        super().setUp()
        np.random.seed(12)

    @slow_test
    def test_pauli_two_design(self):
        """Test standard gradient descent on the Pauli two-design example."""
        circuit = PauliTwoDesign(3, reps=3, seed=2)
        parameters = list(circuit.parameters)
        obs = Z ^ Z ^ I
        expr = ~StateFn(obs) @ StateFn(circuit)

        initial_point = np.array(
            [
                0.1822308,
                -0.27254251,
                0.83684425,
                0.86153976,
                -0.7111668,
                0.82766631,
                0.97867993,
                0.46136964,
                2.27079901,
                0.13382699,
                0.29589915,
                0.64883193,
            ]
        )

        def objective(x):
            return expr.bind_parameters(dict(zip(parameters, x))).eval().real

        optimizer = GradientDescent(maxiter=100, learning_rate=0.1, perturbation=0.1)

        result = optimizer.minimize(circuit.num_parameters, objective, initial_point=initial_point)

        self.assertLess(result[1], -0.95)  # final loss
        self.assertEqual(result[2], 100)  # function evaluations

    # def test_callback(self):
    #     """Test the callback."""

    #     history = []

    #     def callback(*args):
    #         history.append(args)

    #     optimizer = GradientDescent(maxiter=1, callback=callback)

    #     def objective(x):
    #         return np.linalg.norm(x)

    #     _ = optimizer.minimize(objective, np.array([1, -1]))

    #     self.assertEqual(len(history), 1)
    #     self.assertIsInstance(history[0][0], int)  # nfevs
    #     self.assertIsInstance(history[0][1], np.ndarray)  # parameters
    #     self.assertIsInstance(history[0][2], float)  # function value
    #     self.assertIsInstance(history[0][3], float)  # norm of the gradient

    def test_random_failure(self):
        def learning_rate():
            power = 0.6
            constant_coeff = 0.1

            def powerlaw():
                n = 0
                while True:
                    yield constant_coeff * (n**power)
                    n += 1

            return powerlaw()

        def objective(x):
            return (np.linalg.norm(x) - 1) ** 2

        def grad(x):
            if random.choice([True, False]):
                return None
            else:
                return 2 * (np.linalg.norm(x) - 1) * x / np.linalg.norm(x)

        tol = 1e-4
        N = 100
        initial_point = np.random.normal(0, 1, size=(N,))

        optimizer = GradientDescent(maxiter=20, learning_rate=learning_rate)
        optimizer.initialize(x0=initial_point, fun=objective, jac=grad)

        for _ in range(20):
            ask_object = optimizer.ask()
            evaluated_gradient = None

            while evaluated_gradient is None:
                evaluated_gradient = grad(ask_object.x_jac)
                optimizer._state.njev += 1

            tell_object = TellObject(eval_jac=evaluated_gradient)
            optimizer.tell(ask_object=ask_object, tell_object=tell_object)

        result = optimizer.create_result()
        print(result.njev, result.nfev)
        self.assertLess(result.fun, tol)

    def test_iterator_learning_rate(self):
        """Test setting the learning rate as iterator."""

        def learning_rate():
            power = 0.6
            constant_coeff = 0.1

            def powerlaw():
                n = 0
                while True:
                    yield constant_coeff * (n**power)
                    n += 1

            return powerlaw()

        def objective(x):
            return (np.linalg.norm(x) - 1) ** 2

        def grad(x):
            return 2 * (np.linalg.norm(x) - 1) * x / np.linalg.norm(x)

        initial_point = np.array([1, 0.5, -2])

        optimizer = GradientDescent(maxiter=20, learning_rate=learning_rate)
        result = optimizer.minimize(objective, initial_point, grad)

        self.assertLess(result.fun, 1e-5)
