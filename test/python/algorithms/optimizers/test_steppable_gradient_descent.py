# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
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
import random
import numpy as np
from qiskit.algorithms.optimizers import GradientDescent

# from qiskit.circuit.library import PauliTwoDesign
# from qiskit.opflow import I, Z, StateFn
# from qiskit.test.decorators import slow_test


class TestGradientDescent(QiskitAlgorithmsTestCase):
    """Tests for the gradient descent optimizer."""

    def setUp(self):
        super().setUp()
        np.random.seed(12)

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

        result = optimizer.minimize(fun=objective, x0=initial_point, jac=grad)

        self.assertLess(result.fun, 1e-5)

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
                evaluated_gradient = grad(ask_object.x_center)
                optimizer._state.njev += 1

            tell_object = GD_TellObject(gradient=evaluated_gradient)
            optimizer.tell(ask_object=ask_object, tell_object=tell_object)

        result = optimizer.create_result()
        print(result.njev, result.nfev)
        self.assertLess(result.fun, tol)
