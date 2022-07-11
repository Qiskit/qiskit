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


from qiskit.algorithms.optimizers import GradientDescent, GradientDescentState
from qiskit.algorithms.optimizers.steppable_optimizer import TellData
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

        result = optimizer.minimize(objective, x0=initial_point)

        self.assertLess(result.fun, -0.95)  # final loss
        self.assertEqual(result.nfev, 100)  # function evaluations

    def test_callback(self):
        """Test the callback."""

        history = []

        def callback(*args):
            history.append(args)

        optimizer = GradientDescent(maxiter=1, callback=callback)

        def objective(x):
            return np.linalg.norm(x)

        _ = optimizer.minimize(objective, np.array([1, -1]))

        self.assertEqual(len(history), 1)
        self.assertIsInstance(history[0][0], int)  # nfevs
        self.assertIsInstance(history[0][1], np.ndarray)  # parameters
        self.assertIsInstance(history[0][2], float)  # function value
        self.assertIsInstance(history[0][3], float)  # norm of the gradient

    def test_random_failure(self):
        """
        Performs one optimization step where the first function evaluation fails.
        When making an evaluation in a quantum circuit, we might get a failure instead of a result.
        In this case, the failure will be modeled by the function returning a None.
        """

        def objective(x):
            return (np.linalg.norm(x) - 1) ** 2

        def grad(x, njev):
            choices = [True, True, False]
            if choices[njev]:
                return None
            else:
                return 2 * (np.linalg.norm(x) - 1) * x / np.linalg.norm(x)

        dimension = 3
        initial_point = np.ones(dimension)

        optimizer = GradientDescent(maxiter=20, learning_rate=0.1)
        optimizer.start(x0=initial_point, fun=objective, jac=grad)

        ask_data = optimizer.ask()
        evaluated_gradient = None
        while evaluated_gradient is None:
            evaluated_gradient = grad(ask_data.x_jac, optimizer.state.njev)
            optimizer.state.njev += 1

        tell_data = TellData(eval_jac=evaluated_gradient)
        optimizer.tell(ask_data=ask_data, tell_data=tell_data)

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

    # ask,tell,step,evaluate,start,minimize,continue_condition,create_result,

    def test_start(self):
        """Tests if the start method initializes the state properly."""

        def objective(x):
            return (np.linalg.norm(x) - 1) ** 2

        initial_point = np.ones(3)

        optimizer = GradientDescent()

        self.assertIsNone(optimizer.state)
        self.assertIsNone(optimizer.perturbation)
        optimizer.start(x0=initial_point, fun=objective)

        test_state = GradientDescentState(
            x=initial_point,
            fun=objective,
            jac=None,
            nfev=0,
            njev=0,
            nit=0,
            learning_rate=None,
            stepsize=None,
        )

    def test_learning_rate(self):
        """
        Tests if the learning rate is initialized properly for each kind of input:
        float, list and iterator.
        """
        def objective(x):
            return (np.linalg.norm(x) - 1) ** 2

        constant_learning_rate = 0.01
        list_learning_rate = [0.01 * n for n in range(10)]
        generator_learning_rate = (el for el in list_learning_rate)

        initial_point = np.ones(3)
        optimizer = GradientDescent()

        with self.subTest("Check constant learning rate."):
            optimizer.learning_rate = constant_learning_rate
            optimizer.start(x0=initial_point, fun=objective)
            for _ in range(5):
                self.assertEqual(constant_learning_rate,next(optimizer.state.learning_rate))

        with self.subTest("Check learning rate list."):
            optimizer.learning_rate = list_learning_rate
            optimizer.start(x0=initial_point, fun=objective)
            for i in range(5):
                self.assertEqual(list_learning_rate[i],next(optimizer.state.learning_rate))

        with self.subTest("Check learning rate generator."):
            optimizer.learning_rate = lambda : generator_learning_rate
            optimizer.start(x0=initial_point, fun=objective)
            for i in range(5):
                self.assertEqual(list_learning_rate[i],next(optimizer.state.learning_rate))


