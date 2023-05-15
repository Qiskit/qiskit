# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2023.
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
from qiskit.algorithms.optimizers.steppable_optimizer import TellData, AskData
from qiskit.circuit.library import PauliTwoDesign
from qiskit.opflow import I, Z, StateFn


class TestGradientDescent(QiskitAlgorithmsTestCase):
    """Tests for the gradient descent optimizer."""

    def setUp(self):
        super().setUp()
        np.random.seed(12)
        self.initial_point = np.array([1, 1, 1, 1, 0])

    def objective(self, x):
        """Objective Function for the tests"""
        return (np.linalg.norm(x) - 1) ** 2

    def grad(self, x):
        """Gradient of the objective function"""
        return 2 * (np.linalg.norm(x) - 1) * x / np.linalg.norm(x)

    def test_pauli_two_design(self):
        """Test standard gradient descent on the Pauli two-design example."""
        circuit = PauliTwoDesign(3, reps=3, seed=2)
        parameters = list(circuit.parameters)
        with self.assertWarns(DeprecationWarning):
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

        def objective_pauli(x):
            return expr.bind_parameters(dict(zip(parameters, x))).eval().real

        optimizer = GradientDescent(maxiter=100, learning_rate=0.1, perturbation=0.1)

        with self.assertWarns(DeprecationWarning):
            result = optimizer.minimize(objective_pauli, x0=initial_point)
        self.assertLess(result.fun, -0.95)  # final loss
        self.assertEqual(result.nfev, 1300)  # function evaluations

    def test_callback(self):
        """Test the callback."""

        history = []

        def callback(*args):
            history.append(args)

        optimizer = GradientDescent(maxiter=1, callback=callback)

        _ = optimizer.minimize(self.objective, np.array([1, -1]))

        self.assertEqual(len(history), 1)
        self.assertIsInstance(history[0][0], int)  # nfevs
        self.assertIsInstance(history[0][1], np.ndarray)  # parameters
        self.assertIsInstance(history[0][2], float)  # function value
        self.assertIsInstance(history[0][3], float)  # norm of the gradient

    def test_minimize(self):
        """Test setting the learning rate as iterator and minimizing the funciton."""

        def learning_rate():
            power = 0.6
            constant_coeff = 0.1

            def powerlaw():
                n = 0
                while True:
                    yield constant_coeff * (n**power)
                    n += 1

            return powerlaw()

        optimizer = GradientDescent(maxiter=20, learning_rate=learning_rate)
        result = optimizer.minimize(self.objective, self.initial_point, self.grad)

        self.assertLess(result.fun, 1e-5)

    def test_no_start(self):
        """Tests that making a step without having started the optimizer raises an error."""
        optimizer = GradientDescent()
        with self.assertRaises(AttributeError):
            optimizer.step()

    def test_start(self):
        """Tests if the start method initializes the state properly."""
        optimizer = GradientDescent()
        self.assertIsNone(optimizer.state)
        self.assertIsNone(optimizer.perturbation)
        optimizer.start(x0=self.initial_point, fun=self.objective)

        test_state = GradientDescentState(
            x=self.initial_point,
            fun=self.objective,
            jac=None,
            nfev=0,
            njev=0,
            nit=0,
            learning_rate=1,
            stepsize=None,
        )

        self.assertEqual(test_state, optimizer.state)

    def test_ask(self):
        """Test the ask method."""
        optimizer = GradientDescent()
        optimizer.start(fun=self.objective, x0=self.initial_point)

        ask_data = optimizer.ask()
        np.testing.assert_equal(ask_data.x_jac, self.initial_point)
        self.assertIsNone(ask_data.x_fun)

    def test_evaluate(self):
        """Test the evaluate method."""
        optimizer = GradientDescent(perturbation=1e-10)
        optimizer.start(fun=self.objective, x0=self.initial_point)
        ask_data = AskData(x_jac=self.initial_point)
        tell_data = optimizer.evaluate(ask_data=ask_data)
        np.testing.assert_almost_equal(tell_data.eval_jac, self.grad(self.initial_point), decimal=2)

    def test_tell(self):
        """Test the tell method."""
        optimizer = GradientDescent(learning_rate=1.0)
        optimizer.start(fun=self.objective, x0=self.initial_point)
        ask_data = AskData(x_jac=self.initial_point)
        tell_data = TellData(eval_jac=self.initial_point)
        optimizer.tell(ask_data=ask_data, tell_data=tell_data)
        np.testing.assert_equal(optimizer.state.x, np.zeros(optimizer.state.x.shape))

    def test_continue_condition(self):
        """Test if the continue condition is working properly."""
        optimizer = GradientDescent(tol=1)
        optimizer.start(fun=self.objective, x0=self.initial_point)
        self.assertTrue(optimizer.continue_condition())
        optimizer.state.stepsize = 0.1
        self.assertFalse(optimizer.continue_condition())
        optimizer.state.stepsize = 10
        optimizer.state.nit = 1000
        self.assertFalse(optimizer.continue_condition())

    def test_step(self):
        """Tests if performing one step yields the desired result."""
        optimizer = GradientDescent(learning_rate=1.0)
        optimizer.start(fun=self.objective, jac=self.grad, x0=self.initial_point)
        optimizer.step()
        np.testing.assert_almost_equal(
            optimizer.state.x, self.initial_point - self.grad(self.initial_point), 6
        )

    def test_wrong_dimension_gradient(self):
        """Tests if an error is raised when a gradient of the wrong dimension is passed."""

        optimizer = GradientDescent(learning_rate=1.0)
        optimizer.start(fun=self.objective, x0=self.initial_point)
        ask_data = AskData(x_jac=self.initial_point)
        tell_data = TellData(eval_jac=np.array([1.0, 5]))
        with self.assertRaises(ValueError):
            optimizer.tell(ask_data=ask_data, tell_data=tell_data)

        tell_data = TellData(eval_jac=np.array(1))
        with self.assertRaises(ValueError):
            optimizer.tell(ask_data=ask_data, tell_data=tell_data)
