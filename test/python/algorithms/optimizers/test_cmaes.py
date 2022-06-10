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
from qiskit.algorithms.optimizers import SteppableCMAES


class TestCMAES(QiskitAlgorithmsTestCase):
    """Tests for the CMAES optimizer."""

    def setUp(self):
        super().setUp()
        np.random.seed(12)

    def test_cmaes(self):
        def objective(x):
            return (np.linalg.norm(x) - 1) ** 2

        N = 5
        tol = 1e-3
        initial_point = np.random.normal(0, 1, size=(N,))

        optimizer = SteppableCMAES(maxiter=1000)
        optimizer.initialize(x0=initial_point, fun=objective, tol=tol)

        result = optimizer.minimize(fun=objective, x0=initial_point)

        result = optimizer.create_result()
        self.assertLess(result.fun, tol)

    def test_random_failure(self):
        """Tests the case where the function evaluation has a probability of failing"""

        def objective(x):
            return (np.linalg.norm(x) - 1) ** 2

        def objective_fail(x):
            if random.choice([True, False]):
                return None
            else:
                return objective(x)

        N = 5
        tol = 1e-3
        initial_point = np.random.normal(0, 1, size=(N,))

        optimizer = SteppableCMAES(maxiter=40)
        optimizer.initialize(x0=initial_point, fun=objective, tol=tol)

        for _ in range(optimizer.maxiter):
            ask_object = optimizer.ask()
            eval_fun = []
            for x in ask_object.x_fun:
                feval_try = None
                while not feval_try:
                    feval_try = objective_fail(x)
                    optimizer._state.nfev += 1
                eval_fun.append(feval_try)
            tell_object = optimizer.user_evaluate(eval_fun=eval_fun)
            optimizer.tell(ask_object=ask_object, tell_object=tell_object)

            if optimizer.stop_condition():
                break
        result = optimizer.create_result()
        self.assertLess(result.fun, tol)
