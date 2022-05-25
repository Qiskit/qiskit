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
from qiskit.algorithms.optimizers.cmaes import CMAES_TellObject
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

        # result = optimizer.minimize(fun=objective, x0=initial_point)

        for _ in range(optimizer.maxiter):
            optimizer.step()
            # print(optimizer._state)
            if optimizer.stop_condition():
                break

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

        N = 100
        tol = 1e-4
        initial_point = np.random.normal(0, 1, size=(N,))

        optimizer = SteppableCMAES(maxiter=1000)
        optimizer.initialize(x0=initial_point, fun=objective, tol=tol)

        for _ in range(optimizer.maxiter):
            ask_object = optimizer.ask()
            cloud_eval = []
            for x in ask_object.cloud:
                feval = None
                while not feval:
                    feval = objective_fail(x)
                    optimizer._state.nfev += 1
                cloud_eval.append(feval)
            tell_object = CMAES_TellObject(cloud_evaluated=cloud_eval)
            optimizer.tell(ask_object=ask_object, tell_object=tell_object)


            if optimizer.stop_condition():
                break
        print(optimizer._state)
        result = optimizer.create_result()
        self.assertLess(result.fun, tol)
