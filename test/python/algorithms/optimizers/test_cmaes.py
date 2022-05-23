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
from qiskit.algorithms.optimizers.steppable_gradient_descent import GD_TellObject
from qiskit.algorithms.optimizers import SteppableCMAES

# from qiskit.circuit.library import PauliTwoDesign
# from qiskit.opflow import I, Z, StateFn
# from qiskit.test.decorators import slow_test


class TestCMAES(QiskitAlgorithmsTestCase):
    """Tests for the CMAES optimizer."""

    def setUp(self):
        super().setUp()
        np.random.seed(12)

    def test_cmaes(self):
        def objective(x):
            return (np.linalg.norm(x) - 1) ** 2
        N = 150
        initial_point = 0.01 * np.random.normal(0,1,size=(N,))

        optimizer = SteppableCMAES(maxiter=100)
        optimizer.initialize(x0=initial_point, fun=objective)
        print("InitialValue:",objective(initial_point))
        # print(optimizer.weights)
        # print(optimizer.mu)
        for _ in range(100):
            optimizer.step()
            # print(optimizer._state)

        # result = optimizer.minimize(fun=objective, x0=initial_point)
        result = optimizer.create_result()
        self.assertLess(result.fun, 1e-5)
