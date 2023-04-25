# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Forward Euler solver."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np
from ddt import ddt, data, unpack
from scipy.integrate import solve_ivp

from qiskit.algorithms.time_evolvers.variational.solvers.ode.forward_euler_solver import (
    ForwardEulerSolver,
)


@ddt
class TestForwardEulerSolver(QiskitAlgorithmsTestCase):
    """Test Forward Euler solver."""

    @unpack
    @data((4, 16), (16, 35.52713678800501), (320, 53.261108839604795))
    def test_solve(self, timesteps, expected_result):
        """Test Forward Euler solver for a simple ODE."""

        y0 = [1]

        # pylint: disable=unused-argument
        def func(time, y):
            return y

        t_span = [0.0, 4.0]
        sol1 = solve_ivp(func, t_span, y0, method=ForwardEulerSolver, num_t_steps=timesteps)
        np.testing.assert_equal(sol1.y[-1][-1], expected_result)


if __name__ == "__main__":
    unittest.main()
