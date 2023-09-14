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
"""Forward Euler ODE solver."""
from collections.abc import Callable, Sequence

import numpy as np
from scipy.integrate import OdeSolver
from scipy.integrate._ivp.base import ConstantDenseOutput


class ForwardEulerSolver(OdeSolver):
    """Forward Euler ODE solver."""

    def __init__(
        self,
        function: Callable,
        t0: float,
        y0: Sequence,
        t_bound: float,
        vectorized: bool = False,
        support_complex: bool = False,
        num_t_steps: int = 15,
    ):
        """
        Forward Euler ODE solver that implements an interface from SciPy.

        Args:
            function: Right-hand side of the system. The calling signature is ``fun(t, y)``. Here
                ``t`` is a scalar, and there are two options for the ndarray ``y``:
                It can either have shape (n,); then ``fun`` must return array_like with
                shape (n,). Alternatively it can have shape (n, k); then ``fun``
                must return an array_like with shape (n, k), i.e., each column
                corresponds to a single column in ``y``. The choice between the two
                options is determined by `vectorized` argument (see below). The
                vectorized implementation allows a faster approximation of the Jacobian
                by finite differences (required for this solver).
            t0: Initial time.
            y0: Initial state.
            t_bound: Boundary time - the integration won't continue beyond it. It also determines
                the direction of the integration.
            vectorized: Whether ``fun`` is implemented in a vectorized fashion. Default is False.
            support_complex: Whether integration in a complex domain should be supported.
                Generally determined by a derived solver class capabilities. Default is False.
            num_t_steps: Number of time steps for the forward Euler method.
        """
        self._y_old = None
        self._step_length = (t_bound - t0) / num_t_steps
        super().__init__(function, t0, y0, t_bound, vectorized, support_complex)

    def _step_impl(self):
        """
        Takes an Euler step.
        """
        try:
            self._y_old = self.y
            self.y = list(np.add(self.y, self._step_length * self.fun(self.t, self.y)))
            self.t += self._step_length
            return True, None
        except Exception as ex:  # pylint: disable=broad-except
            return False, f"Unknown ODE solver error: {str(ex)}."

    def _dense_output_impl(self):
        return ConstantDenseOutput(self.t_old, self.t, self._y_old)
