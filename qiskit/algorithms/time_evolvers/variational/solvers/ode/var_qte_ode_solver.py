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

"""Class for solving ODEs for Quantum Time Evolution."""
from __future__ import annotations

from collections.abc import Sequence
from functools import partial
from typing import Type

import numpy as np
from scipy.integrate import OdeSolver, solve_ivp

from .abstract_ode_function import AbstractOdeFunction
from .forward_euler_solver import ForwardEulerSolver


class VarQTEOdeSolver:
    """Class for solving ODEs for Quantum Time Evolution."""

    def __init__(
        self,
        init_params: Sequence[float],
        ode_function: AbstractOdeFunction,
        ode_solver: Type[OdeSolver] | str = ForwardEulerSolver,
        num_timesteps: int | None = None,
    ) -> None:
        """
        Initialize ODE Solver.

        Args:
            init_params: Set of initial parameters for time 0.
            ode_function: Generates the ODE function.
            ode_solver: ODE solver callable that implements a SciPy ``OdeSolver`` interface or a
                string indicating a valid method offered by SciPy.
            num_timesteps: The number of timesteps to take. If None, it is
                automatically selected to achieve a timestep of approximately 0.01. Only
                relevant in case of the ``ForwardEulerSolver``.
        """
        self._init_params = init_params
        self._ode_function = ode_function.var_qte_ode_function
        self._ode_solver = ode_solver
        self._num_timesteps = num_timesteps

    def run(
        self, evolution_time: float
    ) -> tuple[Sequence[float], Sequence[Sequence[float]], Sequence[float]]:
        """
        Finds numerical solution with ODE Solver.

        Args:
            evolution_time: Evolution time.

        Returns:
            List of parameters found by an ODE solver for a given ODE function callable.
        """
        # determine the number of timesteps and set the timestep
        num_timesteps = (
            int(np.ceil(evolution_time / 0.01))
            if self._num_timesteps is None
            else self._num_timesteps
        )

        if self._ode_solver == ForwardEulerSolver:
            solve = partial(solve_ivp, num_t_steps=num_timesteps)
        else:
            solve = solve_ivp

        sol = solve(
            self._ode_function,
            (0, evolution_time),
            self._init_params,
            method=self._ode_solver,
        )

        param_vals = sol.y.T
        time_points = sol.t
        final_param_vals = param_vals[-1]

        return final_param_vals, param_vals, time_points
