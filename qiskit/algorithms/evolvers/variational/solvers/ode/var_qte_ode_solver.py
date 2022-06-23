# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class for solving ODEs for Quantum Time Evolution."""

import itertools
from typing import List, Union

from numpy import ndarray
from scipy.integrate import OdeSolver, solve_ivp

from .abstract_ode_function import (
    AbstractOdeFunction,
)
from .forward_euler_solver import ForwardEulerSolver


class VarQTEOdeSolver:
    """Class for solving ODEs for Quantum Time Evolution."""

    def __init__(
        self,
        init_params: List[complex],
        ode_function: AbstractOdeFunction,
        ode_solver: Union[OdeSolver, str] = ForwardEulerSolver,
    ) -> None:
        """
        Initialize ODE Solver.

        Args:
            init_params: Set of initial parameters for time 0.
            ode_function: Generates the ODE function.
            ode_solver: ODE solver callable that implements a SciPy ``OdeSolver`` interface or a
                string indicating a valid method offered by SciPy.
        """
        self._init_params = init_params
        self._ode_function = ode_function.var_qte_ode_function
        self._ode_solver = ode_solver

    def run(self, evolution_time: float) -> List[complex]:
        """
        Finds numerical solution with ODE Solver.

        Args:
            evolution_time: Evolution time.

        Returns:
            List of parameters found by an ODE solver for a given ODE function callable.
        """
        sol = solve_ivp(
            self._ode_function,
            (0, evolution_time),
            self._init_params,
            method=self._ode_solver,
            num_t_steps=25,
        )
        final_params_vals = [lst[-1] for lst in sol.y]

        return final_params_vals
