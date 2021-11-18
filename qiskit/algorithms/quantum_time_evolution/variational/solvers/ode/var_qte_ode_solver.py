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
import itertools

from scipy.integrate import OdeSolver, solve_ivp

from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode.abstract_ode_function_generator import (
    AbstractOdeFunctionGenerator,
)


class VarQteOdeSolver:
    def __init__(
        self,
        init_params,
        ode_function_generator: AbstractOdeFunctionGenerator,
        ode_solver_callable: OdeSolver = "RK45",
    ):
        """
        Initialize ODE Solver
        Args:
            init_params: Set of initial parameters for time 0.
            ode_function_generator: Generator for a function that ODE will use.
            ode_solver_callable: ODE solver callable that follows a SciPy OdeSolver interface.
        """
        self._init_params = init_params
        self._ode_function = ode_function_generator.var_qte_ode_function
        self._ode_solver_callable = ode_solver_callable

    def _run(self, evolution_time: float):
        """
        Find numerical solution with ODE Solver.
        Args:
            evolution_time: Evolution time.
        """
        print(self._ode_function(0, self._init_params))
        print("******************")
        sol = solve_ivp(
            self._ode_function,
            (0, evolution_time),
            self._init_params,
            method=self._ode_solver_callable,
            t_eval=[evolution_time],
        )
        final_params_vals = list(itertools.chain(*sol.y))

        return final_params_vals
