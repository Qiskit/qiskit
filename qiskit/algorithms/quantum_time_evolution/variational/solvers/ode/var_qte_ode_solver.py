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

from scipy.integrate import RK45, OdeSolver

from qiskit import QiskitError
from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode.abstract_ode_function_generator import (
    AbstractOdeFunctionGenerator,
)


class VarQteOdeSolver:
    def __init__(
        self,
        init_params,
        ode_function_generator: AbstractOdeFunctionGenerator,
        ode_solver_callable: OdeSolver = RK45,
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
        ode_solver = self._ode_solver_callable(
            self._ode_function,
            t_bound=evolution_time,
            t0=0,
            y0=self._init_params,  # appending 0 was here in case of bounds, not yet supported.
            vectorized=False,
        )
        param_values = None

        _ = ode_solver.fun(ode_solver.t, ode_solver.y)
        while ode_solver.t < evolution_time:
            ode_solver.step()
            if ode_solver.t <= evolution_time:
                _ = ode_solver.fun(ode_solver.t, ode_solver.y)
            param_values = ode_solver.y  # was ode_solver.y[:-1] if last element error bound
            # related, currently no error bound appended
            if ode_solver.status == "finished":
                break
            elif ode_solver.status == "failed":
                raise QiskitError("ODESolver failed.")

        return param_values
