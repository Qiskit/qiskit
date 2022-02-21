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

"""Class for generating error-based ODE functions."""

from typing import Union, List, Iterable, Callable

import numpy as np
from scipy.optimize import minimize

from qiskit.algorithms.time_evolution.variational.solvers.ode.abstract_ode_function_generator \
    import (
    AbstractOdeFunctionGenerator,
)
from qiskit.algorithms.time_evolution.variational.solvers.var_qte_linear_solver import \
    VarQteLinearSolver


class ErrorBasedOdeFunctionGenerator(AbstractOdeFunctionGenerator):
    """Class for generating error-based ODE functions."""

    def __init__(
        self,
        lse_solver_callable: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.linalg.lstsq,
        optimizer: str = "COBYLA",
        optimizer_tolerance: float = 1e-6,
    ):
        """
        Args:
            lse_solver_callable: Linear system of equations solver that follows a NumPy
            np.linalg.lstsq interface.
            optimizer: Optimizer used in case error_based_ode is true.
            optimizer_tolerance: Numerical tolerance of an optimizer used for convergence to a
                minimum.
        """

        super().__init__(lse_solver_callable)

        self._error_calculator = None
        self._t_param = None

        self._optimizer = optimizer
        self._optimizer_tolerance = optimizer_tolerance

    def var_qte_ode_function(self, time: float, parameters_values: Iterable) -> float:
        """
        Evaluates an ODE function for a given time and parameter values. It is used by an ODE
        solver.

        Args:
            time: Current time of evolution.
            parameters_values: Current values of parameters.

        Returns:
            Tuple containing natural gradient, metric tensor and evolution gradient results
            arising from solving a system of linear equations.
        """
        current_param_dict = dict(zip(self._param_dict.keys(), parameters_values))

        nat_grad_res, metric_res, grad_res = self._varqte_linear_solver._solve_sle(
            self._lse_solver_callable,
            current_param_dict,
            self._t_param,
            time,
        )

        def argmin_fun(dt_param_values: Union[List, np.ndarray]) -> float:
            """
            Search for the dω/dt which minimizes ||e_t||^2.

            Args:
                dt_param_values: Values for dω/dt.

            Returns:
                ||e_t||^2 for given for dω/dt.
            """
            et_squared = self._error_calculator._calc_single_step_error(
                dt_param_values, grad_res, metric_res, current_param_dict
            )[0]

            return et_squared

        # Use the natural gradient result as initial point for least squares solver
        argmin = minimize(
            fun=argmin_fun, x0=nat_grad_res, method=self._optimizer, tol=self._optimizer_tolerance
        )

        # self._et = argmin_fun(argmin.x)
        return argmin.x
