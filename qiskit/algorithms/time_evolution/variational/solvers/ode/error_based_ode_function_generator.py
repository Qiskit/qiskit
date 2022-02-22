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

from typing import Union, List, Iterable

import numpy as np

from qiskit.algorithms.time_evolution.variational.solvers.ode.abstract_ode_function_generator \
    import (
    AbstractOdeFunctionGenerator,
)


class ErrorBasedOdeFunctionGenerator(AbstractOdeFunctionGenerator):
    """Class for generating error-based ODE functions."""

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
        argmin = self._optimizer.minimize(
            fun=argmin_fun, x0=nat_grad_res
        )

        # self._et = argmin_fun(argmin.x)
        return argmin.x
