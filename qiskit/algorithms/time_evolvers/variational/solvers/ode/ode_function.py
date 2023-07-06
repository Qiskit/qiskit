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

"""Class for generating ODE functions based on ODE gradients."""
from collections.abc import Iterable

from .abstract_ode_function import AbstractOdeFunction


class OdeFunction(AbstractOdeFunction):
    """Class for generating ODE functions based on ODE gradients."""

    def var_qte_ode_function(self, time: float, parameter_values: Iterable) -> Iterable:
        """
        Evaluates an ODE function for a given time and parameter values. It is used by an ODE
        solver.

        Args:
            time: Current time of evolution.
            parameter_values: Current values of parameters.

        Returns:
            ODE gradient arising from solving a system of linear equations.
        """
        current_param_dict = dict(zip(self._param_dict.keys(), parameter_values))

        ode_grad_res, _, _ = self._varqte_linear_solver.solve_lse(
            current_param_dict,
            time,
        )

        return ode_grad_res
