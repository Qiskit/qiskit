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

"""Class for generating ODE functions based on natural gradients."""

from typing import Iterable

from qiskit.algorithms.time_evolution.variational.solvers.ode.abstract_ode_function_generator import (
    AbstractOdeFunctionGenerator,
)


class OdeFunctionGenerator(AbstractOdeFunctionGenerator):
    """Class for generating ODE functions based on natural gradients."""

    def var_qte_ode_function(self, time: float, parameters_values: Iterable) -> Iterable:
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

        nat_grad_res, _, _ = self._linear_solver._solve_sle(
            self._variational_principle,
            current_param_dict,
            self._t_param,
            time,
            self._regularization,
        )

        return nat_grad_res
