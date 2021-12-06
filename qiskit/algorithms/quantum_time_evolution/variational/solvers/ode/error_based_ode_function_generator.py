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
"""Class for generating error-based ODE functions."""
from typing import Union, List, Dict, Optional, Iterable

import numpy as np
from scipy.optimize import minimize

from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.gradient_errors.error_calculator import (
    ErrorCalculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode.abstract_ode_function_generator import (
    AbstractOdeFunctionGenerator,
)
from qiskit.circuit import Parameter
from qiskit.opflow import CircuitSampler
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class ErrorBasedOdeFunctionGenerator(AbstractOdeFunctionGenerator):
    def __init__(
        self,
        error_calculator: ErrorCalculator,
        param_dict: Dict[Parameter, Union[float, complex]],
        variational_principle: VariationalPrinciple,
        grad_circ_sampler: CircuitSampler,
        metric_circ_sampler: CircuitSampler,
        energy_sampler: CircuitSampler,
        regularization: Optional[str] = None,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
        t_param: Parameter = None,
        optimizer: str = "COBYLA",
    ):
        super().__init__(
            param_dict,
            variational_principle,
            grad_circ_sampler,
            metric_circ_sampler,
            energy_sampler,
            regularization,
            backend,
            t_param,
        )
        self._error_calculator = error_calculator
        self._optimizer = optimizer

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

        nat_grad_res, metric_res, grad_res = self._linear_solver._solve_sle(
            self._variational_principle,
            current_param_dict,
            self._t_param,
            time,
            self._regularization,
        )

        def argmin_fun(dt_param_values: Union[List, np.ndarray]) -> float:
            """
            Search for the dω/dt which minimizes ||e_t||^2
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
        argmin = minimize(fun=argmin_fun, x0=nat_grad_res, method=self._optimizer, tol=1e-6)
        # argmin = sp.optimize.least_squares(fun=argmin_fun, x0=nat_grad_result, ftol=1e-6)

        print("final dt_omega", np.real(argmin.x))
        # self._et = argmin_fun(argmin.x)
        return argmin.x
