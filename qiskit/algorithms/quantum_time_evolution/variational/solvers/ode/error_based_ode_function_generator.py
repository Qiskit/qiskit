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
from typing import Union, List, Dict, Optional

import numpy as np
from scipy.optimize import minimize

from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.gradient_errors.error_calculator import (
    ErrorCalculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.var_qte_linear_solver import (
    VarQteLinearSolver,
)
from qiskit.circuit import Parameter
from qiskit.opflow import CircuitSampler
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class ErrorBaseOdeFunctionGenerator:
    def __init__(
        self,
        error_calculator: ErrorCalculator,
        param_dict: Dict[Parameter, Union[float, complex]],
        variational_principle: VariationalPrinciple,
        grad_circ_sampler: CircuitSampler,
        metric_circ_sampler: CircuitSampler,
        nat_grad_circ_sampler: CircuitSampler,
        regularization: Optional[str] = None,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
    ):
        self._error_calculator = error_calculator
        self._param_dict = param_dict
        self._variational_principle = variational_principle
        self._regularization = regularization
        self._backend = backend
        self._grad_circ_sampler = grad_circ_sampler
        self._metric_circ_sampler = metric_circ_sampler
        self._nat_grad_circ_sampler = nat_grad_circ_sampler
        self._linear_solver = VarQteLinearSolver(
            self._grad_circ_sampler,
            self._metric_circ_sampler,
            self._nat_grad_circ_sampler,
            self._regularization,
            self._backend,
        )

    def error_based_ode_fun(self):
        nat_grad_res, grad_res, metric_res = self._linear_solver._solve_sle(
            self._variational_principle, self._param_dict
        )

        def argmin_fun(dt_param_values: Union[List, np.ndarray]) -> float:
            """
            Search for the dω/dt which minimizes ||e_t||^2
            Args:
                dt_param_values: values for dω/dt
            Returns:
                ||e_t||^2 for given for dω/dt
            """
            (et_squared) = self._error_calculator._calc_single_step_error(
                dt_param_values, grad_res, metric_res
            )[0]

            return et_squared

        # return nat_grad_result
        # Use the natural gradient result as initial point for least squares solver
        # print('initial natural gradient result', nat_grad_result)
        argmin = minimize(fun=argmin_fun, x0=nat_grad_res, method="COBYLA", tol=1e-6)
        # argmin = sp.optimize.least_squares(fun=argmin_fun, x0=nat_grad_result, ftol=1e-6)

        print("final dt_omega", np.real(argmin.x))
        # self._et = argmin_fun(argmin.x)
        return argmin.x, grad_res, metric_res
