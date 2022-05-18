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

"""Class for solving linear equations for Quantum Time Evolution."""

from typing import Union, List, Dict, Optional, Callable

import numpy as np
from qiskit.circuit import Parameter
from qiskit.opflow import CircuitSampler, OperatorBase
from ..calculators.metric_tensor_calculator import (
    eval_metric_result,
)
from ..calculators.evolution_grad_calculator import (
    eval_grad_result,
)


class VarQTELinearSolver:
    """Class for solving linear equations for Quantum Time Evolution."""

    def __init__(
        self,
        metric_tensor: OperatorBase,
        evolution_grad: OperatorBase,
        lse_solver_callable: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.linalg.lstsq,
        circuit_sampler: Optional[CircuitSampler] = None,
        imag_part_tol: float = 1e-7,
    ) -> None:
        """
        Args:
            metric_tensor: A parametrized operator that represents the left-hand side of an ODE.
            evolution_grad: A parametrized operator that represents the right-hand side of an ODE.
            lse_solver_callable: Linear system of equations solver that follows a NumPy
                ``np.linalg.lstsq`` interface.
            circuit_sampler: Samples circuits using an underlying backend.
            imag_part_tol: Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.
        """

        self._metric_tensor = metric_tensor
        self._evolution_grad = evolution_grad
        self._lse_solver_callable = lse_solver_callable
        self._circuit_sampler = circuit_sampler
        self._imag_part_tol = imag_part_tol

    def solve_sle(
        self,
        param_dict: Dict[Parameter, complex],
        t_param: Optional[Parameter] = None,
        time_value: Optional[float] = None,
    ) -> (Union[List, np.ndarray], Union[List, np.ndarray], np.ndarray):
        """
        Solve the system of linear equations underlying McLachlan's variational principle for the
        calculation without error bounds.

        Args:
            param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            time_value: Time value that will be bound to t_param. It is required if ``t_param`` is
                not ``None``.

        Returns:
            Solution to the LSE, A from Ax=b, b from Ax=b.
        """

        metric_tensor_lse_lhs = self._calc_lse_lhs(param_dict, t_param, time_value)
        evolution_grad_lse_rhs = self._calc_lse_rhs(param_dict, t_param, time_value)

        x = self._lse_solver_callable(metric_tensor_lse_lhs, evolution_grad_lse_rhs)[0]

        return np.real(x), metric_tensor_lse_lhs, evolution_grad_lse_rhs

    def _calc_lse_lhs(
        self,
        param_dict: Dict[Parameter, complex],
        t_param: Optional[Parameter] = None,
        time_value: Optional[float] = None,
    ) -> OperatorBase:

        metric = self._metric_tensor

        if t_param is not None:
            time_dict = {t_param: time_value}
            metric = self._metric_tensor.bind_parameters(time_dict)

        metric_tensor_lse_lhs = eval_metric_result(
            metric,
            param_dict,
            self._circuit_sampler,
        )

        return metric_tensor_lse_lhs

    def _calc_lse_rhs(
        self,
        param_dict: Dict[Parameter, complex],
        t_param: Optional[Parameter] = None,
        time_value: Optional[float] = None,
    ) -> OperatorBase:

        grad = self._evolution_grad

        if t_param is not None:
            time_dict = {t_param: time_value}
            grad = self._evolution_grad.bind_parameters(time_dict)

        evolution_grad_lse_rhs = eval_grad_result(
            grad,
            param_dict,
            self._circuit_sampler,
            self._imag_part_tol,
        )

        return evolution_grad_lse_rhs
