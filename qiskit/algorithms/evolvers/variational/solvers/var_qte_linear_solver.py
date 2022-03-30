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

from qiskit.algorithms.evolvers.variational.calculators.metric_tensor_calculator import (
    eval_metric_result,
)
from qiskit.algorithms.evolvers.variational.calculators.evolution_grad_calculator import (
    eval_grad_result,
)
from qiskit.circuit import Parameter
from qiskit.opflow import CircuitSampler, OperatorBase


#  TODO this might potentially be exposed to the user to instantiate (only with
#   lse_solver_callable in the init; now the callable is passed by the user through VarQTE class)
class VarQTELinearSolver:
    """Class for solving linear equations for Quantum Time Evolution."""

    def __init__(
        self,
        metric_tensor: OperatorBase,
        evolution_grad: OperatorBase,
        lse_solver_callable: Callable[[np.ndarray, np.ndarray], np.ndarray] = np.linalg.lstsq,
        grad_circ_sampler: Optional[CircuitSampler] = None,
        metric_circ_sampler: Optional[CircuitSampler] = None,
        energy_sampler: Optional[CircuitSampler] = None,
        allowed_imaginary_part: float = 1e-7,
    ) -> None:
        """
        Args:
            metric_tensor: A parametrized operator that represents the left-hand side of an ODE.
            evolution_grad: A parametrized operator that represents the right-hand side of an ODE.
            lse_solver_callable: Linear system of equations solver that follows a NumPy
                ``np.linalg.lstsq`` interface.
            grad_circ_sampler: CircuitSampler for evolution gradients.
            metric_circ_sampler: CircuitSampler for metric tensors.
            energy_sampler: CircuitSampler for energy.
            allowed_imaginary_part: Allowed value of an imaginary part that can be neglected if no
                imaginary part is expected.
        """

        self._metric_tensor = metric_tensor
        self._evolution_grad = evolution_grad
        self._lse_solver_callable = lse_solver_callable
        self._grad_circ_sampler = grad_circ_sampler
        self._metric_circ_sampler = metric_circ_sampler
        self._energy_sampler = energy_sampler
        self._allowed_imaginary_part = allowed_imaginary_part

    def _solve_sle(
        self,
        param_dict: Dict[Parameter, Union[float, complex]],
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
            dω/dt, 2Re⟨dψ(ω)/dω|H|ψ(ω) for VarQITE/2Im⟨dψ(ω)/dω|H|ψ(ω) for VarQRTE, Fubini-Study
            Metric.
        """

        metric_tensor_lse_lhs = self._calc_lse_lhs(param_dict, t_param, time_value)
        evolution_grad_lse_rhs = self._calc_lse_rhs(param_dict, t_param, time_value)

        # TODO not all solvers will have rcond param. Keeping for now to keep the same results in
        #  unit tests.
        x = self._lse_solver_callable(metric_tensor_lse_lhs, evolution_grad_lse_rhs, rcond=1e-2)[0]

        return np.real(x), metric_tensor_lse_lhs, evolution_grad_lse_rhs

    def _calc_lse_lhs(
        self,
        param_dict: Dict[Parameter, Union[float, complex]],
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
            self._metric_circ_sampler,
        )

        return metric_tensor_lse_lhs

    def _calc_lse_rhs(
        self,
        param_dict: Dict[Parameter, Union[float, complex]],
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
            self._grad_circ_sampler,
            self._energy_sampler,
            self._allowed_imaginary_part,
        )

        return evolution_grad_lse_rhs
