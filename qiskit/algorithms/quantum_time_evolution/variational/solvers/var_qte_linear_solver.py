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

from qiskit.algorithms.quantum_time_evolution.variational.calculators.tensor_evaluator import (
    eval_grad_result,
    eval_metric_result,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.opflow.gradients.natural_gradient import NaturalGradient
from qiskit.circuit import Parameter
from qiskit.opflow import CircuitSampler
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class VarQteLinearSolver:
    def __init__(
        self,
        grad_circ_sampler: CircuitSampler,
        metric_circ_sampler: CircuitSampler,
        energy_sampler: CircuitSampler,
        regularization: Optional[str] = None,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
    ):

        self._backend = backend
        self._regularization = regularization
        self._grad_circ_sampler = grad_circ_sampler
        self._metric_circ_sampler = metric_circ_sampler
        self._energy_sampler = energy_sampler

    def _solve_sle(
        self,
        var_principle: VariationalPrinciple,
        param_dict: Dict[Parameter, Union[float, complex]],
        t_param: Optional[Parameter] = None,
        time_value: Optional[float] = None,
        regularization: Optional[str] = None,
    ) -> (Union[List, np.ndarray], Union[List, np.ndarray], np.ndarray):
        """
        Solve the system of linear equations underlying McLachlan's variational principle for the
        calculation without error bounds.
        Args:
            var_principle: Variational Principle to be used.
            param_dict: Dictionary which relates parameter values to the parameters in the Ansatz.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            time_value: Time value that will be bound to t_param.
            regularization: Use the following regularization with a least square method to solve the
                underlying system of linear equations.
                Can be either None or ``'ridge'`` or ``'lasso'`` or ``'perturb_diag'``
                ``'ridge'`` and ``'lasso'`` use an automatic optimal parameter search,
                or a penalty term given as Callable.
                If regularization is None but the metric is ill-conditioned or singular then
                a least square solver is used without regularization.
        Returns: dω/dt, 2Re⟨dψ(ω)/dω|H|ψ(ω) for VarQITE/ 2Im⟨dψ(ω)/dω|H|ψ(ω) for VarQRTE,
                Fubini-Study Metric.
        """
        metric_result = eval_metric_result(
            var_principle._raw_metric_tensor,
            param_dict,
            self._metric_circ_sampler,
        )

        if t_param is not None:
            time_dict = {t_param: time_value}
            # TODO
            # Use var_principle to understand type and get gradient
            # bind parameters to H(t)
            # grad
            # natural gradient with nat_grad_combo_fn(x, regularization=None)
            # TODO raw_evolution_grad might be a callback
            grad = var_principle._raw_evolution_grad.bind_parameters(time_dict)
        else:
            grad = var_principle._raw_evolution_grad

        grad_result = eval_grad_result(
            grad, param_dict, self._grad_circ_sampler, self._energy_sampler
        )

        nat_grad_result = NaturalGradient().nat_grad_combo_fn(
            [grad_result, metric_result], regularization=regularization
        )
        return np.real(nat_grad_result)

    # # TODO update
    # def _solve_sle_for_error_bounds(
    #     self,
    #     var_principle: VariationalPrinciple,
    #     param_dict: Dict[Parameter, Union[float, complex]],
    #     t_param: Parameter = None,
    #     t: float = None,
    # ) -> (Union[List, np.ndarray], Union[List, np.ndarray], np.ndarray):
    #     """
    #     Solve the system of linear equations underlying McLachlan's variational principle for the
    #     calculation with error bounds.
    #     Args:
    #         var_principle: Variational Principle to be used.
    #         param_dict: Dictionary which relates parameter values to the parameters in the Ansatz.
    #         t_param: Time parameter in case of a time-dependent Hamiltonian.
    #         t: Time value that will be bound to t_param.
    #     Returns: dω/dt, 2Re⟨dψ(ω)/dω|H|ψ(ω) for VarQITE/ 2Im⟨dψ(ω)/dω|H|ψ(ω) for VarQRTE,
    #     Fubini-Study Metric.
    #     """
    #
    #     # bind time parameter for the current value of time from the ODE solver
    #
    #     # if t_param is not None:
    #     #     # TODO bind here
    #     #     time_dict = {t_param: t}
    #     #     evolution_grad = evolution_grad.bind_parameters(time_dict)
    #     grad_result = eval_grad_result(
    #         var_principle._raw_evolution_grad,
    #         param_dict,
    #         self._grad_circ_sampler,
    #         self._backend,
    #     )
    #     metric_result = eval_metric_result(
    #         var_principle._raw_metric_tensor,
    #         param_dict,
    #         self._metric_circ_sampler,
    #     )
    #
    #     self._inspect_imaginary_parts(grad_result, metric_result)
    #
    #     metric_res = np.real(metric_result)
    #     grad_res = np.real(grad_result)
    #
    #     # Check if numerical instabilities lead to a metric which is not positive semidefinite
    #     metric_res = self._check_and_fix_metric_psd(metric_res)
    #
    #     return grad_res, metric_res

    def _inspect_imaginary_parts(self, grad_res, metric_res):
        if any(np.abs(np.imag(grad_res_item)) > 1e-3 for grad_res_item in grad_res):
            raise Warning("The imaginary part of the gradient are non-negligible.")
        if np.any(
            [
                [np.abs(np.imag(metric_res_item)) > 1e-3 for metric_res_item in metric_res_row]
                for metric_res_row in metric_res
            ]
        ):
            raise Warning("The imaginary part of the metric are non-negligible.")

    def _check_and_fix_metric_psd(self, metric_res):
        while True:
            w, v = np.linalg.eigh(metric_res)

            if not all(ew >= -1e-2 for ew in w):
                raise Warning(
                    "The underlying metric has ein Eigenvalue < ",
                    -1e-2,
                    ". Please use a regularized least-square solver for this problem.",
                )
            if not all(ew >= 0 for ew in w):
                # If not all eigenvalues are non-negative, set them to a small positive
                # value
                w = [max(1e-10, ew) for ew in w]
                # Recompose the adapted eigenvalues with the eigenvectors to get a new metric
                metric_res = np.real(v @ np.diag(w) @ np.linalg.inv(v))
            else:
                # If all eigenvalues are non-negative use the metric
                break
        return metric_res
