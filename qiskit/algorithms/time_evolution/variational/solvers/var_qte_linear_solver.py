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

"""Class for solving linear equations for Quantum Time Evolution."""

from typing import Union, List, Dict, Optional

import numpy as np

from qiskit.algorithms.time_evolution.variational.calculators.metric_tensor_calculator import (
    eval_metric_result,
)
from qiskit.algorithms.time_evolution.variational.calculators.evolution_grad_calculator import (
    eval_grad_result,
)
from qiskit.algorithms.time_evolution.variational.variational_principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.opflow.gradients.natural_gradient import NaturalGradient
from qiskit.circuit import Parameter
from qiskit.opflow import CircuitSampler
from qiskit.providers import BaseBackend
from qiskit.utils import QuantumInstance


class VarQteLinearSolver:
    """Class for solving linear equations for Quantum Time Evolution."""

    def __init__(
        self,
        grad_circ_sampler: CircuitSampler,
        metric_circ_sampler: CircuitSampler,
        energy_sampler: CircuitSampler,
        regularization: Optional[str] = None,
        backend: Optional[Union[BaseBackend, QuantumInstance]] = None,
    ):
        """
        Args:
            grad_circ_sampler: CircuitSampler for evolution gradients.
            metric_circ_sampler: CircuitSampler for metric tensors.
            energy_sampler: CircuitSampler for energy.
            regularization: Use the following regularization with a least square method to solve the
                            underlying system of linear equations.
                            Can be either None or ``'ridge'`` or ``'lasso'`` or ``'perturb_diag'``
                            ``'ridge'`` and ``'lasso'`` use an automatic optimal parameter search,
                            or a penalty term given as Callable.
                            If regularization is None but the metric is ill-conditioned or singular
                            then a least square solver is used without regularization.
            backend: Optional backend tht enables the use of circuit samplers.
        """

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
            param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
            t_param: Time parameter in case of a time-dependent Hamiltonian.
            time_value: Time value that will be bound to t_param. It is required if t_param is
            not None.
            regularization: Use the following regularization with a least square method to solve the
                            underlying system of linear equations.
                            Can be either None or ``'ridge'`` or ``'lasso'`` or ``'perturb_diag'``
                            ``'ridge'`` and ``'lasso'`` use an automatic optimal parameter search,
                            or a penalty term given as Callable.
                            If regularization is None but the metric is ill-conditioned or singular
                            then a least square solver is used without regularization.
        Returns: dω/dt, 2Re⟨dψ(ω)/dω|H|ψ(ω) for VarQITE/ 2Im⟨dψ(ω)/dω|H|ψ(ω) for VarQRTE,
                Fubini-Study Metric.
        """
        grad = var_principle._raw_evolution_grad
        metric = var_principle._raw_metric_tensor

        if t_param is not None:
            time_dict = {t_param: time_value}
            # TODO
            # Use var_principle to understand type and get gradient
            # bind parameters to H(t)
            # grad
            # natural gradient with nat_grad_combo_fn(x, regularization=None)
            # TODO raw_evolution_grad might be passed as a callback
            grad = var_principle._raw_evolution_grad.bind_parameters(time_dict)
            metric = var_principle._raw_metric_tensor.bind_parameters(time_dict)

        grad_result = eval_grad_result(
            grad, param_dict, self._grad_circ_sampler, self._energy_sampler
        )

        metric_result = eval_metric_result(
            metric,
            param_dict,
            self._metric_circ_sampler,
        )

        nat_grad_result = NaturalGradient().nat_grad_combo_fn(
            [grad_result, metric_result], regularization=regularization
        )

        # error-based needs all three, non-error-based needs only nat_grad_result
        return np.real(nat_grad_result), metric_result, grad_result
