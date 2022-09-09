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

"""Class for a Variational Principle."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import (
    CircuitQFI,
    CircuitGradient,
    QFI,
    Gradient,
    CircuitStateFn,
    OperatorBase,
)
from qiskit.primitives import BaseEstimator


class VariationalPrinciple(ABC):
    """A Variational Principle class. It determines the time propagation of parameters in a
    quantum state provided as a parametrized quantum circuit (ansatz)."""

    def __init__(
        self,
        qfi_method: str | CircuitQFI = "lin_comb_full",
        grad_method: str | CircuitGradient = "lin_comb",
    ) -> None:
        """
        Args:
            grad_method: The method used to compute the state gradient. Can be either
                ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'`` or ``CircuitGradient``.
            qfi_method: The method used to compute the QFI. Can be either
                ``'lin_comb_full'`` or ``'overlap_block_diag'`` or ``'overlap_diag'`` or
                ``CircuitQFI``.
        """
        self._qfi_method = qfi_method
        self.qfi = QFI(qfi_method)
        self._grad_method = grad_method
        self._evolution_gradient = Gradient(self._grad_method)
        self._qfi_gradient_callable = None
        self._evolution_gradient_callable = None

    def metric_tensor(
        self,
        estimator: BaseEstimator,
        ansatz: QuantumCircuit,
        bind_params: list[Parameter],
        gradient_params: list[Parameter],
        param_values: list[complex],
    ) -> np.ndarray:
        """
        Calculates a metric tensor according to the rules of this variational principle.

        Args:
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            bind_params: List of parameters that are supposed to be bound.
            gradient_params: List of parameters with respect to which gradients should be computed.
            param_values: Values of parameters to be bound.

        Returns:
            Metric tensor.
        """
        if self._qfi_gradient_callable is None:
            self._qfi_gradient_callable = self.qfi.gradient_wrapper(
                CircuitStateFn(ansatz), bind_params, gradient_params
            )
        metric_tensor = 0.25 * self._qfi_gradient_callable(param_values)

        return metric_tensor

    @abstractmethod
    def evolution_grad(
        self,
        estimator: BaseEstimator,
        hamiltonian: OperatorBase,
        ansatz: QuantumCircuit,
        param_dict: dict[Parameter, complex],
        bind_params: list[Parameter],
        gradient_params: list[Parameter],
        param_values: list[complex],
    ) -> np.ndarray:
        """
        Calculates an evolution gradient according to the rules of this variational principle.

        Args:
            hamiltonian: Operator used for Variational Quantum Time Evolution. The operator may be
                given either as a composed op consisting of a Hermitian observable and a
                ``CircuitStateFn`` or a ``ListOp`` of a ``CircuitStateFn`` with a ``ComboFn``. The
                latter case enables the evaluation of a Quantum Natural Gradient.
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
            bind_params: List of parameters that are supposed to be bound.
            gradient_params: List of parameters with respect to which gradients should be computed.
            param_values: Values of parameters to be bound.

        Returns:
            An evolution gradient.
        """
        pass
