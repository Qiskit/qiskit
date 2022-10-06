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
from qiskit.algorithms import AlgorithmError
from qiskit.algorithms.gradients import (
    BaseEstimatorGradient,
    BaseQFI,
)
from qiskit.circuit import Parameter
from qiskit.opflow import PauliSumOp
from qiskit.quantum_info.operators.base_operator import BaseOperator


class VariationalPrinciple(ABC):
    """A Variational Principle class. It determines the time propagation of parameters in a
    quantum state provided as a parametrized quantum circuit (ansatz)."""

    def __init__(
        self,
        qfi: BaseQFI | None = None,
        gradient: BaseEstimatorGradient | None = None,
    ) -> None:
        """
        Args:
            qfi: Instance of a class used to compute the QFI. If ``None`` provided, ``LinCombQFI``
                is used.
            gradient: Instance of a class used to compute the state gradient. If ``None`` provided,
                ``LinCombEstimatorGradient`` is used.
        """
        self.qfi = qfi
        self.gradient = gradient

    def metric_tensor(self, ansatz: QuantumCircuit, param_values: list[complex]) -> np.ndarray:
        """
        Calculates a metric tensor according to the rules of this variational principle.

        Args:
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            param_values: Values of parameters to be bound.

        Returns:
            Metric tensor.

        Raises:
            AlgorithmError: If a QFI job fails.
        """

        try:
            metric_tensor = 0.25 * self.qfi.run([ansatz], [param_values], [None]).result().qfis[0]
        except Exception as exc:

            raise AlgorithmError("The primitive job failed!") from exc
        return metric_tensor

    @abstractmethod
    def evolution_grad(
        self,
        hamiltonian: BaseOperator | PauliSumOp,
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
