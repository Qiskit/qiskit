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
from abc import ABC, abstractmethod
from typing import Union

from qiskit.algorithms.quantum_time_evolution.variational.calculators import (
    evolution_grad_calculator,
    metric_tensor_calculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.gradient_errors\
    .error_calculator import (
    ErrorCalculator,
)
from qiskit.opflow import CircuitQFI, CircuitGradient, OperatorBase, StateFn


class VariationalPrinciple(ABC):
    def __init__(
            self,
            qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
            grad_method: Union[str, CircuitGradient] = "lin_comb",
            is_error_supported: bool = False,
    ):
        self._qfi_method = qfi_method
        self._grad_method = grad_method
        self._is_error_supported = is_error_supported

    def _lazy_init(self, hamiltonian, ansatz, parameters):
        self._hamiltonian = hamiltonian
        self._ansatz = ansatz
        self._parameters = parameters
        self._operator = ~StateFn(hamiltonian) @ StateFn(ansatz)
        raw_metric_tensor = metric_tensor_calculator.calculate(ansatz, parameters, self._qfi_method)
        raw_evolution_grad = evolution_grad_calculator.calculate(
            hamiltonian, ansatz, parameters, self._grad_method
        )
        self._metric_tensor = self._calc_metric_tensor(raw_metric_tensor)
        self._evolution_grad = self._calc_evolution_grad(raw_evolution_grad)

    @staticmethod
    @abstractmethod
    def _calc_metric_tensor(raw_metric_tensor):
        pass

    @staticmethod
    @abstractmethod
    def _calc_evolution_grad(raw_evolution_grad):
        pass

    @abstractmethod
    def _calc_error_bound(
            self, error, et, h_squared, h_trip, trained_energy, variational_principle
    ):
        pass

    @property
    def metric_tensor(self) -> OperatorBase:
        return self._metric_tensor

    @property
    def evolution_grad(self) -> OperatorBase:
        return self._evolution_grad

    @staticmethod
    def op_real_part(operator: OperatorBase) -> OperatorBase:
        return (operator + operator.adjoint()) / 2.0

    @staticmethod
    def op_imag_part(operator: OperatorBase) -> OperatorBase:
        return (operator - operator.adjoint()) / 2.0
