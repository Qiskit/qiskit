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
from typing import Union, Dict, Optional, List

from qiskit.circuit import Parameter
from qiskit.opflow import (
    CircuitQFI,
    CircuitGradient,
    OperatorBase,
    StateFn,
)


class VariationalPrinciple(ABC):
    def __init__(
        self,
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
        grad_method: Union[str, CircuitGradient] = "lin_comb",
        is_error_supported: Optional[bool] = False,
    ):
        """
        Args:
            grad_method: The method used to compute the state gradient. Can be either
                        ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``.
            qfi_method: The method used to compute the QFI. Can be either
                ``'lin_comb_full'`` or ``'overlap_block_diag'`` or ``'overlap_diag'``.
        """
        self._qfi_method = qfi_method
        self._grad_method = grad_method
        self._is_error_supported = is_error_supported

    def _lazy_init(
        self,
        hamiltonian,
        ansatz,
        parameters: List[Parameter]
    ):

        self._hamiltonian = hamiltonian
        self._ansatz = ansatz
        self._operator = ~StateFn(hamiltonian) @ StateFn(ansatz)
        self._params = parameters



        self._raw_evolution_grad = self._get_raw_evolution_grad(hamiltonian, ansatz, parameters)
        self._raw_metric_tensor = self._get_raw_metric_tensor(ansatz, parameters)

        # self._metric_tensor = self._calc_metric_tensor(raw_metric_tensor, param_dict)
        # self._evolution_grad = self._calc_evolution_grad(raw_evolution_grad, param_dict)

        # self._nat_grad = self._calc_nat_grad(self._operator, param_dict, regularization)

    @abstractmethod
    def _get_raw_metric_tensor(
        self,
        ansatz,
        parameters: List[Parameter],
    ):
        pass

    @abstractmethod
    def _get_raw_evolution_grad(
        self,
        hamiltonian,
        ansatz,
        parameters: List[Parameter],
    ):
        pass

    @staticmethod
    @abstractmethod
    def _calc_metric_tensor(raw_metric_tensor, param_dict: Dict[Parameter, Union[float, complex]]):
        pass

    @staticmethod
    @abstractmethod
    def _calc_evolution_grad(
        raw_evolution_grad, param_dict: Dict[Parameter, Union[float, complex]]
    ):
        pass

    @abstractmethod
    def _calc_nat_grad(
        self,
        raw_operator: OperatorBase,
        param_dict: Dict[Parameter, Union[float, complex]],
        regularization: Optional[str] = None,
    ) -> OperatorBase:
        raise NotImplementedError()

    @abstractmethod
    def _calc_error_bound(
        self, error: float,
            et: float,
            h_squared_expectation: float,
            h_trip: float,
            trained_energy: float
    ) -> float:
        pass

    @property
    def metric_tensor(self) -> OperatorBase:
        return self._metric_tensor

    @property
    def evolution_grad(self) -> OperatorBase:
        return self._evolution_grad
