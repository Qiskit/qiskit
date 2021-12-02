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
from typing import Union, Optional, List, Callable

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import (
    CircuitQFI,
    CircuitGradient,
    StateFn,
    ListOp,
    OperatorBase,
)


class VariationalPrinciple(ABC):
    def __init__(
        self,
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
        grad_method: Union[str, CircuitGradient] = "lin_comb",
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

    def _lazy_init(
        self,
        hamiltonian: OperatorBase,
        ansatz: Union[StateFn, QuantumCircuit],
        parameters: List[Parameter],
    ) -> None:

        self._hamiltonian = hamiltonian
        self._ansatz = ansatz
        self._operator = ~StateFn(hamiltonian) @ StateFn(ansatz)
        self._params = parameters

        self._raw_evolution_grad = self._get_evolution_grad(hamiltonian, ansatz, parameters)
        self._raw_metric_tensor = self._get_metric_tensor(ansatz, parameters)

    @abstractmethod
    def _get_metric_tensor(
        self,
        ansatz: QuantumCircuit,
        parameters: List[Parameter],
    ) -> ListOp:
        pass

    @abstractmethod
    def _get_evolution_grad(
        self,
        hamiltonian: OperatorBase,
        ansatz: Union[StateFn, QuantumCircuit],
        parameters: List[Parameter],
    ) -> Union[OperatorBase, Callable]:
        pass
