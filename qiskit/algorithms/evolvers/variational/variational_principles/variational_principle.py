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

from abc import ABC, abstractmethod
from typing import Union, List, Callable

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import (
    CircuitQFI,
    CircuitGradient,
    StateFn,
    OperatorBase,
    QFI,
    Gradient,
)


class VariationalPrinciple(ABC):
    """A Variational Principle class. It determines the time propagation of parameters in a
    quantum state provided as a parametrized quantum circuit (ansatz)."""

    def __init__(
        self,
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
        grad_method: Union[str, CircuitGradient] = "lin_comb",
    ) -> None:
        """
        Args:
            grad_method: The method used to compute the state gradient. Can be either
                ``'param_shift'`` or ``'lin_comb'`` or ``'fin_diff'``.
            qfi_method: The method used to compute the QFI. Can be either
                ``'lin_comb_full'`` or ``'overlap_block_diag'`` or ``'overlap_diag'`` or
                ``CircuitQFI``.
        """
        self._qfi_method = qfi_method
        self._grad_method = grad_method

    @abstractmethod
    def create_qfi(
        self,
    ) -> QFI:
        """
        Created a QFI instance according to the rules of this variational principle. It will be used
        to calculate a metric tensor required in the ODE.

        Returns:
            QFI instance.
        """
        pass

    @abstractmethod
    def calc_evolution_grad(
        self,
    ) -> Gradient:
        """
        Calculates an evolution gradient according to the rules of this variational principle.

        Returns:
            Transformed evolution gradient.
        """
        pass

    @abstractmethod
    def modify_hamiltonian(self, hamiltonian, ansatz, circuit_sampler, param_dict):
        pass
