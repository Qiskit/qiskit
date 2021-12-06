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
"""Abstract class for an Imaginary Variational Principle."""
from abc import abstractmethod
from typing import Union, Dict

from qiskit import QuantumCircuit
from qiskit.algorithms.quantum_time_evolution.variational.principles.variational_principle import (
    VariationalPrinciple,
)
from qiskit.circuit import Parameter
from qiskit.opflow import (
    CircuitQFI,
    CircuitGradient,
    StateFn,
    OperatorBase,
    ListOp,
)


class ImaginaryVariationalPrinciple(VariationalPrinciple):
    """Abstract class for an Imaginary Variational Principle."""

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
        super().__init__(
            qfi_method,
            grad_method,
        )

    @abstractmethod
    def _get_metric_tensor(
        self,
        ansatz: QuantumCircuit,
        parameters: Dict[Parameter, Union[float, complex]],
    ) -> ListOp:
        """
        Calculates a metric tensor according to the rules of this variational principle.
        Args:
            ansatz: Quantum state to be used for calculating a metric tensor.
            parameters: Parameters with respect to which gradients should be computed.
        Returns:
            Transformed metric tensor.
        """
        pass

    @abstractmethod
    def _get_evolution_grad(
        self,
        hamiltonian: OperatorBase,
        ansatz: Union[StateFn, QuantumCircuit],
        parameters: Dict[Parameter, Union[float, complex]],
    ) -> OperatorBase:
        """
        Calculates an evolution gradient according to the rules of this variational principle.
        Args:
            hamiltonian: Observable for which an evolution gradient should be calculated,
                        e.g., a Hamiltonian of a system.
            ansatz: Quantum state to be used for calculating an evolution gradient.
            parameters: Parameters with respect to which gradients should be computed.
        Returns:
            Transformed evolution gradient.
        """
        pass
