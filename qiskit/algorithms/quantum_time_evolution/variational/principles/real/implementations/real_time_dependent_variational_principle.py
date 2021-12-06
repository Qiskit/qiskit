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
from typing import Union, List

from qiskit import QuantumCircuit
from qiskit.algorithms.quantum_time_evolution.variational.calculators import (
    metric_tensor_calculator,
    evolution_grad_calculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.real.real_variational_principle import (
    RealVariationalPrinciple,
)
from qiskit.circuit import Parameter
from qiskit.opflow import CircuitQFI, Y, OperatorBase, ListOp, StateFn

"""Class for a Real Time Dependent Variational Principle."""


class RealTimeDependentVariationalPrinciple(RealVariationalPrinciple):
    def __init__(
        self,
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
    ):
        """
        Args:
            qfi_method: The method used to compute the QFI. Can be either
                ``'lin_comb_full'`` or ``'overlap_block_diag'`` or ``'overlap_diag'``.
        """
        super().__init__(
            qfi_method,
        )

    def _get_metric_tensor(
        self,
        ansatz: QuantumCircuit,
        parameters: List[Parameter],
    ) -> ListOp:
        """
        Calculates a metric tensor according to the rules of this variational principle.
        Args:
            ansatz: Quantum state to be used for calculating a metric tensor.
            parameters: Parameters with respect to which gradients should be computed.
        Returns:
            Transformed metric tensor.
        """
        raw_metric_tensor_imag = metric_tensor_calculator.calculate(
            ansatz, parameters, self._qfi_method, basis=-1j * Y
        )

        return raw_metric_tensor_imag * 0.25

    def _get_evolution_grad(
        self,
        hamiltonian: OperatorBase,
        ansatz: Union[StateFn, QuantumCircuit],
        parameters: List[Parameter],
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
        raw_evolution_grad_real = evolution_grad_calculator.calculate(
            hamiltonian, ansatz, parameters, self._grad_method
        )

        return raw_evolution_grad_real * 0.5
