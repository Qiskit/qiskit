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

"""Class for a Real Time Dependent Variational Principle."""

from typing import Union, List

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import Y, OperatorBase, StateFn, CircuitQFI
from qiskit.opflow.gradients.circuit_qfis import LinCombFull
from ..calculators import (
    evolution_grad_calculator,
)
from .real_variational_principle import (
    RealVariationalPrinciple,
)


class RealTimeDependentPrinciple(RealVariationalPrinciple):
    """Class for a Real Time Dependent Variational Principle. It works by evaluating the Lagrangian
    corresponding the given system at a parametrized trial state and applying the Euler-Lagrange
    equation. The principle leads to a system of linear equations handled by the
    `~qiskit.algorithms.evolvers.variational.solvers.VarQTELinearSolver` class. The real variant
    means that we consider real time dynamics.
    """

    def __init__(self, qfi_method: Union[str, CircuitQFI] = "lin_comb_full") -> None:
        """
        Args:
            qfi_method: The method used to compute the QFI. Can be either
                ``'lin_comb_full'`` or ``'overlap_block_diag'`` or ``'overlap_diag'`` or
                ``CircuitQFI``.
        """
        if qfi_method == "lin_comb_full" or isinstance(qfi_method, LinCombFull):
            qfi_method = LinCombFull(aux_meas_op=-Y)

        super().__init__(qfi_method)

    def calc_evolution_grad(
        self,
        hamiltonian: OperatorBase,
        ansatz: Union[StateFn, QuantumCircuit],
        parameters: List[Parameter],
    ) -> OperatorBase:
        """
        Calculates an evolution gradient according to the rules of this variational principle.

        Args:
            hamiltonian: Hamiltonian for which an evolution gradient should be calculated.
            ansatz: Quantum state in the form of a parametrized quantum circuit to be used for
                calculating an evolution gradient.
            parameters: Parameters with respect to which gradients should be computed.

        Returns:
            Transformed evolution gradient.
        """
        raw_evolution_grad_real = evolution_grad_calculator.calculate(
            hamiltonian, ansatz, parameters, self._grad_method
        )

        return raw_evolution_grad_real * 0.5
