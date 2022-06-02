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

"""Class for an Imaginary McLachlan's Variational Principle."""

from typing import Union, List

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import StateFn, OperatorBase, QFI
from ..calculators import (
    evolution_grad_calculator,
)
from .imaginary_variational_principle import (
    ImaginaryVariationalPrinciple,
)


class ImaginaryMcLachlanPrinciple(ImaginaryVariationalPrinciple):
    """Class for an Imaginary McLachlan's Variational Principle. It aims to minimize the distance
    between both sides of the Wick-rotated Schrödinger equation with a quantum state given as a
    parametrized trial state. The principle leads to a system of linear equations handled by the
    `~qiskit.algorithms.evolvers.variational.solvers.VarQTELinearSolver` class. The imaginary
    variant means that we consider imaginary time dynamics.
    """

    def create_qfi(
        self,
    ) -> QFI:
        """
        Creates a QFI instance according to the rules of this variational principle. It is used
        to calculate a metric tensor required in the ODE.

        Returns:
            QFI instance.
        """

        return QFI(self._qfi_method)

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
        evolution_grad_real = evolution_grad_calculator.calculate(
            hamiltonian, ansatz, parameters, self._grad_method
        )

        return (-1) * evolution_grad_real * 0.5
