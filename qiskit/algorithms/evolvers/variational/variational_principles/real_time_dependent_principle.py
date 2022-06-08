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
from typing import Union, Dict

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import Y, StateFn, CircuitQFI, OperatorBase, CircuitSampler
from qiskit.opflow.gradients.circuit_gradients import LinComb
from qiskit.opflow.gradients.circuit_qfis import LinCombFull
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
        self._grad_method = LinComb()

        super().__init__(qfi_method)

    def modify_hamiltonian(
        self,
        hamiltonian: OperatorBase,
        ansatz: Union[StateFn, QuantumCircuit],
        circuit_sampler: CircuitSampler,
        param_dict: Dict[Parameter, complex],
    ) -> OperatorBase:
        """
        Modifies a Hamiltonian according to the rules of this variational principle.
        Args:
            hamiltonian:
                Operator used for Variational Quantum Time Evolution.
                The operator may be given either as a composed op consisting of a Hermitian
                observable and a ``CircuitStateFn`` or a ``ListOp`` of a ``CircuitStateFn`` with a
                ``ComboFn``.
                The latter case enables the evaluation of a Quantum Natural Gradient.
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            circuit_sampler: A circuit sampler.
            param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
        Returns:
            An modified Hamiltonian composed with an ansatz.
        """
        return StateFn(hamiltonian, is_measurement=True) @ StateFn(ansatz)
