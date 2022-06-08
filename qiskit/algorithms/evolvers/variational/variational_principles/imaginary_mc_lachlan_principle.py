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
from typing import Dict, Union

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import StateFn, OperatorBase, CircuitSampler
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
        return (-1) * (StateFn(hamiltonian, is_measurement=True) @ StateFn(ansatz))
