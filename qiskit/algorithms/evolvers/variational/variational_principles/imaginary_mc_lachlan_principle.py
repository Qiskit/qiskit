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
from typing import Dict, List, Optional

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import StateFn, OperatorBase, CircuitSampler, ExpectationBase
from qiskit.utils import QuantumInstance
from .imaginary_variational_principle import (
    ImaginaryVariationalPrinciple,
)


class ImaginaryMcLachlanPrinciple(ImaginaryVariationalPrinciple):
    """Class for an Imaginary McLachlan's Variational Principle. It aims to minimize the distance
    between both sides of the Wick-rotated Schrödinger equation with a quantum state given as a
    parametrized trial state. The principle leads to a system of linear equations handled by a
    linear solver. The imaginary variant means that we consider imaginary time dynamics.
    """

    def evolution_grad(
        self,
        hamiltonian: OperatorBase,
        ansatz: QuantumCircuit,
        circuit_sampler: CircuitSampler,
        param_dict: Dict[Parameter, complex],
        bind_params: List[Parameter],
        gradient_params: List[Parameter],
        param_values: List[complex],
        expectation: Optional[ExpectationBase] = None,
        quantum_instance: Optional[QuantumInstance] = None,
    ) -> np.ndarray:
        """
        Calculates an evolution gradient according to the rules of this variational principle.

        Args:
            hamiltonian: Operator used for Variational Quantum Time Evolution. The operator may be
                given either as a composed op consisting of a Hermitian observable and a
                ``CircuitStateFn`` or a ``ListOp`` of a ``CircuitStateFn`` with a ``ComboFn``. The
                latter case enables the evaluation of a Quantum Natural Gradient.
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            circuit_sampler: A circuit sampler.
            param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
            bind_params: List of parameters that are supposed to be bound.
            gradient_params: List of parameters with respect to which gradients should be computed.
            param_values: Values of parameters to be bound.
            expectation: An instance of ``ExpectationBase`` used for calculating an evolution
                gradient. If ``None`` provided, a ``PauliExpectation`` is used.
            quantum_instance: Backend used to evaluate the quantum circuit outputs. If ``None``
                provided, everything will be evaluated based on matrix multiplication (which is
                slow).

        Returns:
            An evolution gradient.
        """
        if self._evolution_gradient_callable is None:
            operator = StateFn(hamiltonian, is_measurement=True) @ StateFn(ansatz)
            self._evolution_gradient_callable = self._evolution_gradient.gradient_wrapper(
                operator, bind_params, gradient_params, quantum_instance, expectation
            )
        evolution_grad_lse_rhs = -0.5 * self._evolution_gradient_callable(param_values)

        return evolution_grad_lse_rhs
