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

"""Class for a Real McLachlan's Variational Principle."""
from typing import Union, Dict, List, Optional

import numpy as np
from numpy import real

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.opflow import (
    StateFn,
    SummedOp,
    Y,
    I,
    PauliExpectation,
    CircuitQFI,
    CircuitSampler,
    OperatorBase,
    ExpectationBase,
)
from qiskit.opflow.gradients.circuit_gradients import LinComb
from qiskit.utils import QuantumInstance
from .real_variational_principle import (
    RealVariationalPrinciple,
)


class RealMcLachlanPrinciple(RealVariationalPrinciple):
    """Class for a Real McLachlan's Variational Principle. It aims to minimize the distance
    between both sides of the Schrödinger equation with a quantum state given as a parametrized
    trial state. The principle leads to a system of linear equations handled by a linear solver.
    The real variant means that we consider real time dynamics.
    """

    def __init__(
        self,
        qfi_method: Union[str, CircuitQFI] = "lin_comb_full",
    ) -> None:
        """
        Args:
            qfi_method: The method used to compute the QFI. Can be either
                ``'lin_comb_full'`` or ``'overlap_block_diag'`` or ``'overlap_diag'`` or
                ``CircuitQFI``.
        """
        self._grad_method = LinComb(aux_meas_op=-Y)
        self._energy_param = None
        self._energy = None

        super().__init__(qfi_method)

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
            self._energy_param = Parameter("alpha")
            modified_hamiltonian = self._construct_expectation(
                hamiltonian, ansatz, self._energy_param
            )

            self._evolution_gradient_callable = self._evolution_gradient.gradient_wrapper(
                modified_hamiltonian,
                bind_params + [self._energy_param],
                gradient_params,
                quantum_instance,
                expectation,
            )

            energy = StateFn(hamiltonian, is_measurement=True) @ StateFn(ansatz)
            if expectation is None:
                expectation = PauliExpectation()
            self._energy = expectation.convert(energy)

        if circuit_sampler is not None:
            energy = circuit_sampler.convert(self._energy, param_dict).eval()
        else:
            energy = self._energy.assign_parameters(param_dict).eval()

        param_values.append(real(energy))
        evolution_grad = 0.5 * self._evolution_gradient_callable(param_values)

        # quick fix due to an error on opflow; to be addressed in a separate PR
        evolution_grad = (-1) * evolution_grad
        return evolution_grad

    @staticmethod
    def _construct_expectation(
        hamiltonian: OperatorBase, ansatz: QuantumCircuit, energy_param: Parameter
    ) -> OperatorBase:
        """
        Modifies a Hamiltonian according to the rules of this variational principle.

        Args:
            hamiltonian: Operator used for Variational Quantum Time Evolution. The operator may be
                given either as a composed op consisting of a Hermitian observable and a
                ``CircuitStateFn`` or a ``ListOp`` of a ``CircuitStateFn`` with a ``ComboFn``. The
                latter case enables the evaluation of a Quantum Natural Gradient.
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            energy_param: Parameter for energy correction.

        Returns:
            An modified Hamiltonian composed with an ansatz.
        """
        energy_term = I ^ hamiltonian.num_qubits
        energy_term *= -1
        energy_term *= energy_param
        modified_hamiltonian = SummedOp([hamiltonian, energy_term]).reduce()
        return StateFn(modified_hamiltonian, is_measurement=True) @ StateFn(ansatz)
