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
)
from qiskit.opflow.gradients.circuit_gradients import LinComb
from qiskit.utils import QuantumInstance
from .real_variational_principle import (
    RealVariationalPrinciple,
)


class RealMcLachlanPrinciple(RealVariationalPrinciple):
    """Class for an Imaginary McLachlan's Variational Principle. It aims to minimize the distance
    between both sides of the Schrödinger equation with a quantum state given as a parametrized
    trial state. The principle leads to a system of linear equations handled by the
    `~qiskit.algorithms.evolvers.variational.solvers.VarQTELinearSolver` class. The real variant
    means that we consider real time dynamics.
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
        self.energy_param = None
        self._energy = None

        super().__init__(qfi_method)

    def calc_evolution_grad(
        self,
        hamiltonian: OperatorBase,
        ansatz: Union[StateFn, QuantumCircuit],
        circuit_sampler: CircuitSampler,
        param_dict: Dict[Parameter, complex],
        bind_params: List[Parameter],
        gradient_params: List[Parameter],
        param_values: List[complex],
        quantum_instance: Optional[QuantumInstance] = None,
    ) -> np.ndarray:
        """
        Calculates an evolution gradient according to the rules of this variational principle.
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
            bind_params: List of parameters that are supposed to be bound.
            gradient_params: List of parameters with respect to which gradients should be computed.
            param_values: Values of parameters to be bound.
            quantum_instance: Backend used to evaluate the quantum circuit outputs. If ``None``
                provided, everything will be evaluated based on matrix multiplication (which is
                slow).
        Returns:
            An evolution gradient.
        """
        if self._evolution_gradient_callable is None:
            modified_hamiltonian, self.energy_param = self.modify_hamiltonian(hamiltonian, ansatz)

            self._evolution_gradient_callable = self._evolution_gradient.gradient_wrapper(
                modified_hamiltonian,
                bind_params + [self.energy_param],
                gradient_params,
                quantum_instance,
            )

            energy = StateFn(hamiltonian, is_measurement=True) @ StateFn(ansatz)
            self._energy = PauliExpectation().convert(energy)

        # TODO param_dict does not contain time
        if circuit_sampler is not None:
            energy = circuit_sampler.convert(self._energy, param_dict).eval()
        else:
            energy = self._energy.assign_parameters(param_dict).eval()
        # print(f"Energy:{energy}")
        param_values.append(real(energy))
        evolution_grad_lse_rhs = 0.5 * self._evolution_gradient_callable(param_values)
        # evolution_grad_lse_rhs = self._post_bind_alpha_param(evolution_grad_lse_rhs, self.energy_param,
        # real(energy))

        # quick fix due to an error on opflow; to be addressed in a separate PR
        evolution_grad_lse_rhs = (-1) * evolution_grad_lse_rhs
        return evolution_grad_lse_rhs

    def _post_bind_alpha_param(
        self, evolution_grad_lse_rhs: np.ndarray, energy_param, energy_value: float
    ) -> np.ndarray:
        """
        In case of a time-dependent Hamiltonian, binds a time parameter in an evolution gradient
        (if they contain a time parameter) with a current time value.
        """
        bound_evolution_grad_lse_rhs = np.zeros(len(evolution_grad_lse_rhs), dtype=complex)
        for i, param_expr in enumerate(evolution_grad_lse_rhs):
            bound_evolution_grad_lse_rhs[i] = param_expr.assign(
                energy_param, energy_value
            ).__complex__()
        evolution_grad_lse_rhs = bound_evolution_grad_lse_rhs
        return evolution_grad_lse_rhs

    def modify_hamiltonian(
        self,
        hamiltonian: OperatorBase,
        ansatz: Union[StateFn, QuantumCircuit],
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
        Returns:
            An modified Hamiltonian composed with an ansatz.
        """
        energy_param = Parameter("alpha")
        energy_term = I ^ hamiltonian.num_qubits
        energy_term *= -1
        energy_term *= energy_param
        hamiltonian_ = SummedOp([hamiltonian, energy_term]).reduce()
        operator = StateFn(hamiltonian_, is_measurement=True) @ StateFn(ansatz)
        return operator, energy_param
