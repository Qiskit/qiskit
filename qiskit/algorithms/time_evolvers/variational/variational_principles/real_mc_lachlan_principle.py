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
from __future__ import annotations

import numpy as np
from numpy import real

from qiskit import QuantumCircuit
from qiskit.algorithms import AlgorithmError
from qiskit.algorithms.gradients import (
    BaseEstimatorGradient,
    BaseQFI,
    LinCombQFI,
)
from qiskit.algorithms.gradients.lin_comb_estimator_gradient import (
    DerivativeType,
    LinCombEstimatorGradient,
)
from qiskit.circuit import Parameter
from qiskit.opflow import (
    PauliSumOp,
)
from qiskit.primitives import Estimator
from qiskit.quantum_info import Pauli, SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator
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
        qfi: BaseQFI | None = None,
        gradient: BaseEstimatorGradient | None = None,
    ) -> None:
        """
        Args:
            qfi: Instance of a class used to compute the QFI. If ``None`` provided, ``LinCombQFI``
                is used.
            gradient: Instance of a class used to compute the state gradient. If ``None`` provided,
                ``LinCombEstimatorGradient`` is used.
        """
        # TODO validate that gradient passed has IMAG
        if gradient is not None and gradient._estimator is not None and qfi is None:
            estimator = gradient._estimator
            qfi = LinCombQFI(estimator)
        elif qfi is None and gradient is None:
            estimator = Estimator()
            qfi = LinCombQFI(estimator)
            gradient = LinCombEstimatorGradient(estimator, derivative_type=DerivativeType.IMAG)
        self._energy_param = None
        self._energy = None

        super().__init__(qfi, gradient)

    def evolution_grad(
        self,
        hamiltonian: BaseOperator | PauliSumOp,
        ansatz: QuantumCircuit,
        param_dict: dict[Parameter, complex],
        bind_params: list[Parameter],
        gradient_params: list[Parameter],
        param_values: list[complex],
    ) -> np.ndarray:
        """
        Calculates an evolution gradient according to the rules of this variational principle.

        Args:
            hamiltonian: Operator used for Variational Quantum Time Evolution. The operator may be
                given either as a composed op consisting of a Hermitian observable and a
                ``CircuitStateFn`` or a ``ListOp`` of a ``CircuitStateFn`` with a ``ComboFn``. The
                latter case enables the evaluation of a Quantum Natural Gradient.
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            param_dict: Dictionary which relates parameter values to the parameters in the ansatz.
            bind_params: List of parameters that are supposed to be bound.
            gradient_params: List of parameters with respect to which gradients should be computed.
            param_values: Values of parameters to be bound.

        Returns:
            An evolution gradient.

        Raises:
            AlgorithmError: If a gradient job fails.
        """
        try:
            estimator_job = self.gradient._estimator.run([ansatz], [hamiltonian], [param_values])
            energy = estimator_job.result().values[0]
        except Exception as exc:
            raise AlgorithmError("The primitive job failed!") from exc

        modified_hamiltonian = self._construct_modified_hamiltonian(hamiltonian, real(energy))

        evolution_grad = (
            0.5
            * self.gradient.run(
                [ansatz],
                [modified_hamiltonian],
                parameters=[gradient_params],
                parameter_values=[param_values],
            )
            .result()
            .gradients[0]
        )

        # quick fix due to an error on opflow; to be addressed in a separate PR
        evolution_grad = (-1) * evolution_grad
        return evolution_grad

    @staticmethod
    def _construct_modified_hamiltonian(
        hamiltonian: BaseOperator | PauliSumOp, energy: float
    ) -> BaseOperator:
        """
        Modifies a Hamiltonian according to the rules of this variational principle.

        Args:
            hamiltonian: Operator used for Variational Quantum Time Evolution. The operator may be
                given either as a composed op consisting of a Hermitian observable and a
                ``CircuitStateFn`` or a ``ListOp`` of a ``CircuitStateFn`` with a ``ComboFn``. The
                latter case enables the evaluation of a Quantum Natural Gradient.
            energy: The energy correction value.

        Returns:
            A modified Hamiltonian.
        """
        if isinstance(hamiltonian, PauliSumOp):
            energy_term = PauliSumOp(SparsePauliOp(Pauli("I" * hamiltonian.num_qubits)), -energy)
            return hamiltonian + energy_term

        energy_term = SparsePauliOp.from_list(
            hamiltonian.to_list() + [("I" * hamiltonian.num_qubits, -energy)]
        )
        return energy_term
