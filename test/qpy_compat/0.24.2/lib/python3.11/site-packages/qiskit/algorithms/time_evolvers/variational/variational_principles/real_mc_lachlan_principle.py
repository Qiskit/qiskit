# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
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

import warnings

from collections.abc import Sequence

import numpy as np
from numpy import real

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.primitives import Estimator
from qiskit.quantum_info import SparsePauliOp
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .real_variational_principle import RealVariationalPrinciple

from ....exceptions import AlgorithmError
from ....gradients import (
    BaseEstimatorGradient,
    BaseQGT,
    DerivativeType,
    LinCombQGT,
    LinCombEstimatorGradient,
)


class RealMcLachlanPrinciple(RealVariationalPrinciple):
    """Class for a Real McLachlan's Variational Principle. It aims to minimize the distance
    between both sides of the Schrödinger equation with a quantum state given as a parametrized
    trial state. The principle leads to a system of linear equations handled by a linear solver.
    The real variant means that we consider real time dynamics.
    """

    def __init__(
        self,
        qgt: BaseQGT | None = None,
        gradient: BaseEstimatorGradient | None = None,
    ) -> None:
        """
        Args:
            qgt: Instance of a the GQT class used to compute the QFI.
                If ``None`` provided, ``LinCombQGT`` is used.
            gradient: Instance of a class used to compute the state gradient.
                If ``None`` provided, ``LinCombEstimatorGradient`` is used.

        Raises:
            AlgorithmError: If the gradient instance does not contain an estimator.
        """
        self._validate_grad_settings(gradient)

        if gradient is not None:
            try:
                estimator = gradient._estimator
            except Exception as exc:
                raise AlgorithmError(
                    "The provided gradient instance does not contain an estimator primitive."
                ) from exc
        else:
            estimator = Estimator()
            gradient = LinCombEstimatorGradient(estimator, derivative_type=DerivativeType.IMAG)

        if qgt is None:
            qgt = LinCombQGT(estimator)

        super().__init__(qgt, gradient)

    def evolution_gradient(
        self,
        hamiltonian: BaseOperator,
        ansatz: QuantumCircuit,
        param_values: Sequence[float],
        gradient_params: Sequence[Parameter] | None = None,
    ) -> np.ndarray:
        """
        Calculates an evolution gradient according to the rules of this variational principle.

        Args:
            hamiltonian: Operator used for Variational Quantum Time Evolution.
            ansatz: Quantum state in the form of a parametrized quantum circuit.
            param_values: Values of parameters to be bound.
            gradient_params: List of parameters with respect to which gradients should be computed.
                If ``None`` given, gradients w.r.t. all parameters will be computed.

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

        try:
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
        except Exception as exc:
            raise AlgorithmError("The gradient primitive job failed!") from exc

        # The BaseEstimatorGradient class returns the gradient of the opposite sign than we expect
        # here (i.e. with a minus sign), hence the correction that cancels it to recover the
        # real McLachlan's principle equations that do not have a minus sign.
        evolution_grad = (-1) * evolution_grad
        return evolution_grad

    @staticmethod
    def _construct_modified_hamiltonian(hamiltonian: BaseOperator, energy: float) -> BaseOperator:
        """
        Modifies a Hamiltonian according to the rules of this variational principle.

        Args:
            hamiltonian: Operator used for Variational Quantum Time Evolution.
            energy: The energy correction value.

        Returns:
            A modified Hamiltonian.
        """
        energy_term = SparsePauliOp.from_list(
            hamiltonian.to_list() + [("I" * hamiltonian.num_qubits, -energy)]
        )
        return energy_term

    @staticmethod
    def _validate_grad_settings(gradient):

        if gradient is not None:
            if not hasattr(gradient, "_derivative_type"):
                raise ValueError(
                    "The gradient instance provided does not support calculating imaginary part. "
                    "Please choose a different gradient class."
                )
            if gradient._derivative_type != DerivativeType.IMAG:
                warnings.warn(
                    "A gradient instance with a setting for calculating real part of the"
                    "gradient was provided. This variational principle requires the"
                    "imaginary part. The setting to imaginary was changed automatically."
                )
                gradient._derivative_type = DerivativeType.IMAG
