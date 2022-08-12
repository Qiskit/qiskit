# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Abstract Base class of Gradient for Estimator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .estimator_gradient_job import EstimatorGradientJob


class BaseEstimatorGradient(ABC):
    """Base class for an EstimatorGradient to compute the gradients of the expectation value."""

    def __init__(
        self,
        estimator: BaseEstimator,
    ):
        """
        Args:
            estimator: The estimator used to compute the gradients.
        """
        self._estimator: BaseEstimator = estimator
        self._circuits: Sequence[QuantumCircuit] = []
        self._circuit_ids: dict[int, int] = {}

    def run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        partial: Sequence[Sequence[Parameter]] | None = None,
        **run_options,
    ) -> EstimatorGradientJob:
        """Run the job of the gradients of expectation values.

        Args:
            circuits: The list of quantum circuits to compute the gradients.
            observables: The list of observables.
            parameter_values: The list of parameter values to be bound to the circuit.
            partial: The list of Parameters to calculate only the gradients of the specified parameters.
                Defaults to None, which means that the gradients of all parameters will be calculated.
            run_options: Backend runtime options used for circuit execution.

        Returns:
            The job object of the gradients of the expectation values. The i-th result corresponds to
            ``circuits[i]`` evaluated with parameters bound as ``parameter_values[i]``.
        """
        return self._run(circuits, observables, parameter_values, partial, **run_options)

    @abstractmethod
    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        partial: Sequence[Sequence[Parameter]] | None = None,
        **run_options,
    ) -> EstimatorGradientJob:
        raise NotImplementedError()
