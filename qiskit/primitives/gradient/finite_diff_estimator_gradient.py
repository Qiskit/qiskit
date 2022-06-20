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

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..base_estimator import BaseEstimator
from ..estimator_result import EstimatorResult
from ..factories import EstimatorFromCircuitsAndObservables
from .base_estimator_gradient import BaseEstimatorGradient
from .estimator_gradient_result import EstimatorGradientResult


class FiniteDiffEstimatorGradient(BaseEstimatorGradient):
    def __init__(
        self,
        estimator_factory: EstimatorFromCircuitsAndObservables,
        circuits: QuantumCircuit | Sequence[QuantumCircuit],
        observables: BaseOperator | PauliSumOp | Sequence[BaseOperator | PauliSumOp],
        epsilon: float = 1e-6,
    ):
        self._epsilon = epsilon
        estimator = estimator_factory(circuits, observables)
        self._num_circuits = len(circuits)
        self._num_observables = len(observables)
        super().__init__(estimator)

    def gradient(
        self,
        circuits: Sequence[int | QuantumCircuit],
        observables: Sequence[int | SparsePauliOp],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        # TODO: support QC and SPO
        gradients = []
        for circuit_index, observable_index, parameter_value in zip(
            circuits, observables, parameter_values
        ):
            dim = len(parameter_value)
            ret = [parameter_value]
            for i in range(dim):
                ei = parameter_value.copy()
                ei[i] += self._epsilon
                ret.append(ei)
            param_array = np.array(ret).tolist()
            circuit_indices = [circuit_index] * (dim + 1)
            observable_indices = [observable_index] * (dim + 1)
            # TODO: batch
            results = self._estimator.__call__(
                circuit_indices, observable_indices, param_array, **run_options
            )

            values = results.values
            gradient = np.zeros(dim)
            f_ref = values[0]
            for i, f_i in enumerate(values[1:]):
                gradient[i] = (f_i - f_ref) / self._epsilon
            gradients.append(gradient)
        return EstimatorGradientResult(values=gradients, metadata=[{}] * len(gradients))

    def estimate(
        self,
        circuits: Sequence[int | QuantumCircuit],
        observables: Sequence[int | SparsePauliOp],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> EstimatorResult:
        # TODO: support QC and SPO
        for circuit in circuits:
            if circuit >= self._num_circuits:
                raise IndexError()
        for observable in observables:
            if observable >= self._num_observables:
                raise IndexError()

        return self._estimator(circuits, observables, parameter_values, **run_options)
