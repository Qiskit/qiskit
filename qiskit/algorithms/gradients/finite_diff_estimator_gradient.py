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

"""Gradient of Sampler with Finite difference method."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from qiskit.algorithms import AlgorithmError
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.providers import Options
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator_gradient import BaseEstimatorGradient
from .estimator_gradient_result import EstimatorGradientResult


class FiniteDiffEstimatorGradient(BaseEstimatorGradient):
    """
    Compute the gradients of the expectation values by finite difference method.
    """

    def __init__(self, estimator: BaseEstimator, epsilon: float, options: Options | None = None):
        """
        Args:
            estimator: The estimator used to compute the gradients.
            epsilon: The offset size for the finite difference gradients.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting

        Raises:
            ValueError: If ``epsilon`` is not positive.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon ({epsilon}) should be positive.")
        self._epsilon = epsilon
        self._base_parameter_values_dict = {}
        super().__init__(estimator, options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        **options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        jobs, metadata_ = [], []
        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):
            # indices of parameters to be differentiated
            if parameters_ is None:
                indices = list(range(circuit.num_parameters))
            else:
                indices = [circuit.parameters.data.index(p) for p in parameters_]
            metadata_.append({"parameters": [circuit.parameters[idx] for idx in indices]})

            offset = np.identity(circuit.num_parameters)[indices, :]
            plus = parameter_values_ + self._epsilon * offset
            minus = parameter_values_ - self._epsilon * offset
            n = 2 * len(indices)
            job = self._estimator.run(
                [circuit] * n, [observable] * n, plus.tolist() + minus.tolist(), **options
            )
            jobs.append(job)

        # combine the results
        try:
            results = [job.result() for job in jobs]
        except Exception as exc:
            raise AlgorithmError("Estimator job failed.") from exc

        gradients = []
        for result in results:
            n = len(result.values) // 2  # is always a multiple of 2
            gradient_ = (result.values[:n] - result.values[n:]) / (2 * self._epsilon)
            gradients.append(gradient_)
        opt = self._get_local_options(options)
        return EstimatorGradientResult(gradients=gradients, metadata=metadata_, options=opt)
