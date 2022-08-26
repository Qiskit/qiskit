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

from copy import copy
from typing import Sequence
import random

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.opflow import PauliSumOp
from qiskit.primitives import BaseEstimator
from qiskit.quantum_info.operators.base_operator import BaseOperator

from .base_estimator_gradient import BaseEstimatorGradient
from .estimator_gradient_result import EstimatorGradientResult


class SPSAEstimatorGradient(BaseEstimatorGradient):
    """
    Compute the gradients of the expectation value by the Simultaneous Perturbation Stochastic
    Approximation (SPSA).
    """

    def __init__(
        self,
        estimator: BaseEstimator,
        epsilon: float = 1e-2,
        seed: int | None = None,
        **run_options,
    ):
        """
        Args:
            estimator: The estimator used to compute the gradients.
            epsilon: The offset size for the finite difference gradients.
            seed: The seed for a random perturbation vector.
            run_options: Backend runtime options used for circuit execution. The order of priority is:
                run_options in `run` method > gradient's default run_options > primitive's default
                setting. Higher priority setting overrides lower priority setting."""
        self._epsilon = epsilon
        self._seed = random.seed(seed) if seed else None

        super().__init__(estimator, **run_options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        observables: Sequence[BaseOperator | PauliSumOp],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
        **run_options,
    ) -> EstimatorGradientResult:
        """Compute the estimator gradients on the given circuits."""
        # if parameters is none, all parameters in each circuit are differentiated.
        if parameters is None:
            parameters = [None for _ in range(len(circuits))]

        jobs, result_indices_all, offsets = [], [], []
        for circuit, observable, parameter_values_, parameters_ in zip(
            circuits, observables, parameter_values, parameters
        ):
            # indices of parameters to be differentiated
            if parameters_ is None:
                indices = list(range(circuit.num_parameters))
            else:
                indices = [circuit.parameters.data.index(p) for p in parameters_]
            result_indices_all.append(indices)

            offset = np.array(
                [(-1) ** (random.randint(0, 1)) for _ in range(len(circuit.parameters))]
            )
            plus = parameter_values_ + self._epsilon * offset
            minus = parameter_values_ - self._epsilon * offset
            offsets.append(offset)

            job = self._estimator.run([circuit] * 2, [observable] * 2, [plus, minus], **run_options)
            jobs.append(job)

        # combine the results
        results = [job.result() for job in jobs]
        gradients, metadata_ = [], []
        for i, result in enumerate(results):
            d = copy(run_options)
            n = len(result.values) // 2  # is always a multiple of 2
            gradient_ = (result.values[:n] - result.values[n:]) / (2 * self._epsilon * offsets[i])
            indices = result_indices_all[i]
            gradient = np.zeros(circuits[i].num_parameters)
            gradient[indices] = gradient_[indices]
            gradients.append(gradient)
            d['gradient_variance'] = np.var(gradient_)
            metadata_.append(result.metadata)
        return EstimatorGradientResult(values=gradients, metadata=metadata_)
