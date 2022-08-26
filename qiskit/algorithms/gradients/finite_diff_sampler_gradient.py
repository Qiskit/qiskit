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

from collections import Counter
from typing import Sequence

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit.result import QuasiDistribution

from .base_sampler_gradient import BaseSamplerGradient
from .sampler_gradient_result import SamplerGradientResult


class FiniteDiffSamplerGradient(BaseSamplerGradient):
    """Compute the gradients of the sampling probability by finite difference method."""

    def __init__(
        self,
        sampler: BaseSampler,
        epsilon: float = 1e-2,
        **run_options,
    ):
        """
        Args:
            sampler: The sampler used to compute the gradients.
            epsilon: The offset size for the finite difference gradients.
            run_options: Backend runtime options used for circuit execution. The order of priority is:
                run_options in `run` method > gradient's default run_options > primitive's default
                setting. Higher priority setting overrides lower priority setting.
        """

        self._epsilon = epsilon
        super().__init__(sampler, **run_options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
        **run_options,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        # if parameters is none, all parameters in each circuit are differentiated.
        if parameters is None:
            parameters = [None for _ in range(len(circuits))]

        jobs, result_indices_all = [], []
        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):
            # indices of parameters to be differentiated
            if parameters_ is None:
                indices = list(range(circuit.num_parameters))
            else:
                indices = [circuit.parameters.data.index(p) for p in parameters_]
            result_indices_all.append(indices)

            offset = np.identity(circuit.num_parameters)[indices, :]
            plus = parameter_values_ + self._epsilon * offset
            minus = parameter_values_ - self._epsilon * offset
            n = 2 * len(indices)

            job = self._sampler.run([circuit] * n, plus.tolist() + minus.tolist(), **run_options)
            jobs.append(job)

        # combine the results
        results = [job.result() for job in jobs]
        gradients = []
        for i, result in enumerate(results):
            n = len(result.quasi_dists) // 2
            dists = [Counter() for _ in range(circuits[i].num_parameters)]
            for j, idx in enumerate(result_indices_all[i]):
                # plus
                dists[idx].update(
                    Counter({k: v / (2 * self._epsilon) for k, v in result.quasi_dists[j].items()})
                )
                # minus
                dists[idx].update(
                    Counter(
                        {
                            k: -1 * v / (2 * self._epsilon)
                            for k, v in result.quasi_dists[j + n].items()
                        }
                    )
                )

            gradients.append([QuasiDistribution(dist) for dist in dists])
        return SamplerGradientResult(quasi_dists=gradients, metadata=run_options)
