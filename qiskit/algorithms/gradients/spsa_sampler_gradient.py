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

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseSampler

from .base_sampler_gradient import BaseSamplerGradient
from .sampler_gradient_result import SamplerGradientResult


class SPSASamplerGradient(BaseSamplerGradient):
    """
    Compute the gradients of the sampling probability by the Simultaneous Perturbation Stochastic
    Approximation (SPSA).
    """

    def __init__(
        self,
        sampler: BaseSampler,
        epsilon: float,
        batch_size: int = 1,
        seed: int | None = None,
        **run_options,
    ):
        """
        Args:
            sampler: The sampler used to compute the gradients.
            epsilon: The offset size for the SPSA gradients.
            batch_size: number of gradients to average.
            seed: The seed for a random perturbation vector.
            run_options: Backend runtime options used for circuit execution. The order of priority is:
                run_options in `run` method > gradient's default run_options > primitive's default
                setting. Higher priority setting overrides lower priority setting.

        Raises:
            ValueError: If ``epsilon`` is not positive.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon ({epsilon}) should be positive.")
        self._batch_size = batch_size
        self._epsilon = epsilon
        self._seed = np.random.default_rng(seed) if seed else np.random.default_rng()

        super().__init__(sampler, **run_options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        **run_options,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        jobs, offsets, metadata_ = [], [], []
        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):
            # indices of parameters to be differentiated
            if parameters_ is None:
                indices = list(range(circuit.num_parameters))
            else:
                indices = [circuit.parameters.data.index(p) for p in parameters_]
            metadata_.append({"parameters": [circuit.parameters[idx] for idx in indices]})

            offset = np.array(
                [
                    (-1) ** (self._seed.integers(0, 2, len(circuit.parameters)))
                    for _ in range(self._batch_size)
                ]
            )
            plus = [parameter_values_ + self._epsilon * offset_ for offset_ in offset]
            minus = [parameter_values_ - self._epsilon * offset_ for offset_ in offset]
            offsets.append(offset)

            job = self._sampler.run([circuit] * 2 * self._batch_size, plus + minus, **run_options)
            jobs.append(job)

        # combine the results
        results = [job.result() for job in jobs]
        gradients = []
        for i, result in enumerate(results):
            grad_dists = np.zeros((self._batch_size, 2 ** circuits[i].num_qubits))
            for j, (dist_plus, dist_minus) in enumerate(
                zip(result.quasi_dists[: self._batch_size], result.quasi_dists[self._batch_size :])
            ):
                grad_dists[j, list(dist_plus.keys())] += list(dist_plus.values())
                grad_dists[j, list(dist_minus.keys())] -= list(dist_minus.values())
            grad_dists /= 2 * self._epsilon
            gradient_ = []
            indices = [circuits[i].parameters.data.index(p) for p in metadata_[i]["parameters"]]
            for j in range(circuits[i].num_parameters):
                if not j in indices:
                    continue
                grad = np.mean(
                    np.array([delta * dist for dist, delta in zip(grad_dists, offsets[i][:, j])]),
                    axis=0,
                )
                gradient_.append(dict(enumerate(grad)))
            gradients.append(gradient_)

        # TODO: include primitive's run_options as well
        return SamplerGradientResult(
            gradients=gradients, metadata=metadata_, run_options=run_options
        )
