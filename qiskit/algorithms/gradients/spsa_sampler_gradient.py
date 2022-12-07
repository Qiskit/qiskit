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
from qiskit.primitives import BaseSampler
from qiskit.providers import Options

from .base_sampler_gradient import BaseSamplerGradient
from .sampler_gradient_result import SamplerGradientResult


class SPSASamplerGradient(BaseSamplerGradient):
    """
    Compute the gradients of the sampling probability by the Simultaneous Perturbation Stochastic
    Approximation (SPSA) [1].

    **Reference:**
    [1] J. C. Spall, Adaptive stochastic approximation by the simultaneous perturbation method in
    IEEE Transactions on Automatic Control, vol. 45, no. 10, pp. 1839-1853, Oct 2020,
    `doi: 10.1109/TAC.2000.880982 <https://ieeexplore.ieee.org/document/880982>`_.
    """

    def __init__(
        self,
        sampler: BaseSampler,
        epsilon: float,
        batch_size: int = 1,
        seed: int | None = None,
        options: Options | None = None,
    ):
        """
        Args:
            sampler: The sampler used to compute the gradients.
            epsilon: The offset size for the SPSA gradients.
            batch_size: number of gradients to average.
            seed: The seed for a random perturbation vector.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting

        Raises:
            ValueError: If ``epsilon`` is not positive.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon ({epsilon}) should be positive.")
        self._batch_size = batch_size
        self._epsilon = epsilon
        self._seed = np.random.default_rng(seed)

        super().__init__(sampler, options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None],
        **options,
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

            job = self._sampler.run([circuit] * 2 * self._batch_size, plus + minus, **options)
            jobs.append(job)

        # combine the results
        try:
            results = [job.result() for job in jobs]
        except Exception as exc:
            raise AlgorithmError("Sampler job failed.") from exc

        gradients = []
        for i, result in enumerate(results):
            dist_diffs = np.zeros((self._batch_size, 2 ** circuits[i].num_qubits))
            for j, (dist_plus, dist_minus) in enumerate(
                zip(result.quasi_dists[: self._batch_size], result.quasi_dists[self._batch_size :])
            ):
                dist_diffs[j, list(dist_plus.keys())] += list(dist_plus.values())
                dist_diffs[j, list(dist_minus.keys())] -= list(dist_minus.values())
            dist_diffs /= 2 * self._epsilon
            gradient = []
            indices = [circuits[i].parameters.data.index(p) for p in metadata_[i]["parameters"]]
            for j in range(circuits[i].num_parameters):
                if not j in indices:
                    continue
                # the gradient for jth parameter is the average of the gradients of the jth parameter
                # for each batch.
                batch_gradients = np.array(
                    [offset * dist_diff for dist_diff, offset in zip(dist_diffs, offsets[i][:, j])]
                )
                gradient_j = np.mean(batch_gradients, axis=0)
                gradient.append(dict(enumerate(gradient_j)))
            gradients.append(gradient)

        opt = self._get_local_options(options)
        return SamplerGradientResult(gradients=gradients, metadata=metadata_, options=opt)
