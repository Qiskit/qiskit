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
        seed: int | None = None,
        **run_options,
    ):
        """
        Args:
            sampler: The sampler used to compute the gradients.
            epsilon: The offset size for the SPSA gradients.
            seed: The seed for a random perturbation vector.
            run_options: Backend runtime options used for circuit execution. The order of priority is:
                run_options in `run` method > gradient's default run_options > primitive's default
                setting. Higher priority setting overrides lower priority setting.

        Raises:
            ValueError: If ``epsilon`` is not float.
        """
        if not isinstance(epsilon, float):
            raise ValueError(f"epsilon must be a float, but got {type(epsilon)} instead.")
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

            offset = (-1) ** (self._seed.integers(0, 2, len(circuit.parameters)))

            plus = parameter_values_ + self._epsilon * offset
            minus = parameter_values_ - self._epsilon * offset
            offsets.append(offset)

            job = self._sampler.run([circuit] * 2, [plus, minus], **run_options)
            jobs.append(job)

        # combine the results
        results = [job.result() for job in jobs]
        gradients = []
        for i, result in enumerate(results):
            grad_dists = np.zeros(2 ** circuits[i].num_qubits)
            dist_plus = result.quasi_dists[0]
            dist_minus = result.quasi_dists[1]
            grad_dists[list(dist_plus.keys())] += list(dist_plus.values())
            grad_dists[list(dist_minus.keys())] -= list(dist_minus.values())
            grad_dists /= 2 * self._epsilon

            indices = [circuits[i].parameters.data.index(p) for p in metadata_[i]["parameters"]]
            gradient_ = []

            gradient_.extend(
                {k: offsets[i][j] * dist for k, dist in enumerate(grad_dists)} for j in indices
            )

            gradients.append(gradient_)

        # TODO: include primitive's run_options as well
        return SamplerGradientResult(
            gradients=gradients, metadata=metadata_, run_options=run_options
        )
