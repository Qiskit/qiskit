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

import sys
from collections import defaultdict
from typing import Sequence

import numpy as np

from qiskit.algorithms import AlgorithmError
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit.providers import Options

from .base_sampler_gradient import BaseSamplerGradient
from .sampler_gradient_result import SamplerGradientResult

if sys.version_info >= (3, 8):
    # pylint: disable=no-name-in-module, ungrouped-imports
    from typing import Literal
else:
    from typing_extensions import Literal


class FiniteDiffSamplerGradient(BaseSamplerGradient):
    """Compute the gradients of the sampling probability by finite difference method."""

    def __init__(
        self,
        sampler: BaseSampler,
        epsilon: float,
        options: Options | None = None,
        *,
        method: Literal["central", "forward", "backward"] = "central",
    ):
        r"""
        Args:
            sampler: The sampler used to compute the gradients.
            epsilon: The offset size for the finite difference gradients.
            options: Primitive backend runtime options used for circuit execution.
                The order of priority is: options in ``run`` method > gradient's
                default options > primitive's default setting.
                Higher priority setting overrides lower priority setting
            method: The computation method of the gradients.

                    - ``central`` computes :math:`\frac{f(x+e)-f(x-e)}{2e}`,
                    - ``forward`` computes :math:`\frac{f(x+e) - f(x)}{e}`,
                    - ``backward`` computes :math:`\frac{f(x)-f(x-e)}{e}`

                where :math:`e` is epsilon.

        Raises:
            ValueError: If ``epsilon`` is not positive.
            TypeError: If ``method`` is invalid.
        """
        if epsilon <= 0:
            raise ValueError(f"epsilon ({epsilon}) should be positive.")
        self._epsilon = epsilon
        if method not in ("central", "forward", "backward"):
            raise TypeError(
                f"The argument method should be central, forward, or backward: {method} is given."
            )
        self._method = method
        super().__init__(sampler, options)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameter_sets: Sequence[set[Parameter]],
        **options,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        jobs, metadata = [], []
        for circuit, parameter_values_, parameter_set in zip(
            circuits, parameter_values, parameter_sets
        ):
            # Indices of parameters to be differentiated
            indices = [
                circuit.parameters.data.index(p) for p in circuit.parameters if p in parameter_set
            ]
            metadata.append({"parameters": [circuit.parameters[idx] for idx in indices]})
            offset = np.identity(circuit.num_parameters)[indices, :]
            if self._method == "central":
                plus = parameter_values_ + self._epsilon * offset
                minus = parameter_values_ - self._epsilon * offset
                n = 2 * len(indices)
                job = self._sampler.run([circuit] * n, plus.tolist() + minus.tolist(), **options)
            elif self._method == "forward":
                plus = parameter_values_ + self._epsilon * offset
                n = len(indices) + 1
                job = self._sampler.run(
                    [circuit] * n, [parameter_values_] + plus.tolist(), **options
                )
            elif self._method == "backward":
                minus = parameter_values_ - self._epsilon * offset
                n = len(indices) + 1
                job = self._sampler.run(
                    [circuit] * n, [parameter_values_] + minus.tolist(), **options
                )
            jobs.append(job)

        # Compute the gradients
        try:
            results = [job.result() for job in jobs]
        except Exception as exc:
            raise AlgorithmError("Sampler job failed.") from exc

        gradients = []
        for result in results:
            if self._method == "central":
                n = len(result.quasi_dists) // 2
                gradient = []
                for dist_plus, dist_minus in zip(result.quasi_dists[:n], result.quasi_dists[n:]):
                    grad_dist = defaultdict(float)
                    for key, value in dist_plus.items():
                        grad_dist[key] += value / (2 * self._epsilon)
                    for key, value in dist_minus.items():
                        grad_dist[key] -= value / (2 * self._epsilon)
                    gradient.append(dict(grad_dist))
            elif self._method == "forward":
                gradient = []
                dist_zero = result.quasi_dists[0]
                for dist_plus in result.quasi_dists[1:]:
                    grad_dist = defaultdict(float)
                    for key, value in dist_plus.items():
                        grad_dist[key] += value / self._epsilon
                    for key, value in dist_zero.items():
                        grad_dist[key] -= value / self._epsilon
                    gradient.append(dict(grad_dist))
            elif self._method == "backward":
                gradient = []
                dist_zero = result.quasi_dists[0]
                for dist_minus in result.quasi_dists[1:]:
                    grad_dist = defaultdict(float)
                    for key, value in dist_zero.items():
                        grad_dist[key] += value / self._epsilon
                    for key, value in dist_minus.items():
                        grad_dist[key] -= value / self._epsilon
                    gradient.append(dict(grad_dist))
            gradients.append(gradient)

        opt = self._get_local_options(options)
        return SamplerGradientResult(gradients=gradients, metadata=metadata, options=opt)
