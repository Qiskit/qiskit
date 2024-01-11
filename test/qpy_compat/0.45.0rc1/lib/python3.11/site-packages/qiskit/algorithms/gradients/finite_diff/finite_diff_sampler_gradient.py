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

from collections import defaultdict
from typing import Literal, Sequence

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit.providers import Options

from ..base.base_sampler_gradient import BaseSamplerGradient
from ..base.sampler_gradient_result import SamplerGradientResult

from ...exceptions import AlgorithmError


class FiniteDiffSamplerGradient(BaseSamplerGradient):
    """
    Compute the gradients of the sampling probability by finite difference method [1].

    **Reference:**
    [1] `Finite difference method <https://en.wikipedia.org/wiki/Finite_difference_method>`_
    """

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
        parameters: Sequence[Sequence[Parameter]],
        **options,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        job_circuits, job_param_values, metadata = [], [], []
        all_n = []
        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):
            # Indices of parameters to be differentiated
            indices = [circuit.parameters.data.index(p) for p in parameters_]
            metadata.append({"parameters": parameters_})
            # Combine inputs into a single job to reduce overhead.
            offset = np.identity(circuit.num_parameters)[indices, :]
            if self._method == "central":
                plus = parameter_values_ + self._epsilon * offset
                minus = parameter_values_ - self._epsilon * offset
                n = 2 * len(indices)
                job_circuits.extend([circuit] * n)
                job_param_values.extend(plus.tolist() + minus.tolist())
                all_n.append(n)
            elif self._method == "forward":
                plus = parameter_values_ + self._epsilon * offset
                n = len(indices) + 1
                job_circuits.extend([circuit] * n)
                job_param_values.extend([parameter_values_] + plus.tolist())
                all_n.append(n)
            elif self._method == "backward":
                minus = parameter_values_ - self._epsilon * offset
                n = len(indices) + 1
                job_circuits.extend([circuit] * n)
                job_param_values.extend([parameter_values_] + minus.tolist())
                all_n.append(n)

        # Run the single job with all circuits.
        job = self._sampler.run(job_circuits, job_param_values, **options)
        try:
            results = job.result()
        except Exception as exc:
            raise AlgorithmError("Sampler job failed.") from exc

        # Compute the gradients.
        gradients = []
        partial_sum_n = 0
        for n in all_n:
            gradient = []
            if self._method == "central":
                result = results.quasi_dists[partial_sum_n : partial_sum_n + n]
                for dist_plus, dist_minus in zip(result[: n // 2], result[n // 2 :]):
                    grad_dist: dict[int, float] = defaultdict(float)
                    for key, value in dist_plus.items():
                        grad_dist[key] += value / (2 * self._epsilon)
                    for key, value in dist_minus.items():
                        grad_dist[key] -= value / (2 * self._epsilon)
                    gradient.append(dict(grad_dist))
            elif self._method == "forward":
                result = results.quasi_dists[partial_sum_n : partial_sum_n + n]
                dist_zero = result[0]
                for dist_plus in result[1:]:
                    grad_dist = defaultdict(float)
                    for key, value in dist_plus.items():
                        grad_dist[key] += value / self._epsilon
                    for key, value in dist_zero.items():
                        grad_dist[key] -= value / self._epsilon
                    gradient.append(dict(grad_dist))

            elif self._method == "backward":
                result = results.quasi_dists[partial_sum_n : partial_sum_n + n]
                dist_zero = result[0]
                for dist_minus in result[1:]:
                    grad_dist = defaultdict(float)
                    for key, value in dist_zero.items():
                        grad_dist[key] += value / self._epsilon
                    for key, value in dist_minus.items():
                        grad_dist[key] -= value / self._epsilon
                    gradient.append(dict(grad_dist))

            partial_sum_n += n
            gradients.append(gradient)

        opt = self._get_local_options(options)
        return SamplerGradientResult(gradients=gradients, metadata=metadata, options=opt)
