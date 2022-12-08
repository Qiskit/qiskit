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
        parameters: Sequence[Sequence[Parameter] | None],
        **options,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        jobs, metadata_ = [], []
        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):
            # indices of parameters to be differentiated
            if parameters_ is None:
                indices = list(range(circuit.num_parameters))
            else:
                indices = [circuit.parameters.data.index(p) for p in parameters_]
            metadata_.append({"parameters": [circuit.parameters[idx] for idx in indices]})
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

        # combine the results
        try:
            results = [job.result() for job in jobs]
        except Exception as exc:
            raise AlgorithmError("Sampler job failed.") from exc

        gradients = []
        for i, result in enumerate(results):
            if self._method == "central":
                n = len(result.quasi_dists) // 2
                gradient_ = []
                for dist_plus, dist_minus in zip(result.quasi_dists[:n], result.quasi_dists[n:]):
                    grad_dist = np.zeros(2 ** circuits[i].num_qubits)
                    grad_dist[list(dist_plus.keys())] += list(dist_plus.values())
                    grad_dist[list(dist_minus.keys())] -= list(dist_minus.values())
                    grad_dist /= 2 * self._epsilon
                    gradient_.append(dict(enumerate(grad_dist)))
            elif self._method == "forward":
                gradient_ = []
                dist_zero = result.quasi_dists[0]
                for dist_plus in result.quasi_dists[1:]:
                    grad_dist = np.zeros(2 ** circuits[i].num_qubits)
                    grad_dist[list(dist_plus.keys())] += list(dist_plus.values())
                    grad_dist[list(dist_zero.keys())] -= list(dist_zero.values())
                    grad_dist /= self._epsilon
                    gradient_.append(dict(enumerate(grad_dist)))
            elif self._method == "backward":
                gradient_ = []
                dist_zero = result.quasi_dists[0]
                for dist_minus in result.quasi_dists[1:]:
                    grad_dist = np.zeros(2 ** circuits[i].num_qubits)
                    grad_dist[list(dist_zero.keys())] += list(dist_zero.values())
                    grad_dist[list(dist_minus.keys())] -= list(dist_minus.values())
                    grad_dist /= self._epsilon
                    gradient_.append(dict(enumerate(grad_dist)))
            gradients.append(gradient_)

        opt = self._get_local_options(options)
        return SamplerGradientResult(gradients=gradients, metadata=metadata_, options=opt)
