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

from collections.abc import Sequence

import numpy as np

from qiskit import QuantumCircuit
from qiskit.result import QuasiDistribution

from ..factories import SamplerFromCircuits
from ..sampler_result import SamplerResult
from .base_sampler_gradient import BaseSamplerGradient
from .sampler_gradient_result import SamplerGradientResult


class FiniteDiffSamplerGradient(BaseSamplerGradient):
    """
    Gradient of Sampler with Finite difference method.
    """

    def __init__(self, sampler_factory: SamplerFromCircuits, circuits, epsilon: float = 1e-6):
        """
        TODO: Write
        """
        self._num_circuits = len(circuits)
        sampler = sampler_factory(circuits)
        super().__init__(sampler)
        self._epsilon = epsilon

    def gradient(
        self,
        circuits: Sequence[int | QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerGradientResult:
        # TODO: support QC and SPO
        gradients = []
        for circuit_index, parameter_value in zip(circuits, parameter_values):
            dim = len(parameter_value)
            params = [parameter_value]
            for i in range(dim):
                ei = parameter_value.copy()
                ei[i] += self._epsilon
                params.append(ei)
            param_list = np.array(params).tolist()
            circuit_indices = [circuit_index] * (dim + 1)
            # TODO: batch
            results = self._sampler.__call__(circuit_indices, param_list, **run_options)

            quasi_dists = results.quasi_dists
            gradient = []
            f_ref = quasi_dists[0]
            for f_i in quasi_dists[1:]:
                gradient.append(
                    QuasiDistribution(
                        {key: (f_i[key] - f_ref[key]) / self._epsilon for key in f_ref}
                    )
                )
            gradients.append(gradient)
        return SamplerGradientResult(quasi_dists=gradient, metadata=[{}] * len(gradients))

    def sample(
        self,
        circuits: Sequence[int | QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        **run_options,
    ) -> SamplerResult:
        # TODO: support QC and SPO
        for circuit in circuits:
            if circuit >= self._num_circuits:
                raise IndexError()

        return self._sampler(circuits, parameter_values, **run_options)
