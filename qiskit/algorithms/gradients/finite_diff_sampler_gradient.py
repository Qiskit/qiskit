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
from qiskit.primitives import BaseSampler, SamplerResult
from qiskit.result import QuasiDistribution

from .base_sampler_gradient import BaseSamplerGradient
from .sampler_gradient_job import SamplerGradientJob
from .utils import make_fin_diff_base_parameter_values


class FiniteDiffSamplerGradient(BaseSamplerGradient):
    """Compute the gradients of the sampling probability with the finite difference method."""

    def __init__(
        self,
        sampler: BaseSampler,
        epsilon: float = 1e-6,
    ):
        """
        Args:
            sampler: The sampler used to compute the gradients.
            epsilon: The offset size for the finite difference gradients.
        """

        self._epsilon = epsilon
        self._base_parameter_values_dict = {}
        super().__init__(sampler)

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        partial: Sequence[Sequence[Parameter]] | None = None,
        **run_options,
    ) -> SamplerGradientJob:
        partial = partial or [[] for _ in range(len(circuits))]
        gradients = []
        status = []
        for circuit, parameter_values_, partial_ in zip(circuits, parameter_values, partial):
            index = self._circuit_ids.get(id(circuit))
            if index is not None:
                circuit_index = index
            else:
                # if the given circuit a new one, make base parameter values for + and - epsilon
                circuit_index = len(self._circuits)
                self._circuit_ids[id(circuit)] = circuit_index
                self._base_parameter_values_dict[
                    circuit_index
                ] = make_fin_diff_base_parameter_values(circuit, self._epsilon)
                self._circuits.append(circuit)

            circuit_parameters = self._circuits[circuit_index].parameters
            base_parameter_values_list = []
            gradient_parameter_values = np.zeros(len(circuit_parameters))

            # a parameter set for the partial option
            parameters = partial_ or circuit_parameters
            param_set = set(parameters)

            result_index = 0
            result_index_map = {}
            # bring the base parameter values for parameters only in the partial parameter set.
            for i, param in enumerate(circuit_parameters):
                gradient_parameter_values[i] = parameter_values_[i]
                if param in param_set:
                    base_parameter_values_list.append(
                        self._base_parameter_values_dict[circuit_index][i * 2]
                    )
                    base_parameter_values_list.append(
                        self._base_parameter_values_dict[circuit_index][i * 2 + 1]
                    )
                    result_index_map[param] = result_index
                    result_index += 1
            # add the given parameter values and the base parameter values
            gradient_parameter_values_list = [
                gradient_parameter_values + base_parameter_values
                for base_parameter_values in base_parameter_values_list
            ]
            gradient_circuits = [self._circuits[circuit_index]] * len(
                gradient_parameter_values_list
            )

            job = self._sampler.run(
                gradient_circuits, gradient_parameter_values_list, **run_options
            )
            results = job.result()

            # Combines the results and coefficients to reconstruct the gradient values
            # for the original circuit parameters
            dists = [Counter() for _ in range(len(parameter_values_))]
            for i, param in enumerate(circuit_parameters):
                if param not in param_set:
                    continue
                # plus
                dists[i].update(
                    Counter(
                        {
                            k: v / (2 * self._epsilon)
                            for k, v in results.quasi_dists[result_index_map[param] * 2].items()
                        }
                    )
                )
                # minus
                dists[i].update(
                    Counter(
                        {
                            k: -1 * v / (2 * self._epsilon)
                            for k, v in results.quasi_dists[result_index_map[param] * 2 + 1].items()
                        }
                    )
                )

            gradients.append(
                SamplerResult([QuasiDistribution(dist) for dist in dists], metadata=run_options)
            )
            status.append(job.status())
        return SamplerGradientJob(results=gradients, status=status)
