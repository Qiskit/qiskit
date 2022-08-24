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
import random

import numpy as np

from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit.result import QuasiDistribution

from .base_sampler_gradient import BaseSamplerGradient
from .sampler_gradient_result import SamplerGradientResult
from .utils import make_spsa_base_parameter_values


class SPSASamplerGradient(BaseSamplerGradient):
    """
    Gradient of Sampler with the Simultaneous Perturbation Stochastic Approximation (SPSA).
    """

    def __init__(
        self,
        sampler: BaseSampler,
        epsilon: float = 1e-6,
        seed: int | None = None,
        **run_options,
    ):
        """
        Args:
            sampler: The sampler used to compute the gradients.
            epsilon: The offset size for the finite difference gradients.
            seed: The seed for a random perturbation vector.
            run_options: Backend runtime options used for circuit execution. The order of priority is:
                run_options in `run` method > gradient's default run_options > primitive's default
                setting. Higher priority setting overrides lower priority setting."""
        self._epsilon = epsilon
        self._seed = random.seed(seed) if seed else None

        super().__init__(sampler, **run_options)

    def _evaluate(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
        **run_options,
    ) -> SamplerGradientResult:
        parameters = parameters or [None for _ in range(len(circuits))]
        gradients = []
        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):

            base_parameter_values_list = make_spsa_base_parameter_values(circuit, self._epsilon)
            circuit_parameters = circuit.parameters
            gradient_parameter_values = np.zeros(len(circuit_parameters))

            # a parameter set for the parameter option
            parameters = parameters_ or circuit_parameters
            param_set = set(parameters)

            gradient_parameter_values = np.array(parameter_values_)
            # add the given parameter values and the base parameter values
            gradient_parameter_values_list = [
                gradient_parameter_values + base_parameter_values
                for base_parameter_values in base_parameter_values_list
            ]
            gradient_circuits = [circuit] * len(gradient_parameter_values_list)
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
                            k: v / (2 * base_parameter_values_list[0][i])
                            for k, v in results.quasi_dists[0].items()
                        }
                    )
                )
                # minus
                dists[i].update(
                    Counter(
                        {
                            k: -1 * v / (2 * base_parameter_values_list[0][i])
                            for k, v in results.quasi_dists[1].items()
                        }
                    )
                )

            gradients.append([QuasiDistribution(dist) for dist in dists])

        return SamplerGradientResult(quasi_dists=gradients, metadata=run_options)
