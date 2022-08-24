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
"""
Gradient of probabilities with parameter shift
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.primitives import BaseSampler
from qiskit.result import QuasiDistribution

from .base_sampler_gradient import BaseSamplerGradient
from .sampler_gradient_result import SamplerGradientResult
from .utils import make_param_shift_base_parameter_values, make_param_shift_gradient_circuit_data


class ParamShiftSamplerGradient(BaseSamplerGradient):
    """Compute the gradients of the sampling probability with the parameter shift method."""

    def __init__(self, sampler: BaseSampler, **run_options):
        """
        Args:
            sampler: The sampler used to compute the gradients.
            run_options: Backend runtime options used for circuit execution. The order of priority is:
                run_options in `run` method > gradient's default run_options > primitive's default
                setting. Higher priority setting overrides lower priority setting.
        """
        self._gradient_circuit_data_dict = {}
        self._base_parameter_values_dict = {}
        super().__init__(sampler, **run_options)

    def _evaluate(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter]] | None = None,
        **run_options,
    ) -> SamplerGradientResult:
        parameters = parameters or [None for _ in range(len(circuits))]

        gradients = []
        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):
            index = self._circuit_ids.get(id(circuit))
            if index is not None:
                circuit_index = index
            else:
                # if the given circuit is  a new one, make gradient circuit data and
                # base parameter values
                circuit_index = len(self._circuits)
                self._circuit_ids[id(circuit)] = circuit_index
                self._gradient_circuit_data_dict[
                    circuit_index
                ] = make_param_shift_gradient_circuit_data(circuit)
                self._base_parameter_values_dict[
                    circuit_index
                ] = make_param_shift_base_parameter_values(
                    self._gradient_circuit_data_dict[circuit_index]
                )
                self._circuits.append(circuit)

            gradient_circuit_data = self._gradient_circuit_data_dict[circuit_index]
            gradient_parameter_map = gradient_circuit_data.gradient_parameter_map
            gradient_parameter_index_map = gradient_circuit_data.gradient_parameter_index_map
            circuit_parameters = self._circuits[circuit_index].parameters
            parameter_value_map = {}
            base_parameter_values_list = []
            gradient_parameter_values = np.zeros(
                len(gradient_circuit_data.gradient_circuit.parameters)
            )

            # a parameter set for the parameter option
            parameters = parameters_ or self._circuits[circuit_index].parameters
            param_set = set(parameters)

            result_index = 0
            result_index_map = {}
            # bring the base parameter values for parameters only in the specified parameter set.
            for i, param in enumerate(circuit_parameters):
                parameter_value_map[param] = parameter_values_[i]
                for g_param in gradient_parameter_map[param]:
                    g_param_idx = gradient_parameter_index_map[g_param]
                    gradient_parameter_values[g_param_idx] = parameter_values_[i]
                    if param in param_set:
                        base_parameter_values_list.append(
                            self._base_parameter_values_dict[circuit_index][g_param_idx * 2]
                        )
                        base_parameter_values_list.append(
                            self._base_parameter_values_dict[circuit_index][g_param_idx * 2 + 1]
                        )
                        result_index_map[g_param] = result_index
                        result_index += 1
            # add the given parameter values and the base parameter values
            gradient_parameter_values_list = [
                gradient_parameter_values + base_parameter_values
                for base_parameter_values in base_parameter_values_list
            ]
            gradient_circuits = [gradient_circuit_data.gradient_circuit] * len(
                gradient_parameter_values_list
            )

            job = self._sampler.run(
                gradient_circuits, gradient_parameter_values_list, **run_options
            )
            results = job.result()

            # Combines the results and coefficients to reconstruct the gradient
            # for the original circuit parameters
            dists = [Counter() for _ in range(len(parameter_values_))]
            for i, param in enumerate(circuit_parameters):
                if param not in param_set:
                    continue
                for g_param in gradient_parameter_map[param]:
                    g_param_idx = gradient_parameter_index_map[g_param]
                    coeff = gradient_circuit_data.coeff_map[g_param] / 2
                    # if coeff has parameters, substitute them with the given parameter values
                    if isinstance(coeff, ParameterExpression):
                        local_map = {p: parameter_value_map[p] for p in coeff.parameters}
                        bound_coeff = float(coeff.bind(local_map))
                    else:
                        bound_coeff = coeff
                    # plus
                    dists[i].update(
                        Counter(
                            {
                                k: bound_coeff * v
                                for k, v in results.quasi_dists[
                                    result_index_map[g_param] * 2
                                ].items()
                            }
                        )
                    )
                    # minus
                    dists[i].update(
                        Counter(
                            {
                                k: -1 * bound_coeff * v
                                for k, v in results.quasi_dists[
                                    result_index_map[g_param] * 2 + 1
                                ].items()
                            }
                        )
                    )
            gradients.append([QuasiDistribution(dist) for dist in dists])

        return SamplerGradientResult(quasi_dists=gradients, metadata=run_options)
