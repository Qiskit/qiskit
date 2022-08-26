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

from copy import copy
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
    """Compute the gradients of the sampling probability by the parameter shift rule."""

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

    def _run(
        self,
        circuits: Sequence[QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        parameters: Sequence[Sequence[Parameter] | None] | None = None,
        **run_options,
    ) -> SamplerGradientResult:
        """Compute the sampler gradients on the given circuits."""
        # if parameters is none, all parameters in each circuit are differentiated.
        if parameters is None:
            parameters = [None for _ in range(len(circuits))]

        jobs, result_indices_all, coeffs_all = [], [], []
        for circuit, parameter_values_, parameters_ in zip(circuits, parameter_values, parameters):
            # a set of parameters to be differentiated
            if parameters_ is None:
                param_set = set(circuit.parameters)
            else:
                param_set = set(parameters_)

            gradient_circuit_data = self._gradient_circuit_data_dict.get(id(circuit))
            base_parameter_values_all = self._base_parameter_values_dict.get(id(circuit))

            if gradient_circuit_data is None and base_parameter_values_all is None:
                gradient_circuit_data = make_param_shift_gradient_circuit_data(circuit)
                self._gradient_circuit_data_dict[id(circuit)] = gradient_circuit_data
                base_parameter_values_all = make_param_shift_base_parameter_values(
                    gradient_circuit_data
                )
                self._base_parameter_values_dict[id(circuit)] = base_parameter_values_all

            plus_offsets, minus_offsets = [], []
            gradient_circuit = gradient_circuit_data.gradient_circuit
            gradient_parameter_values = np.zeros(
                len(gradient_circuit_data.gradient_circuit.parameters)
            )

            # only compute the gradients for parameters in the parameter set
            result_map = []
            coeffs = []
            for i, param in enumerate(circuit.parameters):
                g_params = gradient_circuit_data.gradient_parameter_map[param]
                indices = [gradient_circuit.parameters.data.index(g_param) for g_param in g_params]
                gradient_parameter_values[indices] = parameter_values_[i]
                if param in param_set:
                    plus_offsets.extend(base_parameter_values_all[idx] for idx in indices)
                    minus_offsets.extend(
                        base_parameter_values_all[idx + len(gradient_circuit.parameters)]
                        for idx in indices
                    )
                    result_map.extend(i for _ in range(len(indices)))
                    for g_param in g_params:
                        coeff = gradient_circuit_data.coeff_map[g_param]
                        # if coeff has parameters, we need to substitute
                        if isinstance(coeff, ParameterExpression):
                            local_map = {
                                p: parameter_values_[circuit.parameters.data.index(p)]
                                for p in coeff.parameters
                            }
                            bound_coeff = float(coeff.bind(local_map))
                        else:
                            bound_coeff = coeff
                        coeffs.append(bound_coeff / 2)

            # add the base parameter values to the parameter values
            gradient_parameter_values_plus = [
                gradient_parameter_values + plus_offset for plus_offset in plus_offsets
            ]
            gradient_parameter_values_minus = [
                gradient_parameter_values + minus_offset for minus_offset in minus_offsets
            ]
            n = 2 * len(gradient_parameter_values_plus)

            job = self._sampler.run(
                [gradient_circuit] * n,
                gradient_parameter_values_plus + gradient_parameter_values_minus,
                **run_options,
            )
            jobs.append(job)
            result_indices_all.append(result_map)
            coeffs_all.append(coeffs)

        # combine the results
        results = [job.result() for job in jobs]
        gradients, metadata_ = [], []
        for i, result in enumerate(results):
            n = len(result.quasi_dists) // 2
            dists = [Counter() for _ in range(circuits[i].num_parameters)]
            for j, (idx, coeff) in enumerate(zip(result_indices_all[i], coeffs_all[i])):
                # plus
                dists[idx].update(Counter({k: v * coeff for k, v in result.quasi_dists[j].items()}))
                # minus
                dists[idx].update(
                    Counter({k: -v * coeff for k, v in result.quasi_dists[j + n].items()})
                )
            gradients.append([QuasiDistribution(dist) for dist in dists])
        return SamplerGradientResult(quasi_dists=gradients, metadata=metadata_, run_options=run_options)
