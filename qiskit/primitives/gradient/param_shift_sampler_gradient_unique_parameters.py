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

from copy import copy, deepcopy
from collections import Iterable, Counter, defaultdict
from dataclasses import dataclass
from email.mime import base
from hashlib import new
from typing import Sequence, Type

import numpy as np

from qiskit import transpile
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit
from qiskit.result import QuasiDistribution

from .sampler_gradient_result import SamplerGradientResult
from .utils import GradientCircuitData, rebuild_circuit_with_unique_parameters, make_base_parameter_values_parameter_shift
from ..base_sampler import BaseSampler
from ..sampler_result import SamplerResult
from ..utils import init_circuit


class ParamShiftSamplerGradientUniqueParameters:
    """Parameter shift estimator gradient"""

    def __init__(self, sampler: Type[BaseSampler], circuits: QuantumCircuit | Iterable[QuantumCircuit]):
        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        circuits = tuple(init_circuit(circuit) for circuit in circuits)

        self._circuits = circuits

        self._gradient_circuit_data_dict = {}
        for i, circuit in enumerate(circuits):
            self._gradient_circuit_data_dict[i] = rebuild_circuit_with_unique_parameters(circuit)

        self._base_parameter_values_dict = {}
        for k, gradient_circuit_data in self._gradient_circuit_data_dict.items():
            self._base_parameter_values_dict[k] = make_base_parameter_values_parameter_shift(gradient_circuit_data)

        # TODO: this should be modified to add new gradient circuits after new primitives change
        # call rebuild_circuits_with_unique_parameters when first time calculating the gradient for a circuit
        self._sampler = sampler(circuits=[gradient_circuit_data.gradient_circuit for _, gradient_circuit_data in self._gradient_circuit_data_dict.items()])

    def __call__(
        self,
        circuits: Sequence[int | QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        partial: Sequence[Sequence[Parameter]] | None = None,
        **run_options,
    ) -> SamplerResult:

        partial = partial or [[] for _ in range(len(circuits))]

        gradients = []
        for circuit_index, parameter_values_, partial_ in zip(circuits, parameter_values, partial):

            gradient_circuit_data = self._gradient_circuit_data_dict[circuit_index]
            gradient_parameter_map = gradient_circuit_data.gradient_parameter_map
            gradient_parameter_index_map = gradient_circuit_data.gradient_parameter_index_map
            circuit_parameters = self._circuits[circuit_index].parameters
            parameter_value_map = {}
            base_parameter_values_list = []
            gradient_parameter_values = np.zeros(len(gradient_circuit_data.gradient_circuit.parameters))

            # a parameter set for the partial option
            parameters = partial_ or self._circuits[circuit_index].parameters
            param_set = set(parameters)

            result_index = 0
            result_index_map = {}
            # bring the base parameter values for parameters only in the partial parameter set.
            for i, param in enumerate(circuit_parameters):
                parameter_value_map[param] = parameter_values_[i]
                for g_param in gradient_parameter_map[param]:
                    g_param_idx = gradient_parameter_index_map[g_param]
                    gradient_parameter_values[g_param_idx] = parameter_values_[i]
                    if param in param_set:
                        base_parameter_values_list.append(self._base_parameter_values_dict[circuit_index][g_param_idx * 2])
                        base_parameter_values_list.append(self._base_parameter_values_dict[circuit_index][g_param_idx * 2 + 1])
                        result_index_map[g_param] = result_index
                        result_index += 1
            # add the given parameter values and the base parameter values
            gradient_parameter_values_list = [gradient_parameter_values + base_parameter_values for base_parameter_values in base_parameter_values_list]
            circuit_indices = [circuit_index] * len(gradient_parameter_values_list)

            results = self._sampler.__call__(circuit_indices, gradient_parameter_values_list)

            # Combines the results and coefficients to reconstruct the gradient for the original circuit parameters

            dists = [Counter() for _ in range(len(parameter_values_))]
            print(dists)
            for i, param in enumerate(circuit_parameters):
                if param not in param_set:
                    continue
                for g_param in gradient_parameter_map[param]:
                    g_param_idx = gradient_parameter_index_map[g_param]
                    coeff = gradient_circuit_data.coeff_map[g_param]/2
                    # if coeff has parameters, substitute them with the given parameter values
                    if isinstance(coeff, ParameterExpression):
                        local_map = {p: parameter_value_map[p] for p in coeff.parameters}
                        bound_coeff = float(coeff.bind(local_map))
                    else:
                        bound_coeff = coeff
                    # plus
                    dists[i].update(
                            Counter({k: bound_coeff * v for k, v in results.quasi_dists[result_index_map[g_param] * 2].items()})
                    )
                    # minus
                    dists[i].update(
                            Counter({k: -1 * bound_coeff * v for k, v in results.quasi_dists[result_index_map[g_param] * 2 + 1].items()})
                    )
            gradients.append([QuasiDistribution(dist) for dist in dists])
        return SamplerGradientResult(quasi_dists=gradients, metadata=[{}] * len(gradients))
