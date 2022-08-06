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
Gradient of probabilities with linear combination of unitaries (LCU)
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Sequence, Type

import numpy as np

from qiskit import transpile
from qiskit.circuit import (
    ClassicalRegister,
    Gate,
    Instruction,
    Parameter,
    ParameterExpression,
    QuantumCircuit,
    QuantumRegister,
)
from qiskit.circuit.library.standard_gates import (
    CXGate,
    CYGate,
    CZGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZXGate,
    RZZGate,
)
from qiskit.result import QuasiDistribution

from .sampler_gradient_result import SamplerGradientResult
from .utils import make_gradient_circuit_lin_comb
from ..base_sampler import BaseSampler
from ..sampler_result import SamplerResult
from ..utils import init_circuit

@dataclass
class SubEstimator:
    coeff: float | ParameterExpression
    circuit: QuantumCircuit
    index: int


class LinCombSamplerGradientNew:
    """LCU sampler gradient"""

    def __init__(self, sampler: Type[BaseSampler], circuits: QuantumCircuit | Sequence[QuantumCircuit]):

        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        print(circuits)
        circuits = tuple(init_circuit(circuit) for circuit in circuits)

        self._gradient_circuit_data_dict = {}
        self._circuits = circuits

        for i, circuit in enumerate(circuits):
            self._gradient_circuit_data_dict[i] = make_gradient_circuit_lin_comb(circuit, add_measurement=True)

        idx = 0
        gradient_circuits = []
        print(self._gradient_circuit_data_dict.items())
        for i, grad_circuit_data in self._gradient_circuit_data_dict.items():
            print(grad_circuit_data)
            for param in self._circuits[i].parameters:
                for grad in grad_circuit_data[param]:
                    grad.index = idx
                    gradient_circuits.append(grad.gradient_circuit)
                    idx += 1

        # TODO: this should be modified to add new gradient circuits after new primitives change
        # call rebuild_circuits_with_unique_parameters when first time calculating the gradient for a circuit
        self._sampler = sampler(circuits=gradient_circuits)

        print(gradient_circuits[0])


    def __call__(
        self,
        circuits: Sequence[int | QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        partial: Sequence[Sequence[Parameter]] | None = None,
        **run_options,
    ) -> SamplerGradientResult:

        partial = partial or [[] for _ in range(len(circuits))]

        gradients = []
        for circuit_index, parameter_values_, partial_ in zip(circuits, parameter_values, partial):
            gradient_circuit_data = self._gradient_circuit_data_dict[circuit_index]
            circuit_parameters = self._circuits[circuit_index].parameters
            parameter_value_map = {}

            # a parameter set for the partial option
            parameters = partial_ or self._circuits[circuit_index].parameters
            param_set = set(parameters)

            result_index = 0
            result_index_map = defaultdict(list)
            circuit_indices = []
            # gradient circuit indices and result indices
            for i, param in enumerate(circuit_parameters):
                parameter_value_map[param] = parameter_values_[i]
                if not param in param_set:
                    continue
                for grad in gradient_circuit_data[param]:
                    circuit_indices.append(grad.index)
                    result_index_map[param].append(result_index)
                    result_index+=1
            gradient_parameter_values_list = [parameter_values_ for i in range(len(circuit_indices))]

            results = self._sampler.__call__(circuit_indices, gradient_parameter_values_list)

            param_set = set(parameters)
            dists = [Counter() for _ in range(len(parameter_values_))]
            num_bitstrings = 2**self._circuits[circuit_index].num_qubits
            i = 0
            for i, param in enumerate(circuit_parameters):
                if param not in param_set:
                    continue
                for j, grad in enumerate(gradient_circuit_data[param]):
                    coeff = grad.coeff
                    result_index = result_index_map[param][j]
                    # if coeff has parameters, substitute them with the given parameter values
                    if isinstance(coeff, ParameterExpression):
                        local_map = {p: parameter_value_map[p] for p in coeff.parameters}
                        bound_coeff = float(coeff.bind(local_map))
                    else:
                        bound_coeff = coeff
                    for k, v in results.quasi_dists[result_index].items():
                        sign, k2 = divmod(k, num_bitstrings)
                        dists[i][k2] += (-1) ** sign * bound_coeff * v

            gradients.append([QuasiDistribution(dist) for dist in dists])
        return SamplerGradientResult(quasi_dists=gradients, metadata=[{}] * len(gradients))
