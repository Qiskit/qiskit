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

from collections import defaultdict
from typing import Sequence, Type

import numpy as np

from qiskit.circuit import (
    Parameter,
    ParameterExpression,
    QuantumCircuit,

)

from qiskit.quantum_info import SparsePauliOp, Pauli

from .utils import make_lin_comb_gradient_circuit
from ..base_estimator import BaseEstimator
from .estimator_gradient_result import EstimatorGradientResult
from ..utils import init_circuit

Pauli_Z = Pauli("Z")


class LinCombEstimatorGradient:
    """LCU sampler gradient"""

    def __init__(self, estimator: Type[BaseEstimator], circuits: QuantumCircuit | Sequence[QuantumCircuit],
                observables: SparsePauliOp):

        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        circuits = tuple(init_circuit(circuit) for circuit in circuits)

        self._gradient_circuit_data_dict = {}
        self._circuits = circuits

        for i, circuit in enumerate(circuits):
            self._gradient_circuit_data_dict[i] = make_lin_comb_gradient_circuit(circuit)

        idx = 0
        gradient_circuits = []
        for i, grad_circuit_data in self._gradient_circuit_data_dict.items():
            for param in self._circuits[i].parameters:
                for grad in grad_circuit_data[param]:
                    grad.index = idx
                    gradient_circuits.append(grad.gradient_circuit)
                    idx += 1
        # TODO: support multiple observables
        self._observables = observables.expand(Pauli_Z)
        self._estimator = estimator

    def __call__(
        self,
        circuits: Sequence[int | QuantumCircuit],
        observables: Sequence[int | SparsePauliOp],
        parameter_values: Sequence[Sequence[float]],
        partial: Sequence[Sequence[Parameter]] | None = None,
        **run_options,
    ) -> EstimatorGradientResult:

        partial = partial or [[] for _ in range(len(circuits))]

        gradients = []
        for circuit_index, observable, parameter_values_, partial_ in zip(circuits, observables, parameter_values, partial):
            gradient_circuit_data = self._gradient_circuit_data_dict[circuit_index]
            circuit_parameters = self._circuits[circuit_index].parameters
            parameter_value_map = {}

            # a parameter set for the partial option
            parameters = partial_ or self._circuits[circuit_index].parameters
            param_set = set(parameters)

            result_index = 0
            result_index_map = defaultdict(list)
            gradient_circuits = []
            # gradient circuit indices and result indices
            for i, param in enumerate(circuit_parameters):
                parameter_value_map[param] = parameter_values_[i]
                if not param in param_set:
                    continue
                for grad in gradient_circuit_data[param]:
                    gradient_circuits.append(grad.gradient_circuit)
                    result_index_map[param].append(result_index)
                    result_index+=1
            gradient_parameter_values_list = [parameter_values_ for i in range(len(gradient_circuits))]
            observables_indices = [self._observables] * len(gradient_circuits)

            results = self._estimator.run(gradient_circuits, observables_indices, gradient_parameter_values_list).result()


            values = np.zeros(len(circuit_parameters))
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

                    values[i] +=  bound_coeff * results.values[result_index_map[param][j]]

            gradients.append(values)
        return EstimatorGradientResult(values=gradients, metadata=[{}] * len(gradients))
