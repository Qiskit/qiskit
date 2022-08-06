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
from qiskit.quantum_info import SparsePauliOp

from .utils import (
    ParameterShiftGradientCircuitData,
    make_gradient_circuit_param_shift,
    make_base_parameter_values_parameter_shift,
)
from ..base_estimator import BaseEstimator
from .estimator_gradient_result import EstimatorGradientResult
from ..utils import init_circuit

# @dataclass
# class SubSampler:
#     coeff: float | ParameterExpression
#     circuit: QuantumCircuit
#     index: int

# dataclass for g_circuit作って
# base_parameter_values_listとかcoeff_listとか全部入れた方がいいかも


class ParamShiftEstimatorGradientUniqueParameters:
    """Parameter shift estimator gradient"""

    SUPPORTED_GATES = ["x", "y", "z", "h", "rx", "ry", "rz", "p", "cx", "cy", "cz"]

    def __init__(
        self,
        estimator: Type[BaseEstimator],
        circuits: QuantumCircuit | Iterable[QuantumCircuit],
        observables: SparsePauliOp,
    ):

        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        circuits = tuple(init_circuit(circuit) for circuit in circuits)

        self._gradient_circuit_data_dict = {}
        self._circuits = circuits
        for i, circuit in enumerate(circuits):
            self._gradient_circuit_data_dict[i] = make_gradient_circuit_param_shift(circuit)

        self._base_parameter_values_dict = {}
        for k, gradient_circuit_data in self._gradient_circuit_data_dict.items():
            self._base_parameter_values_dict[k] = make_base_parameter_values_parameter_shift(
                gradient_circuit_data
            )

        # TODO: this should be modified to add new gradient circuits after new primitives change
        # call rebuild_circuits_with_unique_parameters when first time calculating the gradient for a circuit
        self._estimator = estimator(
            circuits=[
                gradient_circuit_data.gradient_circuit
                for _, gradient_circuit_data in self._gradient_circuit_data_dict.items()

            ], observables=observables
        )
        #print(self._estimator.__call__([0], [0],[[1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4,1,2]]))


    def __call__(
        self,
        circuits: Sequence[int | QuantumCircuit],
        observables: Sequence[int | SparsePauliOp],
        parameter_values: Sequence[float],
        partial: Sequence[Parameter] | None = None,
        **run_options,
    ) -> EstimatorGradientResult:

        partial = partial or [[] for _ in range(len(circuits))]

        gradients = []
        for circuit_index, observable, parameter_values_, partial_ in zip(circuits, observables, parameter_values, partial):

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
            observables_indices = [observable] * len(gradient_parameter_values_list)

            results = self._estimator.__call__(circuit_indices, observables_indices, gradient_parameter_values_list)

            # Combines the results and coefficients to reconstruct the gradient for the original circuit parameters

            values = np.zeros(len(parameter_values_))

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
                    values[i] += bound_coeff*results.values[result_index_map[g_param] * 2]
                    # minus
                    values[i] -= bound_coeff*results.values[result_index_map[g_param] * 2 + 1]
            gradients.append(values)
        return EstimatorGradientResult(values=gradients, metadata=[{}] * len(gradients))

