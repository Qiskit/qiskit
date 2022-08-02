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

from ..base_estimator import BaseEstimator
from .estimator_gradient_result import EstimatorGradientResult
from ..utils import init_circuit

# @dataclass
# class SubSampler:
#     coeff: float | ParameterExpression
#     circuit: QuantumCircuit
#     index: int

#dataclass for g_circuit作って
#base_parameter_values_listとかcoeff_listとか全部入れた方がいいかも

class ParamShiftEstimatorGradientUniqueParameters:
    """Parameter shift estimator gradient"""

    SUPPORTED_GATES = ["x", "y", "z", "h", "rx", "ry", "rz", "p", "cx", "cy", "cz"]

    def __init__(self,
                estimator: Type[BaseEstimator],
                circuits: QuantumCircuit | Iterable[QuantumCircuit],
                observables: BaseOperator | PauliSumOp | Sequence[BaseOperator | PauliSumOp]):

        if isinstance(circuits, QuantumCircuit):
            circuits = (circuits,)
        circuits = tuple(init_circuit(circuit) for circuit in circuits)

        self._circuits = circuits
        self._g_circuit_dict = {}
        self._rebuild_circuits_with_unique_parameters()
        self._prepare_base_parameter_values()
        # TODO: this should be modified to add new gradient circuits after new primitives change
        self._estimator = estimator(circuits=[self._g_circuit_dict[i]['g_circuit'] for i in range(len(circuits))], observables=observables)
#        print(self._sampler)
#        return
        # self._same_param_dict = defaultdict(list)
        # self._grad = self._preprocessing()
        # circuits = [self._circuit]
        # #print('self._glad: ',self._grad.items())
        # for param, lst in self._grad.items():
        #     #print(param, lst)
        #     for arg in lst:
        #         circuits.append(arg.circuit)



    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        pass

    def _rebuild_circuits_with_unique_parameters(self):
        for circuit_idx, circuit in enumerate(self._circuits):
            g_circuit = circuit.copy_empty_like(f'g_{circuit.name}')
            param_inst_dict = defaultdict(list)
            parameter_map = defaultdict(list)
            coeff_map = {}

            for inst in circuit.data:
                new_inst = deepcopy(inst)
                qubit_idx = circuit.qubits.index(inst[1][0])
                new_inst.qubits = (g_circuit.qubits[qubit_idx],)

                # Assign new unique parameters when the instruction is a parameterized.
                if inst.operation.is_parameterized():
                    parameters = inst.operation.params
                    new_inst_parameters = []
                    # For a gate with multiple parameters e.g. a U gate
                    for parameter in parameters:
                        subs_map = {}
                        # For a gate parameter with multiple parameter variables.
                        # e.g. ry(θ) with θ = (2x + y)
                        for parameter_variable in parameter.parameters:
                            if parameter_variable in param_inst_dict:
                                new_parameter_variable = Parameter(f'g{parameter_variable.name}_{len(param_inst_dict[parameter_variable])+1}')
                            else:
                                new_parameter_variable = Parameter(f'g{parameter_variable.name}_1')
                            subs_map[parameter_variable] = new_parameter_variable
                            param_inst_dict[parameter_variable].append(inst)
                            parameter_map[parameter_variable].append(new_parameter_variable)
                            # Coefficient to calculate derivative i.e. dt/dw in df/dt * dt/dw
                            coeff_map[new_parameter_variable] = parameter.gradient(parameter_variable)
                        # Substitute the parameter variables with the corresponding new parameter variables in ``subs_map``.
                        new_parameter = parameter.subs(subs_map)
                        new_inst_parameters.append(new_parameter)
                        print(new_inst_parameters)
                    new_inst.operation.params = new_inst_parameters
                g_circuit.append(new_inst)
                print(new_inst)
            # for global phase
            subs_map = {}
            if isinstance(g_circuit.global_phase, ParameterExpression):
                for parameter_variable in g_circuit.global_phase.parameters:
                    if parameter_variable in param_inst_dict:
                        new_parameter_variable = parameter_map[parameter_variable][0]
                    else:
                        new_parameter_variable = Parameter(f'g{parameter_variable.name}_1')
                    subs_map[parameter_variable] = new_parameter_variable
                g_circuit.global_phase = g_circuit.global_phase.subs(subs_map)
            print(g_circuit.draw())
            print(g_circuit.num_parameters)
            self._g_circuit_dict[circuit_idx] = {'g_circuit': g_circuit,'parameter_map': parameter_map, 'coeff_map': coeff_map}


    def _prepare_base_parameter_values(self):
        # construct internal parameter values for the parameter shift
        for circuit_idx, circuit in enumerate(self._circuits):
            parameter_map = self._g_circuit_dict[circuit_idx]['parameter_map']
            g_circuit = self._g_circuit_dict[circuit_idx]['g_circuit']
            coeff_map =  self._g_circuit_dict[circuit_idx]['coeff_map']
            g_param_index_map = {}

            for i, g_param in enumerate(g_circuit.parameters):
                g_param_index_map[g_param] = i
            self._g_circuit_dict[circuit_idx]['g_param_index_map'] = g_param_index_map

            base_parameter_values = []
            base_coeffs = []

            # for i, param in enumerate(circuit.parameters):
            #     sub_parameter_values = []
            #     coeffs = []
            #     for g_param in parameter_map[param]:
            #         # prepare base decomposed parameter values for each original parameter
            #         g_param_idx = g_param_index_map[g_param]
            #         # for + pi/2 in the parameter shift rule
            #         parameter_values_plus = np.zeros(len(g_circuit.parameters))
            #         parameter_values_plus[g_param_idx] += np.pi/2
            #         # for - pi/2 in the parameter shift rule
            #         parameter_values_minus = np.zeros(len(g_circuit.parameters))
            #         parameter_values_minus[g_param_idx] -= np.pi/2
            #         sub_parameter_values.append((parameter_values_plus, parameter_values_minus))
            #         coeffs.append((coeff_map[g_param]/2, -1*coeff_map[g_param]/2))
            #     base_parameter_values.append(sub_parameter_values)
            #     base_coeffs.append(coeffs)
            for i, param in enumerate(circuit.parameters):
                for g_param in parameter_map[param]:
                    # prepare base decomposed parameter values for each original parameter
                    g_param_idx = g_param_index_map[g_param]
                    # for + pi/2 in the parameter shift rule
                    parameter_values_plus = np.zeros(len(g_circuit.parameters))
                    parameter_values_plus[g_param_idx] += np.pi/2
                    base_parameter_values.append(parameter_values_plus)
                    base_coeffs.append(coeff_map[g_param]/2)
                    # for - pi/2 in the parameter shift rule
                    parameter_values_minus = np.zeros(len(g_circuit.parameters))
                    parameter_values_minus[g_param_idx] -= np.pi/2
                    base_parameter_values.append(parameter_values_minus)
                    #base_coeffs.append(-1*coeff_map[g_param]/2)
            self._g_circuit_dict[circuit_idx]['base_parameter_values'] = base_parameter_values
            self._g_circuit_dict[circuit_idx]['base_coeffs'] = base_coeffs

    def __call__(
        self,
        circuits: Sequence[int | QuantumCircuit],
        observables: Sequence[int | SparsePauliOp],
        parameter_values: Sequence[float],
        partial: Sequence[Parameter] | None = None,
        **run_options,
    ) -> EstimatorGradientResult:

        circuit_idx = circuits[0]
        g_circuit = self._g_circuit_dict[circuit_idx]['g_circuit']
        parameter_map = self._g_circuit_dict[circuit_idx]['parameter_map']
        g_param_index_map = self._g_circuit_dict[circuit_idx]['g_param_index_map']
        base_coeffs = self._g_circuit_dict[circuit_idx]['base_coeffs']

        partial_parameters = partial or self._circuits[circuit_idx].parameters

        param_map = {}
        base_parameter_values_list = []

        g_parameter_values = np.zeros(len(g_circuit.parameters))
        result_index = 0
        result_index_map = {}
        for i, param in enumerate(self._circuits[circuit_idx].parameters):
            param_map[param] = parameter_values[i]
            for g_param in parameter_map[param]:
                g_param_idx = g_param_index_map[g_param]
                g_parameter_values[g_param_idx] = parameter_values[i]
                base_parameter_values_list.append(self._g_circuit_dict[circuit_idx]['base_parameter_values'][g_param_idx * 2])
                base_parameter_values_list.append(self._g_circuit_dict[circuit_idx]['base_parameter_values'][g_param_idx * 2 + 1])
                result_index_map[g_param] = result_index
                result_index += 1
        print('----------------------------------------------------------------')
        g_parameter_values_list = [g_parameter_values + base_parameter_values for base_parameter_values in base_parameter_values_list]
        print(g_parameter_values_list)
        print(len(g_circuit.parameters))
        for i in g_circuit.parameters:
            print(i)
        observable_indices = [observable_index for observable_index in observables for _ in range(len(g_parameter_values_list))]
        print('observable indices', observable_indices)


        results = self._estimator.__call__([circuit_idx]* len(g_parameter_values_list)*len(observables),
                                        observable_indices, g_parameter_values_list*len(observables))
        print(results)

        # Combines the results and coefficients to reconstruct the gradient for the original circuit parameters
        param_set = set(partial_parameters)
        values = np.zeros(len(self._circuits[circuit_idx].parameters))
        metadata = [{} for _ in range(len(partial_parameters))]


        for i in range(len(observables)):
            for j, param in enumerate(self._circuits[circuit_idx].parameters):
                if param not in param_set:
                    continue
                for g_param in parameter_map[param]:
                    g_param_idx = g_param_index_map[g_param]
                    coeff = base_coeffs[g_param_idx]

                    if isinstance(coeff, ParameterExpression):
                        local_map = {p: param_map[p] for p in coeff.parameters}
                        bound_coeff = float(coeff.bind(local_map))
                    else:
                        bound_coeff = coeff
                    # plus

                    print(j*len(self._circuits[circuit_idx].parameters))
                    print(result_index_map[g_param] * 2)
                    print(result_index_map[g_param] * 2 + 1)
                    values[j] += bound_coeff * results.values[j*len(self._circuits[circuit_idx].parameters) +result_index_map[g_param] * 2]
                    values[j] -= bound_coeff * results.values[j*len(self._circuits[circuit_idx].parameters) +result_index_map[g_param] * 2 + 1]

        return EstimatorGradientResult(
            values=[values], metadata=metadata
        )