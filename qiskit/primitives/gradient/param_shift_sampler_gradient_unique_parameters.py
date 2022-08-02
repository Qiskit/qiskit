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

# @dataclass
# class SubSampler:
#     coeff: float | ParameterExpression
#     circuit: QuantumCircuit
#     index: int

#dataclass for g_circuit作って
#base_parameter_values_listとかcoeff_listとか全部入れた方がいいかも

class ParamShiftSamplerGradientUniqueParameters:
    """Parameter shift estimator gradient"""

    SUPPORTED_GATES = ["x", "y", "z", "h", "rx", "ry", "rz", "p", "cx", "cy", "cz"]

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



    # def _rebuild_circuits_with_unique_parameters(self):

    #     for circuit_idx, circuit in enumerate(self._circuits):
    #         g_circuit = circuit.copy_empty_like(f'g_{circuit.name}')
    #         param_inst_dict = defaultdict(list)
    #         parameter_map = defaultdict(list)
    #         coeff_map = {}

    #         for inst in circuit.data:
    #             new_inst = deepcopy(inst)
    #             qubit_idx = circuit.qubits.index(inst[1][0])
    #             new_inst.qubits = (g_circuit.qubits[qubit_idx],)

    #             # Assign new unique parameters when the instruction is a parameterized.
    #             if inst.operation.is_parameterized():
    #                 parameters = inst.operation.params
    #                 new_inst_parameters = []
    #                 # For a gate with multiple parameters e.g. a U gate
    #                 for parameter in parameters:
    #                     subs_map = {}
    #                     # For a gate parameter with multiple parameter variables.
    #                     # e.g. ry(θ) with θ = (2x + y)
    #                     for parameter_variable in parameter.parameters:
    #                         if parameter_variable in param_inst_dict:
    #                             new_parameter_variable = Parameter(f'g{parameter_variable.name}_{len(param_inst_dict[parameter_variable])+1}')
    #                         else:
    #                             new_parameter_variable = Parameter(f'g{parameter_variable.name}_1')
    #                         subs_map[parameter_variable] = new_parameter_variable
    #                         param_inst_dict[parameter_variable].append(inst)
    #                         parameter_map[parameter_variable].append(new_parameter_variable)
    #                         # Coefficient to calculate derivative i.e. dt/dw in df/dt * dt/dw
    #                         coeff_map[new_parameter_variable] = parameter.gradient(parameter_variable)
    #                     # Substitute the parameter variables with the corresponding new parameter variables in ``subs_map``.
    #                     new_parameter = parameter.subs(subs_map)
    #                     new_inst_parameters.append(new_parameter)
    #                     print(new_inst_parameters)
    #                 new_inst.operation.params = new_inst_parameters
    #             g_circuit.append(new_inst)
    #             print(new_inst)
    #         # for global phase
    #         subs_map = {}
    #         if isinstance(g_circuit.global_phase, ParameterExpression):
    #             for parameter_variable in g_circuit.global_phase.parameters:
    #                 if parameter_variable in param_inst_dict:
    #                     new_parameter_variable = parameter_map[parameter_variable][0]
    #                 else:
    #                     new_parameter_variable = Parameter(f'g{parameter_variable.name}_1')
    #                 subs_map[parameter_variable] = new_parameter_variable
    #             g_circuit.global_phase = g_circuit.global_phase.subs(subs_map)
    #         print(g_circuit.draw())
    #         print(g_circuit.num_parameters)
    #         self._g_circuit_dict[circuit_idx] = {'g_circuit': g_circuit,'parameter_map': parameter_map, 'coeff_map': coeff_map}


    # def _prepare_base_parameter_values(self):
    #     # construct internal parameter values for the parameter shift
    #     for circuit_idx, circuit in enumerate(self._circuits):
    #         parameter_map = self._g_circuit_dict[circuit_idx]['parameter_map']
    #         g_circuit = self._g_circuit_dict[circuit_idx]['g_circuit']
    #         coeff_map =  self._g_circuit_dict[circuit_idx]['coeff_map']
    #         g_param_index_map = {}

    #         for i, g_param in enumerate(g_circuit.parameters):
    #             g_param_index_map[g_param] = i
    #         self._g_circuit_dict[circuit_idx]['g_param_index_map'] = g_param_index_map

    #         base_parameter_values = []
    #         base_coeffs = []

    #         # for i, param in enumerate(circuit.parameters):
    #         #     sub_parameter_values = []
    #         #     coeffs = []
    #         #     for g_param in parameter_map[param]:
    #         #         # prepare base decomposed parameter values for each original parameter
    #         #         g_param_idx = g_param_index_map[g_param]
    #         #         # for + pi/2 in the parameter shift rule
    #         #         parameter_values_plus = np.zeros(len(g_circuit.parameters))
    #         #         parameter_values_plus[g_param_idx] += np.pi/2
    #         #         # for - pi/2 in the parameter shift rule
    #         #         parameter_values_minus = np.zeros(len(g_circuit.parameters))
    #         #         parameter_values_minus[g_param_idx] -= np.pi/2
    #         #         sub_parameter_values.append((parameter_values_plus, parameter_values_minus))
    #         #         coeffs.append((coeff_map[g_param]/2, -1*coeff_map[g_param]/2))
    #         #     base_parameter_values.append(sub_parameter_values)
    #         #     base_coeffs.append(coeffs)
    #         for i, param in enumerate(circuit.parameters):
    #             for g_param in parameter_map[param]:
    #                 # prepare base decomposed parameter values for each original parameter
    #                 g_param_idx = g_param_index_map[g_param]
    #                 # for + pi/2 in the parameter shift rule
    #                 parameter_values_plus = np.zeros(len(g_circuit.parameters))
    #                 parameter_values_plus[g_param_idx] += np.pi/2
    #                 base_parameter_values.append(parameter_values_plus)
    #                 base_coeffs.append(coeff_map[g_param]/2)
    #                 # for - pi/2 in the parameter shift rule
    #                 parameter_values_minus = np.zeros(len(g_circuit.parameters))
    #                 parameter_values_minus[g_param_idx] -= np.pi/2
    #                 base_parameter_values.append(parameter_values_minus)
    #                 #base_coeffs.append(-1*coeff_map[g_param]/2)
    #         self._g_circuit_dict[circuit_idx]['base_parameter_values'] = base_parameter_values
    #         self._g_circuit_dict[circuit_idx]['base_coeffs'] = base_coeffs

    def __call__(
        self,
        circuits: Sequence[int | QuantumCircuit],
        parameter_values: Sequence[Sequence[float]],
        partial: Sequence[Sequence[Parameter]] | None = None,
        **run_options,
    ) -> SamplerResult:

        partial = partial or [[] for _ in range(len(circuits))]
        # print(partial)
        # print(circuits)
        # print(len(circuits))
        gradients = []
        for circuit_index, parameter_values_, partial_ in zip(circuits, parameter_values, partial):
            print(circuit_index, parameter_values_)
            gradient_circuit_data = self._gradient_circuit_data_dict[circuit_index]

            #gradient_circuit = gradient_circuit_data.gradient_circuit


            gradient_parameter_map = gradient_circuit_data.gradient_parameter_map
            gradient_parameter_index_map = gradient_circuit_data.gradient_parameter_index_map
            #base_coeffs = self._g_circuit_dict[circuit_index]['base_coeffs']

            parameters = partial_ or self._circuits[circuit_index].parameters
            param_set = set(parameters)

            parameter_value_map = {}
            base_parameter_values_list = []

            circuit_parameters = self._circuits[circuit_index].parameters


            gradient_parameter_values = np.zeros(len(gradient_circuit_data.gradient_circuit.parameters))
            #g_parameter_values = np.zeros(len(gradient_circuit.parameters))
            result_index = 0
            result_index_map = {}
            for i, param in enumerate(circuit_parameters):
                parameter_value_map[param] = parameter_values_[i]
                for g_param in gradient_parameter_map[param]:
                    g_param_idx = gradient_parameter_index_map[g_param]
                    gradient_parameter_values[g_param_idx] = parameter_values_[i]
                    if param in param_set:
                        base_parameter_values_list.append(self._base_parameter_values_dict[circuit_index][g_param_idx * 2])
                        base_parameter_values_list.append(self._base_parameter_values_dict[circuit_index][g_param_idx * 2 + 1])
                        # base_parameter_values_list.append(self._g_circuit_dict[circuit_index]['base_parameter_values'][g_param_idx * 2])
                        # base_parameter_values_list.append(self._g_circuit_dict[circuit_index]['base_parameter_values'][g_param_idx * 2 + 1])
                        result_index_map[g_param] = result_index
                        result_index += 1
            #print('----------------------------------------------------------------')
            gradient_parameter_values_list = [gradient_parameter_values + base_parameter_values for base_parameter_values in base_parameter_values_list]
            #print(g_parameter_values_list)
            # print(len(gradient_circuit.parameters))
            # for i in gradient_circuit.parameters:
            #     print(i)
            circuit_indices = [circuit_index] * len(gradient_parameter_values_list)
            results = self._sampler.__call__(circuit_indices, gradient_parameter_values_list)
            #print(results)

            # Combines the results and coefficients to reconstruct the gradient for the original circuit parameters
            #param_set = set(parameters)
            dists = [Counter() for _ in range(len(parameter_values_))]

            print(dists)

            for i, param in enumerate(circuit_parameters):
                if param not in param_set:
                    continue
                for g_param in gradient_parameter_map[param]:
                    g_param_idx = gradient_parameter_index_map[g_param]
                    # TODO: base_coeffs やめて coeff/2するようにする
                    coeff = gradient_circuit_data.coeff_map[g_param]/2

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

        # return SamplerResult(
        #     quasi_dists=[QuasiDistribution(dist) for dist in dists], metadata=metadata
        # )