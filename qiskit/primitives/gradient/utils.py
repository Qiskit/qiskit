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
Utility functions for gradients
"""

from __future__ import annotations

from copy import copy, deepcopy
from collections import Iterable, Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict

from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit


import numpy as np


# @dataclass(frozen=True)
# class EstimatorGradientResult:
#     """Result of EstimatorGradient.

#     Args:
#         values (np.ndarray): The list of the gradients.
#         metadata (list[dict]): List of the metadata.
#     """

#     values: "list[np.ndarray[Any, np.dtype[np.float64]]]"
#     metadata: list[dict[str, Any]]


@dataclass
class GradientCircuitData:
    circuit: QuantumCircuit
    gradient_circuit: QuantumCircuit
    gradient_parameter_map: Dict[Parameter, Parameter]
    coeff_map: Dict[Parameter, float | ParameterExpression]
    gradient_parameter_index_map: Dict[Parameter, int]

def rebuild_circuit_with_unique_parameters(circuit: QuantumCircuit):
    g_circuit = circuit.copy_empty_like(f'g_{circuit.name}')
    param_inst_dict = defaultdict(list)
    g_parameter_map = defaultdict(list)
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
                    g_parameter_map[parameter_variable].append(new_parameter_variable)
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
                new_parameter_variable = g_parameter_map[parameter_variable][0]
            else:
                new_parameter_variable = Parameter(f'g{parameter_variable.name}_1')
            subs_map[parameter_variable] = new_parameter_variable
        g_circuit.global_phase = g_circuit.global_phase.subs(subs_map)
    print(g_circuit.draw())
    print(g_circuit.num_parameters)
    g_parameter_index_map = {}
    for i, g_param in enumerate(g_circuit.parameters):
        g_parameter_index_map[g_param] = i
    return GradientCircuitData(circuit=circuit, gradient_circuit=g_circuit, gradient_parameter_map=g_parameter_map, coeff_map=coeff_map,gradient_parameter_index_map=g_parameter_index_map)
    #self._g_circuit_dict[circuit_idx] = {'g_circuit': g_circuit,'parameter_map': parameter_map, 'coeff_map': coeff_map}

def make_base_parameter_values_parameter_shift(gradient_circuit_data: GradientCircuitData):
    # Make internal parameter values for the parameter shift
    parameter_map = gradient_circuit_data.gradient_parameter_map
    gradient_circuit = gradient_circuit_data.gradient_circuit
    coeff_map =  gradient_circuit_data.coeff_map
    #g_param_index_map = {}

    # for i, g_param in enumerate(g_circuit.parameters):
    #     g_param_index_map[g_param] = i
    # self._g_circuit_dict[circuit_idx]['g_param_index_map'] = g_param_index_map

    base_parameter_values = []
    #base_coeffs = []

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
    num_g_parameters = len(gradient_circuit_data.gradient_circuit.parameters)
    for i, param in enumerate(gradient_circuit_data.circuit.parameters):
        for g_param in gradient_circuit_data.gradient_parameter_map[param]:
            # prepare base decomposed parameter values for each original parameter
            g_param_idx = gradient_circuit_data.gradient_parameter_index_map[g_param]
            # for + pi/2 in the parameter shift rule
            parameter_values_plus = np.zeros(num_g_parameters)
            parameter_values_plus[g_param_idx] += np.pi/2
            base_parameter_values.append(parameter_values_plus)
            #base_coeffs.append(coeff_map[g_param]/2)
            # for - pi/2 in the parameter shift rule
            parameter_values_minus = np.zeros(num_g_parameters)
            parameter_values_minus[g_param_idx] -= np.pi/2
            base_parameter_values.append(parameter_values_minus)
            #base_coeffs.append(-1*coeff_map[g_param]/2)
    #self._g_circuit_dict[circuit_idx]['base_parameter_values'] = base_parameter_values
    #self._g_circuit_dict[circuit_idx]['base_coeffs'] = base_coeffs
    # coeff_mapの値はわざわざリストにしなくてもgradientの関数の中で最後にかけるだけでいいか。
    return base_parameter_values