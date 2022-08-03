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

from qiskit import transpile
from qiskit.circuit import Parameter, ParameterExpression, QuantumCircuit


import numpy as np


@dataclass
class GradientCircuitData:
    circuit: QuantumCircuit
    gradient_circuit: QuantumCircuit
    gradient_parameter_map: Dict[Parameter, Parameter]
    gradient_parameter_index_map: Dict[Parameter, int]
    gradient_virtual_parameter_map: Dict[Parameter, Parameter]
    coeff_map: Dict[Parameter, float | ParameterExpression]

def rebuild_circuit_with_unique_parameters(circuit: QuantumCircuit):
    SUPPORTED_GATES = ["x", "y", "z", "h", "rx", "ry", "rz", "p", "cx", "cy", "cz"]

    circuit = transpile(circuit, basis_gates=SUPPORTED_GATES, optimization_level=0)
    g_circuit = circuit.copy_empty_like(f'g_{circuit.name}')
    param_inst_dict = defaultdict(list)
    g_parameter_map = defaultdict(list)
    g_virtual_parameter_map = {}
    num_virtual_parameter_variables = 0
    coeff_map = {}

    for inst in circuit.data:
        new_inst = deepcopy(inst)
        qubit_indices = [circuit.qubits.index(qubit) for qubit in inst[1]]
        new_inst.qubits = tuple(g_circuit.qubits[qubit_index] for qubit_index in qubit_indices)

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
                    # Coefficient to calculate derivative i.e. dw/dt in df/dw * dw/dt
                    coeff_map[new_parameter_variable] = parameter.gradient(parameter_variable)
                    #
                # Substitute the parameter variables with the corresponding new parameter variables in ``subs_map``.
                new_parameter = parameter.subs(subs_map)

                # If the number of parameter variables is more than one, add a new virtual parameter variable.
                # This virtual parameter variable is used to calculate df/dw when f has multiple parameter variables.
                # e.g. ry(θ) with θ = (2x + y) becomes ry(θ+v)
                #print('parameters!!!!!!!', new_parameter.parameters)
                if 1 < len(new_parameter.parameters):
                    virtual_parameter_variable = Parameter(f'vθ_{num_virtual_parameter_variables+1}')
                    num_virtual_parameter_variables += 1
                    for new_parameter_variable in new_parameter.parameters:
                        g_virtual_parameter_map[new_parameter_variable] = virtual_parameter_variable
                    new_parameter = new_parameter + virtual_parameter_variable
                #print('new_parameter: ', new_parameter)
                new_inst_parameters.append(new_parameter)
                #print(new_inst_parameters)
            new_inst.operation.params = new_inst_parameters
        #print(inst)
        #print(new_inst)
        g_circuit.append(new_inst)

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
    return GradientCircuitData(circuit=circuit, gradient_circuit=g_circuit,gradient_virtual_parameter_map=g_virtual_parameter_map,
        gradient_parameter_map=g_parameter_map, coeff_map=coeff_map,gradient_parameter_index_map=g_parameter_index_map)

def make_base_parameter_values_parameter_shift(gradient_circuit_data: GradientCircuitData):
    # Make internal parameter values for the parameter shift
    parameter_map = gradient_circuit_data.gradient_parameter_map
    gradient_circuit = gradient_circuit_data.gradient_circuit

    num_g_parameters = len(gradient_circuit_data.gradient_circuit.parameters)
    base_parameter_values = []

    # Make base decomposed parameter values for each original parameter
    for i, param in enumerate(gradient_circuit_data.circuit.parameters):
        for g_param in gradient_circuit_data.gradient_parameter_map[param]:
            # use the related virtual parameter if it exists
            if g_param in gradient_circuit_data.gradient_virtual_parameter_map:
                g_param = gradient_circuit_data.gradient_virtual_parameter_map[g_param]
                print(gradient_circuit_data.gradient_parameter_index_map[g_param])

            g_param_idx = gradient_circuit_data.gradient_parameter_index_map[g_param]
            # for + pi/2 in the parameter shift rule
            parameter_values_plus = np.zeros(num_g_parameters)
            parameter_values_plus[g_param_idx] += np.pi/2
            base_parameter_values.append(parameter_values_plus)
            # for - pi/2 in the parameter shift rule
            parameter_values_minus = np.zeros(num_g_parameters)
            parameter_values_minus[g_param_idx] -= np.pi/2
            base_parameter_values.append(parameter_values_minus)

    for i in base_parameter_values:
        print(i)
    return base_parameter_values