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
from hashlib import new
from typing import TYPE_CHECKING, Any, Dict

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

import numpy as np

from qiskit.converters import isinstanceint


@dataclass
class ParameterShiftGradientCircuitData:
    circuit: QuantumCircuit
    gradient_circuit: QuantumCircuit
    gradient_parameter_map: Dict[Parameter, Parameter]
    gradient_parameter_index_map: Dict[Parameter, int]
    gradient_virtual_parameter_map: Dict[Parameter, Parameter]
    coeff_map: Dict[Parameter, float | ParameterExpression]

def make_gradient_circuit_param_shift(circuit: QuantumCircuit):
    SUPPORTED_GATES = ["x", "y", "z", "h", "rx", "ry", "rz", "p", "cx", "cy", "cz"]

    circuit2 = transpile(circuit, basis_gates=SUPPORTED_GATES, optimization_level=0)
    g_circuit = circuit2.copy_empty_like(f'g_{circuit2.name}')
    param_inst_dict = defaultdict(list)
    g_parameter_map = defaultdict(list)
    g_virtual_parameter_map = {}
    num_virtual_parameter_variables = 0
    coeff_map = {}

    for inst in circuit2.data:
        new_inst = deepcopy(inst)
        qubit_indices = [circuit2.qubits.index(qubit) for qubit in inst[1]]
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

                # If new_parameter is not a single parameter variable, then add a new virtual parameter variable.
                # e.g. ry(θ) with θ = (2x + y) becomes ry(θ + virtual_variable)
                if not isinstance(new_parameter, Parameter):
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
    return ParameterShiftGradientCircuitData(circuit=circuit2, gradient_circuit=g_circuit,gradient_virtual_parameter_map=g_virtual_parameter_map,
        gradient_parameter_map=g_parameter_map, coeff_map=coeff_map,gradient_parameter_index_map=g_parameter_index_map)

def make_base_parameter_values_parameter_shift(gradient_circuit_data: ParameterShiftGradientCircuitData):
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

def make_base_parameter_values_fin_diff(circuit: QuantumCircuit, epsilon):

    base_parameter_values = []

    # Make base decomposed parameter values for each original parameter
    for i, _ in enumerate(circuit.parameters):
        parameter_values_plus = np.zeros(len(circuit.parameters))
        parameter_values_plus[i] += epsilon
        base_parameter_values.append(parameter_values_plus)
        # for - epsilon in the finite diff
        parameter_values_minus = np.zeros(len(circuit.parameters))
        parameter_values_minus[i] -= epsilon
        base_parameter_values.append(parameter_values_minus)

    for i in base_parameter_values:
        print(i)
    return base_parameter_values

@dataclass
class LinearCombGradientCircuit:
    gradient_circuit: QuantumCircuit
    coeff: float | ParameterExpression
    index: int = 0

def make_gradient_circuit_lin_comb(circuit: QuantumCircuit, add_measurement: bool= False):
    SUPPORTED_GATES = [
        "rx",
        "ry",
        "rz",
        "rzx",
        "rzz",
        "ryy",
        "rxx",
        "cx",
        "cy",
        "cz",
        "ccx",
        "swap",
        "iswap",
        "h",
        "t",
        "s",
        "sdg",
        "x",
        "y",
        "z",
    ]
    circuit2 = transpile(circuit, basis_gates=SUPPORTED_GATES, optimization_level=0)

    qr_aux = QuantumRegister(1, "aux")
    cr_aux = ClassicalRegister(1, "aux")
    circuit2.add_register(qr_aux)
    circuit2.add_bits(cr_aux)
    circuit2.h(qr_aux)
    circuit2.data.insert(0, circuit2.data.pop())
    circuit2.sdg(qr_aux)
    circuit2.data.insert(1, circuit2.data.pop())

    grad_dict = defaultdict(list)
    for i, (inst, qregs, _) in enumerate(circuit2.data):
        if inst.is_parameterized():
            param = inst.params[0]
            for p in param.parameters:
                gate = _gate_gradient(inst)
                circuit3 = circuit2.copy()
                # insert `gate` to i-th position
                circuit3.append(gate, [qr_aux[0]] + qregs, [])
                circuit3.data.insert(i, circuit3.data.pop())
                #
                circuit3.h(qr_aux)
                if add_measurement:
                    circuit3.measure(qr_aux, cr_aux)
                grad_dict[p].append(LinearCombGradientCircuit(circuit3, param.gradient(p)))
                print(circuit3)

    return grad_dict

def _gate_gradient(gate: Gate) -> Instruction:
    if isinstance(gate, RXGate):
        # theta
        return CXGate()
    if isinstance(gate, RYGate):
        # theta
        return CYGate()
    if isinstance(gate, RZGate):
        # theta
        return CZGate()
    if isinstance(gate, RXXGate):
        # theta
        cxx_circ = QuantumCircuit(3)
        cxx_circ.cx(0, 1)
        cxx_circ.cx(0, 2)
        cxx = cxx_circ.to_instruction()
        return cxx
    if isinstance(gate, RYYGate):
        # theta
        cyy_circ = QuantumCircuit(3)
        cyy_circ.cy(0, 1)
        cyy_circ.cy(0, 2)
        cyy = cyy_circ.to_instruction()
        return cyy
    if isinstance(gate, RZZGate):
        # theta
        czz_circ = QuantumCircuit(3)
        czz_circ.cz(0, 1)
        czz_circ.cz(0, 2)
        czz = czz_circ.to_instruction()
        return czz
    if isinstance(gate, RZXGate):
        # theta
        czx_circ = QuantumCircuit(3)
        czx_circ.cx(0, 2)
        czx_circ.cz(0, 1)
        czx = czx_circ.to_instruction()
        return czx
    raise TypeError(f"Unrecognized parameterized gate, {gate}")
