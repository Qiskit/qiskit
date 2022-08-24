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

# pylint: disable=invalid-name

"""
Utility functions for gradients
"""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import random
from typing import Dict, List

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


@dataclass
class ParameterShiftGradientCircuitData:
    """Stores gradient circuit data for the parameter shift method

    Args:
        circuit (QuantumCircuit): The original quantum circuit
        gradient_circuit (QuantumCircuit): An internal quantum circuit used to calculate the gradient
        gradient_parameter_map (dict): A dictionary maps the parameters of ``circuit`` to
            the parameters of ``gradient_circuit``.
        gradient_parameter_index_map (dict): A dictionary maps the parameters of ``gradient_circuit``
            to their index.
        gradient_virtual_parameter_map (dict): A dictionary maps the parameters of ``gradient_circuit``
            to the virtual parameter variables. A virtual parameter variable is added if a parameter
            expression has more than one parameter.
        coeff_map (dict): A dictionary maps the parameters of ``gradient_circuit`` to their coefficients
            used to calculate gradients.
    """

    circuit: QuantumCircuit
    gradient_circuit: QuantumCircuit
    gradient_parameter_map: Dict[Parameter, Parameter]
    gradient_parameter_index_map: Dict[Parameter, int]
    gradient_virtual_parameter_map: Dict[Parameter, Parameter]
    coeff_map: Dict[Parameter, float | ParameterExpression]


def make_param_shift_gradient_circuit_data(
    circuit: QuantumCircuit,
) -> ParameterShiftGradientCircuitData:
    """Makes a gradient circuit data for the parameter shift method. This re-assigns each parameter in
        ``circuit`` to a unique parameter, and construct a new gradient circuit with those new
        parameters. Also, it makes maps used in later calculations.

    Args:
        circuit: The original quantum circuit

    Returns:
        necessary data to calculate gradients with the parameter shift method.
    """

    supported_gates = [
        "x",
        "y",
        "z",
        "h",
        "rx",
        "ry",
        "rz",
        "p",
        "cx",
        "cy",
        "cz",
        "ryy",
        "rxx",
        "rzz",
    ]

    circuit2 = transpile(circuit, basis_gates=supported_gates, optimization_level=0)
    g_circuit = circuit2.copy_empty_like(f"g_{circuit2.name}")
    param_inst_dict = defaultdict(list)
    g_parameter_map = defaultdict(list)
    g_virtual_parameter_map = {}
    num_virtual_parameter_variables = 0
    coeff_map = {}

    for inst in circuit2.data:
        new_inst = deepcopy(inst)
        qubit_indices = [circuit2.qubits.index(qubit) for qubit in inst[1]]
        new_inst.qubits = tuple(g_circuit.qubits[qubit_index] for qubit_index in qubit_indices)

        # Assign new unique parameters when the instruction is parameterized.
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
                        new_parameter_variable = Parameter(
                            f"g{parameter_variable.name}_{len(param_inst_dict[parameter_variable])+1}"
                        )
                    else:
                        new_parameter_variable = Parameter(f"g{parameter_variable.name}_1")
                    subs_map[parameter_variable] = new_parameter_variable
                    param_inst_dict[parameter_variable].append(inst)
                    g_parameter_map[parameter_variable].append(new_parameter_variable)
                    # Coefficient to calculate derivative i.e. dw/dt in df/dw * dw/dt
                    coeff_map[new_parameter_variable] = parameter.gradient(parameter_variable)
                # Substitute the parameter variables with the corresponding new parameter
                # variables in ``subs_map``.
                new_parameter = parameter.subs(subs_map)
                # If new_parameter is not a single parameter variable, then add a new virtual
                # parameter variable. e.g. ry(θ) with θ = (2x + y) becomes ry(θ + virtual_variable)
                if not isinstance(new_parameter, Parameter):
                    virtual_parameter_variable = Parameter(
                        f"vθ_{num_virtual_parameter_variables+1}"
                    )
                    num_virtual_parameter_variables += 1
                    for new_parameter_variable in new_parameter.parameters:
                        g_virtual_parameter_map[new_parameter_variable] = virtual_parameter_variable
                    new_parameter = new_parameter + virtual_parameter_variable
                new_inst_parameters.append(new_parameter)
            new_inst.operation.params = new_inst_parameters
        g_circuit.append(new_inst)

    # for global phase
    subs_map = {}
    if isinstance(g_circuit.global_phase, ParameterExpression):
        for parameter_variable in g_circuit.global_phase.parameters:
            if parameter_variable in param_inst_dict:
                new_parameter_variable = g_parameter_map[parameter_variable][0]
            else:
                new_parameter_variable = Parameter(f"g{parameter_variable.name}_1")
            subs_map[parameter_variable] = new_parameter_variable
        g_circuit.global_phase = g_circuit.global_phase.subs(subs_map)
    g_parameter_index_map = {}

    for i, g_param in enumerate(g_circuit.parameters):
        g_parameter_index_map[g_param] = i

    return ParameterShiftGradientCircuitData(
        circuit=circuit2,
        gradient_circuit=g_circuit,
        gradient_virtual_parameter_map=g_virtual_parameter_map,
        gradient_parameter_map=g_parameter_map,
        coeff_map=coeff_map,
        gradient_parameter_index_map=g_parameter_index_map,
    )


def make_param_shift_base_parameter_values(
    gradient_circuit_data: ParameterShiftGradientCircuitData,
) -> List[np.ndarray]:
    """Makes base parameter values for the parameter shift method. Each base parameter value will
        be added to the given parameter values in later calculations.

    Args:
        gradient_circuit_data: gradient circuit data for the base parameter values.

    Returns:
        The base parameter values for the parameter shift method.
    """
    # Make internal parameter values for the parameter shift
    num_g_parameters = len(gradient_circuit_data.gradient_circuit.parameters)
    base_parameter_values = []
    # Make base decomposed parameter values for each original parameter
    for param in gradient_circuit_data.circuit.parameters:
        for g_param in gradient_circuit_data.gradient_parameter_map[param]:
            # use the related virtual parameter if it exists
            if g_param in gradient_circuit_data.gradient_virtual_parameter_map:
                g_param = gradient_circuit_data.gradient_virtual_parameter_map[g_param]
            g_param_idx = gradient_circuit_data.gradient_parameter_index_map[g_param]
            # for + pi/2 in the parameter shift rule
            parameter_values_plus = np.zeros(num_g_parameters)
            parameter_values_plus[g_param_idx] += np.pi / 2
            base_parameter_values.append(parameter_values_plus)
            # for - pi/2 in the parameter shift rule
            parameter_values_minus = np.zeros(num_g_parameters)
            parameter_values_minus[g_param_idx] -= np.pi / 2
            base_parameter_values.append(parameter_values_minus)
    return base_parameter_values


def make_fin_diff_base_parameter_values(
    circuit: QuantumCircuit, epsilon: float = 1e-6
) -> List[np.ndarray]:
    """Makes base parameter values for the finite difference method. Each base parameter value will
        be added to the given parameter values in later calculations.

    Args:
        circuit: circuit for the base parameter values.
        epsilon: The offset size for the finite difference gradients.

    Returns:
        List: The base parameter values for the finite difference method.
    """
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
    return base_parameter_values


@dataclass
class LinearCombGradientCircuit:
    """Gradient circuit for the linear combination of unitaries method.

    Args:
        gradient_circuit (QuantumCircuit): A gradient circuit  for the linear combination
            of unitaries method.
        coeff (float | ParameterExpression): A coefficient corresponds to the gradient circuit.
    """

    gradient_circuit: QuantumCircuit
    coeff: float | ParameterExpression


def make_lin_comb_gradient_circuit(
    circuit: QuantumCircuit, add_measurement: bool = False
) -> Dict[Parameter, List[LinearCombGradientCircuit]]:
    """Makes gradient circuits for the linear combination of unitaries method.

    Args:
        circuit: The original quantum circuit.
        add_measurement: If True, add measurements to the gradient circuit. Defaults to False.
            ``LinCombSamplerGradient`` calls this method with `add_measurement` is True.

    Returns:
        A dictionary mapping a parameter to the corresponding list of ``LinearCombGradientCircuit``
    """
    supported_gates = [
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
    circuit2 = transpile(circuit, basis_gates=supported_gates, optimization_level=0)

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
                circuit3.h(qr_aux)
                if add_measurement:
                    circuit3.measure(qr_aux, cr_aux)
                grad_dict[p].append(LinearCombGradientCircuit(circuit3, param.gradient(p)))

    return grad_dict


def _gate_gradient(gate: Gate) -> Instruction:
    """Returns the derivative of the gate"""
    # pylint: disable=too-many-return-statements
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


def make_spsa_base_parameter_values(
    circuit: QuantumCircuit, epsilon: float = 1e-6
) -> List[np.ndarray]:
    """Makes base parameter values for the SPSA. Each base parameter value will
        be added to the given parameter values in later calculations.

    Args:
        circuit: circuit for the base parameter values.
        epsilon: The offset size for the finite difference gradients.

    Returns:
        List: The base parameter values for the SPSA.
    """

    base_parameter_values = []
    # Make a perturbation vector
    parameter_values_plus = np.array(
        [(-1) ** (random.randint(0, 1)) for _ in range(len(circuit.parameters))]
    )
    parameter_values_plus = epsilon * parameter_values_plus
    parameter_values_minus = -parameter_values_plus
    base_parameter_values.append(parameter_values_plus)
    base_parameter_values.append(parameter_values_minus)
    return base_parameter_values
