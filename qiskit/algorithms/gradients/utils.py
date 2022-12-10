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
    XGate,
)


@dataclass
class ParameterShiftGradientCircuit:
    """Stores gradient circuit data for the parameter shift method"""

    circuit: QuantumCircuit
    """The original quantum circuit"""
    gradient_circuit: QuantumCircuit
    """An internal quantum circuit used to calculate the gradient"""
    gradient_parameter_map: dict[Parameter, Parameter]
    """A dictionary maps the parameters of ``circuit`` to the parameters of ``gradient_circuit``"""
    gradient_virtual_parameter_map: dict[Parameter, Parameter]
    """A dictionary maps the parameters of ``gradient_circuit`` to the virtual parameter variables"""
    coeff_map: dict[Parameter, float | ParameterExpression]
    """A dictionary maps the parameters of ``gradient_circuit`` to their coefficients"""


def _make_param_shift_gradient_circuit_data(
    circuit: QuantumCircuit,
) -> ParameterShiftGradientCircuit:
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
        "rzx",
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

    return ParameterShiftGradientCircuit(
        circuit=circuit2,
        gradient_circuit=g_circuit,
        gradient_virtual_parameter_map=g_virtual_parameter_map,
        gradient_parameter_map=g_parameter_map,
        coeff_map=coeff_map,
    )


def _make_param_shift_base_parameter_values(
    gradient_circuit_data: ParameterShiftGradientCircuit,
) -> list[np.ndarray]:
    """Makes base parameter values for the parameter shift method. Each base parameter value will
        be added to the given parameter values in later calculations.

    Args:
        gradient_circuit_data: gradient circuit data for the base parameter values.

    Returns:
        The base parameter values for the parameter shift method.
    """
    # Make internal parameter values for the parameter shift
    g_parameters = gradient_circuit_data.gradient_circuit.parameters
    plus_offsets = []
    minus_offsets = []
    # Make base decomposed parameter values for each original parameter
    for g_param in g_parameters:
        if g_param in gradient_circuit_data.gradient_virtual_parameter_map:
            g_param = gradient_circuit_data.gradient_virtual_parameter_map[g_param]
        idx = g_parameters.data.index(g_param)
        plus = np.zeros(len(g_parameters))
        plus[idx] += np.pi / 2
        minus = np.zeros(len(g_parameters))
        minus[idx] -= np.pi / 2
        plus_offsets.append(plus)
        minus_offsets.append(minus)
    return plus_offsets + minus_offsets


def _param_shift_preprocessing(circuit: QuantumCircuit) -> ParameterShiftGradientCircuit:
    """Preprocessing for the parameter shift method.

    Args:
        circuit: The original quantum circuit

    Returns:
        necessary data to calculate gradients with the parameter shift method.
    """
    gradient_circuit_data = _make_param_shift_gradient_circuit_data(circuit)
    base_parameter_values = _make_param_shift_base_parameter_values(gradient_circuit_data)

    return gradient_circuit_data, base_parameter_values


def _make_param_shift_parameter_values(
    gradient_circuit_data: ParameterShiftGradientCircuit,
    base_parameter_values: list[np.ndarray],
    parameter_values: np.ndarray,
    param_set: set[Parameter],
) -> list[np.ndarray]:
    """Makes parameter values for the parameter shift method. Each parameter value will be added to
        the base parameter values in later calculations.

    Args:
        gradient_circuit_data: gradient circuit data for the parameter shift method.
        base_parameter_values: base parameter values for the parameter shift method.
        parameter_values: parameter values to be added to the base parameter values.
        param_set: set of parameters to be used in the parameter shift method.

    Returns:
        The parameter values for the parameter shift method.
    """
    circuit = gradient_circuit_data.circuit
    gradient_circuit = gradient_circuit_data.gradient_circuit
    gradient_parameter_values = np.zeros(len(gradient_circuit_data.gradient_circuit.parameters))
    plus_offsets, minus_offsets, result_indices, coeffs = [], [], [], []
    result_idx = 0
    for i, param in enumerate(circuit.parameters):
        g_params = gradient_circuit_data.gradient_parameter_map[param]
        indices = [gradient_circuit.parameters.data.index(g_param) for g_param in g_params]
        gradient_parameter_values[indices] = parameter_values[i]
        if param in param_set:
            plus_offsets.extend(base_parameter_values[idx] for idx in indices)
            minus_offsets.extend(
                base_parameter_values[idx + len(gradient_circuit.parameters)] for idx in indices
            )
            result_indices.extend(result_idx for _ in range(len(indices)))
            result_idx += 1
            for g_param in g_params:
                coeff = gradient_circuit_data.coeff_map[g_param]
                # if coeff has parameters, we need to substitute
                if isinstance(coeff, ParameterExpression):
                    local_map = {
                        p: parameter_values[circuit.parameters.data.index(p)]
                        for p in coeff.parameters
                    }
                    bound_coeff = float(coeff.bind(local_map))
                else:
                    bound_coeff = coeff
                coeffs.append(bound_coeff / 2)

    # add the base parameter values to the parameter values
    gradient_parameter_values_plus = [
        gradient_parameter_values + plus_offset for plus_offset in plus_offsets
    ]
    gradient_parameter_values_minus = [
        gradient_parameter_values + minus_offset for minus_offset in minus_offsets
    ]
    return gradient_parameter_values_plus, gradient_parameter_values_minus, result_indices, coeffs


@dataclass
class LinearCombGradientCircuit:
    """Gradient circuit for the linear combination of unitaries method."""

    gradient_circuit: QuantumCircuit
    """A gradient circuit  for the linear combination of unitaries method."""
    coeff: float | ParameterExpression
    """A coefficient corresponds to the gradient circuit."""


def _make_lin_comb_gradient_circuit(
    circuit: QuantumCircuit,
    add_measurement: bool = False,
) -> dict[Parameter, list[LinearCombGradientCircuit]]:
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
    qr_aux = QuantumRegister(1, "qr_aux")
    cr_aux = ClassicalRegister(1, "cr_aux")
    circuit2.add_register(qr_aux)
    circuit2.add_register(cr_aux)
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
        return CXGate()
    if isinstance(gate, RYGate):
        return CYGate()
    if isinstance(gate, RZGate):
        return CZGate()
    if isinstance(gate, RXXGate):
        cxx_circ = QuantumCircuit(3)
        cxx_circ.cx(0, 1)
        cxx_circ.cx(0, 2)
        cxx = cxx_circ.to_instruction()
        return cxx
    if isinstance(gate, RYYGate):
        cyy_circ = QuantumCircuit(3)
        cyy_circ.cy(0, 1)
        cyy_circ.cy(0, 2)
        cyy = cyy_circ.to_instruction()
        return cyy
    if isinstance(gate, RZZGate):
        czz_circ = QuantumCircuit(3)
        czz_circ.cz(0, 1)
        czz_circ.cz(0, 2)
        czz = czz_circ.to_instruction()
        return czz
    if isinstance(gate, RZXGate):
        czx_circ = QuantumCircuit(3)
        czx_circ.cx(0, 2)
        czx_circ.cz(0, 1)
        czx = czx_circ.to_instruction()
        return czx
    raise TypeError(f"Unrecognized parameterized gate, {gate}")


def _make_lin_comb_qfi_circuit(
    circuit: QuantumCircuit, add_measurement: bool = False
) -> dict[Parameter, list[LinearCombGradientCircuit]]:
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

    grad_dict = defaultdict(list)
    for i, (inst_i, qregs_i, _) in enumerate(circuit2.data):
        if not inst_i.is_parameterized():
            continue
        for j, (inst_j, qregs_j, _) in enumerate(circuit2.data):
            if inst_j.is_parameterized():
                param_i = inst_i.params[0]
                param_j = inst_j.params[0]

                for p_i in param_i.parameters:
                    for p_j in param_j.parameters:
                        if circuit2.parameters.data.index(p_i) > circuit2.parameters.data.index(
                            p_j
                        ):
                            continue
                        gate_i = _gate_gradient(inst_i)
                        gate_j = _gate_gradient(inst_j)
                        circuit3 = circuit2.copy()
                        if i < j:
                            # insert gate_j to j-th position
                            circuit3.append(gate_j, [qr_aux[0]] + qregs_j, [])
                            circuit3.data.insert(j, circuit3.data.pop())
                            # insert gate_i to i-th position with two X gates at its sides
                            circuit3.append(XGate(), [qr_aux[0]], [])
                            circuit3.data.insert(i, circuit3.data.pop())
                            circuit3.append(gate_i, [qr_aux[0]] + qregs_i, [])
                            circuit3.data.insert(i, circuit3.data.pop())
                            circuit3.append(XGate(), [qr_aux[0]], [])
                            circuit3.data.insert(i, circuit3.data.pop())
                        else:
                            # insert gate_i to i-th position
                            circuit3.append(gate_i, [qr_aux[0]] + qregs_i, [])
                            circuit3.data.insert(i, circuit3.data.pop())
                            # insert gate_j to j-th position with two X gates at its sides
                            circuit3.append(XGate(), [qr_aux[0]], [])
                            circuit3.data.insert(j, circuit3.data.pop())
                            circuit3.append(gate_j, [qr_aux[0]] + qregs_j, [])
                            circuit3.data.insert(j, circuit3.data.pop())
                            circuit3.append(XGate(), [qr_aux[0]], [])
                            circuit3.data.insert(j, circuit3.data.pop())

                        circuit3.h(qr_aux)
                        if add_measurement:
                            circuit3.measure(qr_aux, cr_aux)
                        grad_dict[
                            circuit2.parameters.data.index(p_i), circuit2.parameters.data.index(p_j)
                        ].append(
                            LinearCombGradientCircuit(
                                circuit3, param_i.gradient(p_i) * param_j.gradient(p_j)
                            )
                        )

    return grad_dict
