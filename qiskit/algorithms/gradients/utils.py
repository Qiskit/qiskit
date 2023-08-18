# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
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

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np

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
from qiskit.quantum_info import SparsePauliOp


################################################################################
## Gradient circuits and Enum
################################################################################
class DerivativeType(Enum):
    """Types of derivative."""

    REAL = "real"
    IMAG = "imag"
    COMPLEX = "complex"


@dataclass
class GradientCircuit:
    """Gradient circuit with unique parameters and mapping information."""

    gradient_circuit: QuantumCircuit
    """An internal quantum circuit with unique parameters used to calculate the gradient"""
    parameter_map: dict[Parameter, list[tuple[Parameter, float | ParameterExpression]]]
    """A dictionary maps the parameters of ``circuit`` to the parameters of ``gradient_circuit`` with
    coefficients"""
    gradient_parameter_map: dict[Parameter, ParameterExpression]
    """A dictionary maps the parameters of ``gradient_circuit`` to the parameter expressions of
    ``circuit``"""


@dataclass
class LinearCombGradientCircuit:
    """Gradient circuit for the linear combination of unitaries method."""

    gradient_circuit: QuantumCircuit
    """A gradient circuit  for the linear combination of unitaries method."""
    coeff: float | ParameterExpression
    """A coefficient corresponds to the gradient circuit."""


################################################################################
## Parameter shift gradient
################################################################################
def _make_param_shift_parameter_values(
    circuit: QuantumCircuit,
    parameter_values: np.ndarray | list[float],
    parameters: Sequence[Parameter],
) -> list[np.ndarray]:
    """Returns a list of parameter values with offsets for parameter shift rule.

    Args:
        circuit: The original quantum circuit
        parameter_values: parameter values to be added to the base parameter values.
        parameters: The parameters to be shifted.

    Returns:
        A list of parameter values with offsets for parameter shift rule.
    """
    indices = [circuit.parameters.data.index(p) for p in parameters]
    offset = np.identity(circuit.num_parameters)[indices, :]
    plus_offsets = parameter_values + offset * np.pi / 2
    minus_offsets = parameter_values - offset * np.pi / 2
    return plus_offsets.tolist() + minus_offsets.tolist()


################################################################################
## Linear combination gradient and Linear combination QGT
################################################################################
def _make_lin_comb_gradient_circuit(
    circuit: QuantumCircuit, add_measurement: bool = False
) -> dict[Parameter, QuantumCircuit]:
    """Makes a circuit that computes the linear combination of the gradient circuits."""
    circuit_temp = circuit.copy()
    qr_aux = QuantumRegister(1, "qr_aux")
    cr_aux = ClassicalRegister(1, "cr_aux")
    circuit_temp.add_register(qr_aux)
    circuit_temp.add_register(cr_aux)
    circuit_temp.h(qr_aux)
    circuit_temp.data.insert(0, circuit_temp.data.pop())
    circuit_temp.sdg(qr_aux)
    circuit_temp.data.insert(1, circuit_temp.data.pop())

    lin_comb_circuits = {}
    for i, instruction in enumerate(circuit_temp.data):
        if instruction.operation.is_parameterized():
            for p in instruction.operation.params[0].parameters:
                gate = _gate_gradient(instruction.operation)
                lin_comb_circuit = circuit_temp.copy()
                # insert `gate` to i-th position
                lin_comb_circuit.append(gate, [qr_aux[0]] + list(instruction.qubits), [])
                lin_comb_circuit.data.insert(i, lin_comb_circuit.data.pop())
                lin_comb_circuit.h(qr_aux)
                if add_measurement:
                    lin_comb_circuit.measure(qr_aux, cr_aux)
                lin_comb_circuits[p] = lin_comb_circuit

    return lin_comb_circuits


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


def _make_lin_comb_qgt_circuit(
    circuit: QuantumCircuit, add_measurement: bool = False
) -> dict[tuple[Parameter, Parameter], QuantumCircuit]:
    """Makes a circuit that computes the linear combination of the QGT circuits."""
    circuit_temp = circuit.copy()
    qr_aux = QuantumRegister(1, "aux")
    circuit_temp.add_register(qr_aux)
    if add_measurement:
        cr_aux = ClassicalRegister(1, "aux")
        circuit_temp.add_bits(cr_aux)
    circuit_temp.h(qr_aux)
    circuit_temp.data.insert(0, circuit_temp.data.pop())

    lin_comb_qgt_circuits = {}
    for i, instruction_i in enumerate(circuit_temp.data):
        if not instruction_i.operation.is_parameterized():
            continue
        for j, instruction_j in enumerate(circuit_temp.data):
            if not instruction_j.operation.is_parameterized():
                continue
            # Calculate the QGT of the i-th gate with respect to the j-th gate.
            param_i = instruction_i.operation.params[0]
            param_j = instruction_j.operation.params[0]

            for p_i in param_i.parameters:
                for p_j in param_j.parameters:
                    if circuit_temp.parameters.data.index(p_i) > circuit_temp.parameters.data.index(
                        p_j
                    ):
                        continue
                    gate_i = _gate_gradient(instruction_i.operation)
                    gate_j = _gate_gradient(instruction_j.operation)
                    lin_comb_qgt_circuit = circuit_temp.copy()
                    if i < j:
                        # insert gate_j to j-th position
                        lin_comb_qgt_circuit.append(
                            gate_j, [qr_aux[0]] + list(instruction_j.qubits), []
                        )
                        lin_comb_qgt_circuit.data.insert(j, lin_comb_qgt_circuit.data.pop())
                        # insert gate_i to i-th position with two X gates at its sides
                        lin_comb_qgt_circuit.append(XGate(), [qr_aux[0]], [])
                        lin_comb_qgt_circuit.data.insert(i, lin_comb_qgt_circuit.data.pop())
                        lin_comb_qgt_circuit.append(
                            gate_i, [qr_aux[0]] + list(instruction_i.qubits), []
                        )
                        lin_comb_qgt_circuit.data.insert(i, lin_comb_qgt_circuit.data.pop())
                        lin_comb_qgt_circuit.append(XGate(), [qr_aux[0]], [])
                        lin_comb_qgt_circuit.data.insert(i, lin_comb_qgt_circuit.data.pop())
                    else:
                        # insert gate_i to i-th position
                        lin_comb_qgt_circuit.append(
                            gate_i, [qr_aux[0]] + list(instruction_i.qubits), []
                        )
                        lin_comb_qgt_circuit.data.insert(i, lin_comb_qgt_circuit.data.pop())
                        # insert gate_j to j-th position with two X gates at its sides
                        lin_comb_qgt_circuit.append(XGate(), [qr_aux[0]], [])
                        lin_comb_qgt_circuit.data.insert(j, lin_comb_qgt_circuit.data.pop())
                        lin_comb_qgt_circuit.append(
                            gate_j, [qr_aux[0]] + list(instruction_j.qubits), []
                        )
                        lin_comb_qgt_circuit.data.insert(j, lin_comb_qgt_circuit.data.pop())
                        lin_comb_qgt_circuit.append(XGate(), [qr_aux[0]], [])
                        lin_comb_qgt_circuit.data.insert(j, lin_comb_qgt_circuit.data.pop())

                    lin_comb_qgt_circuit.h(qr_aux)
                    if add_measurement:
                        lin_comb_qgt_circuit.measure(qr_aux, cr_aux)
                    lin_comb_qgt_circuits[(p_i, p_j)] = lin_comb_qgt_circuit

    return lin_comb_qgt_circuits


def _make_lin_comb_observables(
    observable: SparsePauliOp,
    derivative_type: DerivativeType,
) -> tuple[SparsePauliOp, SparsePauliOp | None]:
    """Make the observable with an ancillary operator for the linear combination gradient.

    Args:
        observable: The observable.
        derivative_type: The type of derivative. Can be either ``DerivativeType.REAL``
            ``DerivativeType.IMAG``, or ``DerivativeType.COMPLEX``.

    Returns:
        The observable with an ancillary operator for the linear combination gradient.

    Raises:
        ValueError: If the derivative type is not supported.
    """
    if derivative_type == DerivativeType.REAL:
        return observable.expand(SparsePauliOp.from_list([("Z", 1)])), None
    elif derivative_type == DerivativeType.IMAG:
        return observable.expand(SparsePauliOp.from_list([("Y", -1)])), None
    elif derivative_type == DerivativeType.COMPLEX:
        return observable.expand(SparsePauliOp.from_list([("Z", 1)])), observable.expand(
            SparsePauliOp.from_list([("Y", -1)])
        )
    else:
        raise ValueError(f"Derivative type {derivative_type} is not supported.")


################################################################################
## Preprocess
################################################################################
def _assign_unique_parameters(
    circuit: QuantumCircuit,
) -> GradientCircuit:
    """Assign unique parameters to the circuit.

    Args:
        circuit: The circuit to assign unique parameters.

    Returns:
        The circuit with unique parameters and the mapping from the original parameters to the
        unique parameters.
    """
    gradient_circuit = circuit.copy_empty_like(f"{circuit.name}_gradient")
    parameter_map = defaultdict(list)
    gradient_parameter_map = {}
    num_gradient_parameters = 0
    for instruction in circuit.data:
        if instruction.operation.is_parameterized():
            new_op_params = []
            for angle in instruction.operation.params:
                new_parameter = Parameter(f"__gθ{num_gradient_parameters}")
                new_op_params.append(new_parameter)
                num_gradient_parameters += 1
                for parameter in angle.parameters:
                    parameter_map[parameter].append((new_parameter, angle.gradient(parameter)))
                gradient_parameter_map[new_parameter] = angle
            instruction.operation.params = new_op_params
        gradient_circuit.append(instruction.operation, instruction.qubits, instruction.clbits)
    # For the global phase
    gradient_circuit.global_phase = circuit.global_phase
    if isinstance(gradient_circuit.global_phase, ParameterExpression):
        substitution_map = {}
        for parameter in gradient_circuit.global_phase.parameters:
            if parameter in parameter_map:
                substitution_map[parameter] = parameter_map[parameter][0][0]
            else:
                new_parameter = Parameter(f"__gθ{num_gradient_parameters}")
                substitution_map[parameter] = new_parameter
                parameter_map[parameter].append((new_parameter, 1))
                num_gradient_parameters += 1
        gradient_circuit.global_phase = gradient_circuit.global_phase.subs(substitution_map)
    return GradientCircuit(gradient_circuit, parameter_map, gradient_parameter_map)


def _make_gradient_parameter_values(
    circuit: QuantumCircuit,
    gradient_circuit: GradientCircuit,
    parameter_values: np.ndarray,
) -> np.ndarray:
    """Makes parameter values for the gradient circuit.

    Args:
        circuit: The original quantum circuit
        gradient_circuit: The gradient circuit
        parameter_values: The parameter values for the original circuit
        parameter_set: The parameter set to calculate gradients

    Returns:
        The parameter values for the gradient circuit.
    """
    g_circuit = gradient_circuit.gradient_circuit
    g_parameter_values = np.empty(len(g_circuit.parameters))
    for i, g_parameter in enumerate(g_circuit.parameters):
        expr = gradient_circuit.gradient_parameter_map[g_parameter]
        bound_expr = expr.bind(
            {p: parameter_values[circuit.parameters.data.index(p)] for p in expr.parameters}
        )
        g_parameter_values[i] = float(bound_expr)
    return g_parameter_values


def _make_gradient_parameters(
    gradient_circuit: GradientCircuit,
    parameters: Sequence[Parameter],
) -> Sequence[Parameter]:
    """Makes parameter set for the gradient circuit.

    Args:
        gradient_circuit: The gradient circuit
        parameters: The parameters in the original circuit to calculate gradients

    Returns:
        The parameters in the gradient circuit to calculate gradients.
    """
    g_parameters = [
        g_parameter
        for parameter in parameters
        for g_parameter, _ in gradient_circuit.parameter_map[parameter]
    ]
    # make g_parameters unique and return it.
    return list(dict.fromkeys(g_parameters))
