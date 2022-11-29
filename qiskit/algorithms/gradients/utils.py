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
from dataclasses import dataclass
from typing import Sequence

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

@dataclass
class GradientCircuit:
    """Stores gradient circuit data for the parameter shift method"""

    gradient_circuit: QuantumCircuit
    """An internal quantum circuit used to calculate the gradient"""
    parameter_map: dict[Parameter, list[tuple[Parameter, float | ParameterExpression]]]
    """A dictionary maps the parameters of ``circuit`` to the parameters of ``gradient_circuit`` with
    coefficients"""
    gradient_parameter_map: dict[Parameter, ParameterExpression]
    """A dictionary maps the parameters of ``gradient_circuit`` to the parameter expressions of ``circuit``"""


def _make_param_shift_parameter_values(
    circuit: QuantumCircuit,
    parameter_values: np.ndarray,
    parameter_set: set[Parameter],
) -> list[np.ndarray]:
    """Makes the final parameter values for the parameter shift method by adding each parameter value
        to the base parameter values.

    Args:
        circuit: The original quantum circuit
        parameter_values: parameter values to be added to the base parameter values.
        param_set: set of parameters to be differentiated

    Returns:
        The final parameter values for the parameter shift method and the coefficients.
    """
    plus_offsets, minus_offsets = [], []
    indices = [idx for idx, param in enumerate(circuit.parameters) if param in parameter_set]
    offset = np.identity(circuit.num_parameters)[indices, :]
    plus_offsets = parameter_values + offset * np.pi / 2
    minus_offsets = parameter_values - offset * np.pi / 2
    return plus_offsets.tolist() + minus_offsets.tolist()


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

def _param_shift_preprocessing(circuit: QuantumCircuit) -> ParameterShiftGradientCircuit:
    """Preprocessing for the parameter shift method.
    Args:
        circuit: The original quantum circuit
    Returns:
        necessary data to calculate gradients with the parameter shift method.
    """
    return None

