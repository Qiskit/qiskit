# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Synthesis algorithms for MCX gates."""
from __future__ import annotations

from math import ceil
import numpy as np

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library import CXGate, CCXGate, C3XGate, C4XGate


def synth_mcx_recursive(num_control_qubits: int, num_clean_ancillas: int) -> QuantumCircuit | None:
    """
    Synthesize MCX gate using the recursive algorithm used for ``MCXRecursive`` gates.

    This method does not require any ancilla qubits when the number of control qubits is
    up to 4, and requires exactly one ancilla qubits when the number of control qubits is
    greater than or equal to 5.

    Args:
        num_control_qubits: The number of control qubits for the gate.
        num_ancilla_qubits: The number of available ancilla qubits.

    Returns:
        The synthesized quantum circuit, or ``None`` if the number of available ancilla qubits
            is not sufficient.

    """

    print(f"In synth_mcx_recursive: {num_clean_ancillas = }")

    if num_control_qubits <= 4:
        num_clean_ancillas_required = 0
    else:
        num_clean_ancillas_required = 1

    if num_clean_ancillas < num_clean_ancillas_required:
        # Not enough ancilla qubits available to run the algorithm
        return None

    num_qubits = num_control_qubits + num_clean_ancillas_required + 1

    q = QuantumRegister(num_qubits, name="q")
    qc = QuantumCircuit(q)
    if num_control_qubits == 1:
        qc._append(CXGate(), q[:], [])
    elif num_control_qubits == 2:
        qc._append(CCXGate(), q[:], [])
    elif num_control_qubits == 3:
        qc._append(C3XGate(), q[:], [])
    elif num_control_qubits == 4:
        qc._append(C4XGate(), q[:], [])
    else:
        for instr, qargs, cargs in _recurse(q[:-1], q_ancilla=q[-1]):
            qc._append(instr, qargs, cargs)
    return qc


def _recurse(q, q_ancilla=None):
    # print(f"_RECURSE: {q = }, {q_ancilla = }")
    # recursion stop
    if len(q) == 4:
        return [(C3XGate(), q[:], [])]
    if len(q) == 5:
        return [(C4XGate(), q[:], [])]
    if len(q) < 4:
        raise AttributeError("Something went wrong in the recursion, have less than 4 qubits.")

    # recurse
    num_ctrl_qubits = len(q) - 1
    middle = ceil(num_ctrl_qubits / 2)
    first_half = [*q[:middle], q_ancilla]
    second_half = [*q[middle:num_ctrl_qubits], q_ancilla, q[num_ctrl_qubits]]

    rule = []
    rule += _recurse(first_half, q_ancilla=q[middle])
    rule += _recurse(second_half, q_ancilla=q[middle - 1])
    rule += _recurse(first_half, q_ancilla=q[middle])
    rule += _recurse(second_half, q_ancilla=q[middle - 1])

    return rule


def synth_mcx_using_mcphase(num_control_qubits: int) -> QuantumCircuit | None:
    """
    Synthesize MCX gate using the reduction to MCPhase.

    This method does not require any ancilla qubits.

    Args:
        num_control_qubits: The number of control qubits for the gate.

    Returns:
        The synthesized quantum circuit.
    """
    num_qubits = num_control_qubits + 1

    q = QuantumRegister(num_qubits, name="q")
    qc = QuantumCircuit(q)

    if num_control_qubits == 1:
        qc._append(CXGate(), q[:], [])
    elif num_control_qubits == 2:
        qc._append(CCXGate(), q[:], [])
    elif num_control_qubits == 3:
        qc._append(C3XGate(), q[:], [])
    elif num_control_qubits == 4:
        qc._append(C4XGate(), q[:], [])
    else:
        q_controls = list(range(num_control_qubits))
        q_target = num_control_qubits
        qc.h(q_target)
        qc.mcp(np.pi, q_controls, q_target)
        qc.h(q_target)

    return qc
