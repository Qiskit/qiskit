# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Compute modular sum of two qubit registers without any ancillary qubits."""

from __future__ import annotations
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit.library import XGate, MCMTGate


def adder_modular_v17(num_qubits: int) -> QuantumCircuit:
    r"""
    Gidney's variant of Van Rentergem Adder without any carry qubits based on Fig. 15 of [1].

    [1] Gidney, https://arxiv.org/abs/1706.07884
    """

    if num_qubits < 1:
        raise ValueError("The number of qubits must be at least 1.")

    qr_a = QuantumRegister(num_qubits, "a")
    qr_b = QuantumRegister(num_qubits, "b")
    qc = QuantumCircuit(qr_a, qr_b)

    if num_qubits == 1:
        qc.cx(qr_a[0], qr_b[0])
        return qc

    mcmt_a = MCMTGate(XGate(), 1, num_qubits - 1)
    mcmt_b = MCMTGate(XGate(), 1, num_qubits)

    qc.compose(mcmt_a, [qr_a[-1]] + qr_a[:-1], inplace=True)
    qc.compose(mcmt_b, [qr_a[-1]] + qr_b[:], inplace=True)

    # Ripple forward.
    for i in range(num_qubits - 1):
        qc.cx(qr_a[-1], qr_b[i])
        qc.cx(qr_a[i], qr_a[-1])
        qc.rccx(qr_b[i], qr_a[-1], qr_a[i])
        qc.cx(qr_a[i], qr_a[-1])

    qc.cx(qr_a[-1], qr_b[-1])  # High bit toggle.

    # Ripple backward.
    for i in range(num_qubits - 2, -1, -1):
        qc.cx(qr_a[i], qr_a[-1])
        qc.rccx(qr_b[i], qr_a[-1], qr_a[i])
        qc.cx(qr_a[i], qr_a[-1])
        qc.cx(qr_a[i], qr_b[i])

    qc.compose(mcmt_b, [qr_a[-1]] + qr_b[:], inplace=True)
    qc.compose(mcmt_a, [qr_a[-1]] + qr_a[:-1], inplace=True)

    return qc
