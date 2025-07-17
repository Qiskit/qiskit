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

"""Synthesis for multiple-control, multiple-target X Gate."""
from __future__ import annotations

from qiskit.circuit import QuantumCircuit, QuantumRegister


def synth_mcmt_x(
        num_ctrl_qubits: int, num_target_qubits: int, ctrl_state: int | None = None
) -> QuantumCircuit:
    if len(ctrl_state) > 0:
        assert len(ctrl_state) == num_ctrl_qubits, "ctrl_state must match num_ctrl length"

    qr_c = QuantumRegister(num_ctrl_qubits, "ctrl")
    qr_t = QuantumRegister(num_target_qubits, "targ")
    qc = QuantumCircuit(qr_c, qr_t)

    if num_ctrl_qubits == 1:
        qc.cx([qr_c[0]]* num_target_qubits, qr_t, ctrl_state=ctrl_state)
        return qc

    # Linear nearest-neighbor style CX ladder before and after MCX
    for i in range(num_target_qubits - 1, 0, -1):
        qc.cx(qr_t[i - 1], qr_t[i])

    qc.mcx(qr_c, qr_t[0], ctrl_state=ctrl_state)

    for i in range(1, num_target_qubits):
        qc.cx(qr_t[i - 1], qr_t[i])

    return qc
