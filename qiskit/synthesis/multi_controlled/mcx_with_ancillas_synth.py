# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Module containing multi-controlled circuits synthesis with ancillary qubits."""

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit


def synth_mcx_n_dirty_ancillas_ickhc(
    num_qubits: int,
    num_ctrl_qubits: int = None,
    relative_phase: bool = False,
    action_only: bool = False,
):
    """Synthesis of an MCX gate with n controls and n-2 dirty ancillary qubits,
    producing a circuit with at most 8*n-6 CX gates"""

    q = QuantumRegister(num_qubits, name="q")
    qc = QuantumCircuit(q, name="mcx_vchain")
    q_controls = q[:num_ctrl_qubits]
    q_target = q[num_ctrl_qubits]
    q_ancillas = q[num_ctrl_qubits + 1 :]

    if num_ctrl_qubits == 1:
        qc.cx(q_controls, q_target)
        return qc
    elif num_ctrl_qubits == 2:
        qc.ccx(q_controls[0], q_controls[1], q_target)
        return qc
    elif not relative_phase and num_ctrl_qubits == 3:
        # pylint: disable=cyclic-import
        from qiskit.circuit.library.standard_gates.x import C3XGate

        qc._append(C3XGate(), [*q_controls, q_target], [])
        return qc

    num_ancillas = num_ctrl_qubits - 2
    targets = [q_target] + q_ancillas[:num_ancillas][::-1]

    for j in range(2):
        for i in range(num_ctrl_qubits):  # action part
            if i < num_ctrl_qubits - 2:
                if targets[i] != q_target or relative_phase:
                    # gate cancelling

                    # cancel rightmost gates of action part
                    # with leftmost gates of reset part
                    if relative_phase and targets[i] == q_target and j == 1:
                        qc.cx(q_ancillas[num_ancillas - i - 1], targets[i])
                        qc.t(targets[i])
                        qc.cx(q_controls[num_ctrl_qubits - i - 1], targets[i])
                        qc.tdg(targets[i])
                        qc.h(targets[i])
                    else:
                        qc.h(targets[i])
                        qc.t(targets[i])
                        qc.cx(q_controls[num_ctrl_qubits - i - 1], targets[i])
                        qc.tdg(targets[i])
                        qc.cx(q_ancillas[num_ancillas - i - 1], targets[i])
                else:
                    controls = [
                        q_controls[num_ctrl_qubits - i - 1],
                        q_ancillas[num_ancillas - i - 1],
                    ]

                    qc.ccx(controls[0], controls[1], targets[i])
            else:
                # implements an optimized toffoli operation
                # up to a diagonal gate, akin to lemma 6 of arXiv:1501.06911
                qc.h(targets[i])
                qc.t(targets[i])
                qc.cx(q_controls[num_ctrl_qubits - i - 2], targets[i])
                qc.tdg(targets[i])
                qc.cx(q_controls[num_ctrl_qubits - i - 1], targets[i])
                qc.t(targets[i])
                qc.cx(q_controls[num_ctrl_qubits - i - 2], targets[i])
                qc.tdg(targets[i])
                qc.h(targets[i])

                break

        for i in range(num_ancillas - 1):  # reset part
            qc.cx(q_ancillas[i], q_ancillas[i + 1])
            qc.t(q_ancillas[i + 1])
            qc.cx(q_controls[2 + i], q_ancillas[i + 1])
            qc.tdg(q_ancillas[i + 1])
            qc.h(q_ancillas[i + 1])

        if action_only:
            qc.ccx(q_controls[-1], q_ancillas[-1], q_target)

            break

    return qc


def synth_mcx_n_clean_ancillas(num_qubits: int, num_ctrl_qubits: int = None):
    """Synthesis of an MCX gate with n controls and n-2 clean ancillary qubits,
    producing a circuit with at most 6*n-6 CX gates"""

    q = QuantumRegister(num_qubits, name="q")
    qc = QuantumCircuit(q, name="mcx_vchain")
    q_controls = q[:num_ctrl_qubits]
    q_target = q[num_ctrl_qubits]
    q_ancillas = q[num_ctrl_qubits + 1 :]

    qc.rccx(q_controls[0], q_controls[1], q_ancillas[0])
    i = 0
    for j in range(2, num_ctrl_qubits - 1):
        qc.rccx(q_controls[j], q_ancillas[i], q_ancillas[i + 1])

        i += 1

    qc.ccx(q_controls[-1], q_ancillas[i], q_target)

    for j in reversed(range(2, num_ctrl_qubits - 1)):
        qc.rccx(q_controls[j], q_ancillas[i - 1], q_ancillas[i])

        i -= 1

    qc.rccx(q_controls[0], q_controls[1], q_ancillas[i])

    return qc
