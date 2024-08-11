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

"""Module containing multi-controlled circuits synthesis with ancillary qubits."""

from math import ceil
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates.x import C3XGate, C4XGate


def synth_mcx_n_dirty_i15(
    num_ctrl_qubits: int,
    relative_phase: bool = False,
    action_only: bool = False,
):
    """
    Synthesize a multi-controlled X gate with :math:`k` controls using :math:`k - 2`
    dirty ancillary qubits producing a circuit with :math:`2 * k - 1` qubits and at most
    :math:`8 * k - 6` CX gates, by Iten et. al. [1].

    Args:
        num_ctrl_qubits: The number of control qubits.

        relative_phase: when set to ``True``, the method applies the optimized multi-controlled X gate
            up to a relative phase, in a way that, by lemma 8 of [1], the relative
            phases of the ``action part`` cancel out with the phases of the ``reset part``.

        action_only: when set to ``True``, the method applies only the ``action part`` of lemma 8 of [1].

    Returns:
        The synthesized quantum circuit.

    References:
        1. Iten et. al., *Quantum Circuits for Isometries*, Phys. Rev. A 93, 032318 (2016),
           `arXiv:1501.06911 <http://arxiv.org/abs/1501.06911>`_
    """

    num_qubits = 2 * num_ctrl_qubits - 1
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


def synth_mcx_n_clean_m15(num_ctrl_qubits: int):
    """
    Synthesize a multi-controlled X gate with :math:`k` controls using :math:`k - 2`
    clean ancillary qubits with producing a circuit with :math:`2 * k - 1` qubits
    and at most :math:`6 * k - 6` CX gates, by Maslov [1].

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    References:
        1. Maslov., Phys. Rev. A 93, 022311 (2016),
           `arXiv:1508.03273 <https://arxiv.org/pdf/1508.03273>`_
    """

    num_qubits = 2 * num_ctrl_qubits - 1
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


def synth_mcx_1_clean_b95(num_ctrl_qubits: int):
    """
    Synthesize a multi-controlled X gate with :math:`k` controls using a single
    clean ancillary qubit producing a circuit with :math:`k + 2` qubits and at most
    :math:`16 * k - 8` CX gates, by Barenco et al. [1].

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    References:
        1. Barenco et. al., Phys.Rev. A52 3457 (1995),
           `arXiv:quant-ph/9503016 <https://arxiv.org/abs/quant-ph/9503016>`_
    """

    if num_ctrl_qubits == 3:
        q = QuantumRegister(4, name="q")
        qc = QuantumCircuit(q, name="mcx")
        qc._append(C3XGate(), q[:], [])
        return qc

    elif num_ctrl_qubits == 4:
        q = QuantumRegister(5, name="q")
        qc = QuantumCircuit(q, name="mcx")
        qc._append(C4XGate(), q[:], [])
        return qc

    num_qubits = num_ctrl_qubits + 2
    q = QuantumRegister(num_qubits, name="q")
    qc = QuantumCircuit(q, name="mcx_recursive")

    num_ctrl_qubits = len(q) - 1
    q_ancilla = q[-1]
    q_target = q[-2]
    middle = ceil(num_ctrl_qubits / 2)
    first_half = [*q[:middle]]
    second_half = [*q[middle : num_ctrl_qubits - 1], q_ancilla]

    qc_first_half = synth_mcx_n_dirty_i15(num_ctrl_qubits=len(first_half))
    qc_second_half = synth_mcx_n_dirty_i15(num_ctrl_qubits=len(second_half))

    qc.append(
        qc_first_half,
        qargs=[*first_half, q_ancilla, *q[middle : middle + len(first_half) - 2]],
        cargs=[],
    )
    qc.append(
        qc_second_half,
        qargs=[*second_half, q_target, *q[: len(second_half) - 2]],
        cargs=[],
    )
    qc.append(
        qc_first_half,
        qargs=[*first_half, q_ancilla, *q[middle : middle + len(first_half) - 2]],
        cargs=[],
    )
    qc.append(
        qc_second_half,
        qargs=[*second_half, q_target, *q[: len(second_half) - 2]],
        cargs=[],
    )

    return qc
