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

"""Module containing multi-controlled circuits synthesis with and without ancillary qubits."""

from typing import List, Tuple
from math import ceil
import numpy as np

from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import (
    HGate,
    MCU1Gate,
    CU1Gate,
    RC3XGate,
    C3SXGate,
)


def synth_mcx_n_dirty_i15(
    num_ctrl_qubits: int,
    relative_phase: bool = False,
    action_only: bool = False,
) -> QuantumCircuit:
    r"""
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

    if num_ctrl_qubits == 1:
        num_qubits = 2
    else:
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
        circuit = synth_c3x()
        qc.compose(circuit, [*q_controls, q_target], inplace=True, copy=False)
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


def synth_mcx_n_clean_m15(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
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


def synth_mcx_1_clean_b95(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
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
        return synth_c3x()

    elif num_ctrl_qubits == 4:
        return synth_c4x()

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


def synth_mcx_gray_code(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k` controls using the Gray code.

    Produces a quantum circuit with :math:`k + 1` qubits. This method
    produces exponentially many CX gates and should be used only for small
    values of :math:`k`.

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.
    """
    num_qubits = num_ctrl_qubits + 1
    q = QuantumRegister(num_qubits, name="q")
    qc = QuantumCircuit(q, name="mcx_gray")
    qc._append(HGate(), [q[-1]], [])
    qc._append(MCU1Gate(np.pi, num_ctrl_qubits=num_ctrl_qubits), q[:], [])
    qc._append(HGate(), [q[-1]], [])
    return qc


def synth_mcx_noaux_v24(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesize a multi-controlled X gate with :math:`k` controls based on
    the implementation for MCPhaseGate.

    In turn, the MCPhase gate uses the decomposition for multi-controlled
    special unitaries described in [1].

    Produces a quantum circuit with :math:`k + 1` qubits.
    The number of CX-gates is quadratic in :math:`k`.

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    References:
        1. Vale et. al., *Circuit Decomposition of Multicontrolled Special Unitary
           Single-Qubit Gates*, IEEE TCAD 43(3) (2024),
           `arXiv:2302.06377 <https://arxiv.org/abs/2302.06377>`_
    """
    if num_ctrl_qubits == 3:
        return synth_c3x()

    if num_ctrl_qubits == 4:
        return synth_c4x()

    num_qubits = num_ctrl_qubits + 1
    q = QuantumRegister(num_qubits, name="q")
    qc = QuantumCircuit(q)
    q_controls = list(range(num_ctrl_qubits))
    q_target = num_ctrl_qubits
    qc.h(q_target)
    qc.mcp(np.pi, q_controls, q_target)
    qc.h(q_target)
    return qc


def linear_depth_ladder_ops(qreg: List[int]) -> Tuple[QuantumCircuit, List[int]]:
    r"""
    Helper function to create linear-depth ladder operations used in Khattar and Gidney's MCX synthesis.
    In particular, this implements Step-1 and Step-2 on Fig. 3 of [1] except for the first and last
    CCX gates.

    Args:
        qreg: List of qubit indices to apply the ladder operations on. qreg[0] is assumed to be ancilla.

    Returns:
        QuantumCircuit: Linear-depth ladder circuit.
        int: Index of control qubit to apply the final CCX gate.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arxiv:2407.17966 https://arxiv.org/abs/2407.17966`_
    """

    n = len(qreg)
    assert n > 3, "n = n_ctrls + 1 => n_ctrls >= 3 to use MCX ladder. Otherwise, use CCX"
    qc = QuantumCircuit(n)

    # up-ladder
    for i in range(2, n - 2, 2):
        qc.ccx(qreg[i + 1], qreg[i + 2], qreg[i])
        qc.x(qreg[i])

    # down-ladder
    if n % 2 != 0:
        x, y, t = n - 3, n - 5, n - 6
    else:
        x, y, t = n - 1, n - 4, n - 5

    if t > 0:
        qc.ccx(qreg[x], qreg[y], qreg[t])
        qc.x(qreg[t])

    for i in range(t, 2, -2):
        qc.ccx(qreg[i], qreg[i - 1], qreg[i - 2])
        qc.x(qreg[i - 2])

    mid_second_ctrl = 1 + max(0, 6 - n)
    final_ctrl = qreg[mid_second_ctrl] - 1
    return qc, final_ctrl


def synth_mcx_1_kg24(num_ctrl_qubits: int, clean=True) -> QuantumCircuit:
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`1` ancillary qubit as
    described in Sec. 5 of [1].

    Args:
        num_ctrl_qubits: The number of control qubits.
        clean: If True, the ancilla is clean, otherwise it is dirty.

    Returns:
        The synthesized quantum circuit.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arxiv:2407.17966 https://arxiv.org/abs/2407.17966`_
    """

    q_controls = QuantumRegister(num_ctrl_qubits, name="ctrl")
    q_target = QuantumRegister(1, name="targ")
    q_ancilla = QuantumRegister(1, name="anc")
    qc = QuantumCircuit(q_controls, q_target, q_ancilla, name="mcx_linear_depth")

    ladder_ops, final_ctrl = linear_depth_ladder_ops(list(range(num_ctrl_qubits + 1)))
    qc.ccx(q_controls[0], q_controls[1], q_ancilla)  #                  # create cond. clean ancilla
    qc.compose(ladder_ops, q_ancilla[:] + q_controls[:], inplace=True)  # up-ladder
    qc.ccx(q_ancilla, q_controls[final_ctrl], q_target)  #              # target
    qc.compose(  #                                                      # down-ladder
        ladder_ops.inverse(),
        q_ancilla[:] + q_controls[:],
        inplace=True,
    )
    qc.ccx(q_controls[0], q_controls[1], q_ancilla)

    if not clean:
        # perform toggle-detection if ancilla is dirty
        qc.compose(ladder_ops, q_ancilla[:] + q_controls[:], inplace=True)
        qc.ccx(q_ancilla, q_controls[0], q_target)
        qc.compose(ladder_ops.inverse(), q_ancilla[:] + q_controls[:], inplace=True)

    return qc


def synth_mcx_1_clean_kg24(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`1` clean ancillary qubit
    producing a circuit with :math:`2k-3` Toffoli gates and depth :math:`O(k)` as described in Sec. 5.1 of [1].

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arxiv:2407.17966 https://arxiv.org/abs/2407.17966`_
    """

    return synth_mcx_1_kg24(num_ctrl_qubits, clean=True)


def synth_mcx_1_dirty_kg24(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`1` dirty ancillary qubit
    producing a circuit with :math:`4k-8` Toffoli gates and depth :math:`O(k)` as described in Sec. 5.2 of [1].

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arxiv:2407.17966 https://arxiv.org/abs/2407.17966`_
    """

    return synth_mcx_1_kg24(num_ctrl_qubits, clean=False)


def CCXN(n):
    r"""
    Construct a quantum circuit for creating n-condionally clean ancillae using 3n qubits. This
    implements Fig. 4a of [1]. The order of returned qubits is x, y, target.

    Args:
        n: Number of conditionally clean ancillae to create.

    Returns:
        QuantumCircuit: The quantum circuit for creating n-condionally clean ancillae.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arxiv:2407.17966 https://arxiv.org/abs/2407.17966`_
    """

    n_qubits = 3 * n
    q = QuantumRegister(n_qubits, name="q")
    qc = QuantumCircuit(q, name=f"ccxn_{n}")
    x, y, t = q[:n], q[n : 2 * n], q[2 * n :]
    for x, y, t in zip(x, y, t):
        qc.x(t)
        qc.ccx(x, y, t)

    return qc


def build_logn_depth_ccx_ladder(
    ancilla_idx: int, ctrls: List[int], skip_cond_clean=False
) -> Tuple[QuantumCircuit, List[int]]:
    r"""
    Helper function to build a log-depth ladder compose of CCX and X gates as shown in Fig. 4b of [1].

    Args:
        alloc_anc: Index of the ancillary qubit.
        ctrls: List of control qubits.
        skip_cond_clean: If True, do not include the conditionally clean ancilla (step 1 and 5 in
        Fig. 4b of [1]).

    Returns:
        QuantumCircuit: The log-depth ladder circuit.
        List[int]: List of remaining control qubits.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arxiv:2407.17966 https://arxiv.org/abs/2407.17966`_
    """

    qc = QuantumCircuit(len(ctrls) + 1)
    anc = [ancilla_idx]
    final_ctrls = []

    while len(ctrls) > 1:
        next_batch_len = min(len(anc) + 1, len(ctrls))
        ctrls, nxt_batch = ctrls[next_batch_len:], ctrls[:next_batch_len]
        new_anc = []
        while len(nxt_batch) > 1:
            ccx_n = len(nxt_batch) // 2
            st = int(len(nxt_batch) % 2)
            ccx_x, ccx_y, ccx_t = (
                nxt_batch[st : st + ccx_n],
                nxt_batch[st + ccx_n :],
                anc[-ccx_n:],
            )
            assert len(ccx_x) == len(ccx_y) == len(ccx_t) == ccx_n >= 1
            if ccx_t != [ancilla_idx]:
                qc.compose(CCXN(ccx_n), ccx_x + ccx_y + ccx_t, inplace=True)
            else:
                if not skip_cond_clean:
                    qc.ccx(ccx_x[0], ccx_y[0], ccx_t[0])  #  # create conditionally clean ancilla
            new_anc += nxt_batch[st:]  #                     # newly created cond. clean ancilla
            nxt_batch = ccx_t + nxt_batch[:st]
            anc = anc[:-ccx_n]

        anc = sorted(anc + new_anc)
        final_ctrls += nxt_batch

    final_ctrls += ctrls
    final_ctrls = sorted(final_ctrls)
    return qc, final_ctrls[:-1]  #                          # exclude ancilla


def synth_mcx_2_kg24(num_ctrl_qubits: int, clean=True) -> QuantumCircuit:
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`2` ancillary qubits.

    Args:
        num_ctrl_qubits: The number of control qubits.
        clean: If True, the ancilla is clean, otherwise it is dirty.

    Returns:
        The synthesized quantum circuit.
    """

    q_control = QuantumRegister(num_ctrl_qubits, name="ctrl")
    q_target = QuantumRegister(1, name="targ")
    q_ancilla = QuantumRegister(2, name="anc")
    qc = QuantumCircuit(q_control, q_target, q_ancilla, name="mcx_logn_depth")

    ladder_ops, final_ctrls = build_logn_depth_ccx_ladder(
        num_ctrl_qubits, list(range(num_ctrl_qubits))
    )
    qc.compose(ladder_ops, q_control[:] + [q_ancilla[0]], inplace=True)
    if len(final_ctrls) == 1:  # Already a toffoli
        qc.ccx(q_ancilla[0], q_control[final_ctrls[0]], q_target)
    else:
        mid_mcx = synth_mcx_1_clean_kg24(len(final_ctrls) + 1)
        qc.compose(
            mid_mcx,
            [q_ancilla[0]]
            + q_control[final_ctrls]
            + q_target[:]
            + [q_ancilla[1]],  # ctrls, targ, anc
            inplace=True,
        )
    qc.compose(ladder_ops.inverse(), q_control[:] + [q_ancilla[0]], inplace=True)

    if not clean:
        # perform toggle-detection if ancilla is dirty
        ladder_ops_new, final_ctrls = build_logn_depth_ccx_ladder(
            num_ctrl_qubits, list(range(num_ctrl_qubits)), skip_cond_clean=True
        )
        qc.compose(ladder_ops_new, q_control[:] + [q_ancilla[0]], inplace=True)
        if len(final_ctrls) == 1:
            qc.ccx(q_ancilla[0], q_control[final_ctrls[0]], q_target)
        else:
            qc.compose(
                mid_mcx,
                [q_ancilla[0]] + q_control[final_ctrls] + q_target[:] + [q_ancilla[1]],
                inplace=True,
            )
        qc.compose(ladder_ops_new.inverse(), q_control[:] + [q_ancilla[0]], inplace=True)

    return qc


def synth_mcx_2_clean_kg24(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`2` clean ancillary qubits
    producing a circuit with :math:`2k-3` Toffoli gates and depth :math:`O(\log(k))` as described in
    Sec. 5.3 of [1].

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arxiv:2407.17966 https://arxiv.org/abs/2407.17966`_
    """

    return synth_mcx_2_kg24(num_ctrl_qubits, clean=True)


def synth_mcx_2_dirty_kg24(num_ctrl_qubits: int) -> QuantumCircuit:
    r"""
    Synthesise a multi-controlled X gate with :math:`k` controls using :math:`2` dirty ancillary qubits
    producing a circuit with :math:`4k-8` Toffoli gates and depth :math:`O(\log(k))` as described in
    Sec. 5.4 of [1].

    Args:
        num_ctrl_qubits: The number of control qubits.

    Returns:
        The synthesized quantum circuit.

    References:
        1. Khattar and Gidney, Rise of conditionally clean ancillae for optimizing quantum circuits
        `arxiv:2407.17966 https://arxiv.org/abs/2407.17966`_
    """

    return synth_mcx_2_kg24(num_ctrl_qubits, clean=False)


def synth_c3x() -> QuantumCircuit:
    """Efficient synthesis of 3-controlled X-gate."""

    q = QuantumRegister(4, name="q")
    qc = QuantumCircuit(q, name="mcx")
    qc.h(3)
    qc.p(np.pi / 8, [0, 1, 2, 3])
    qc.cx(0, 1)
    qc.p(-np.pi / 8, 1)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.p(-np.pi / 8, 2)
    qc.cx(0, 2)
    qc.p(np.pi / 8, 2)
    qc.cx(1, 2)
    qc.p(-np.pi / 8, 2)
    qc.cx(0, 2)
    qc.cx(2, 3)
    qc.p(-np.pi / 8, 3)
    qc.cx(1, 3)
    qc.p(np.pi / 8, 3)
    qc.cx(2, 3)
    qc.p(-np.pi / 8, 3)
    qc.cx(0, 3)
    qc.p(np.pi / 8, 3)
    qc.cx(2, 3)
    qc.p(-np.pi / 8, 3)
    qc.cx(1, 3)
    qc.p(np.pi / 8, 3)
    qc.cx(2, 3)
    qc.p(-np.pi / 8, 3)
    qc.cx(0, 3)
    qc.h(3)
    return qc


def synth_c4x() -> QuantumCircuit:
    """Efficient synthesis of 4-controlled X-gate."""

    q = QuantumRegister(5, name="q")
    qc = QuantumCircuit(q, name="mcx")

    rules = [
        (HGate(), [q[4]], []),
        (CU1Gate(np.pi / 2), [q[3], q[4]], []),
        (HGate(), [q[4]], []),
        (RC3XGate(), [q[0], q[1], q[2], q[3]], []),
        (HGate(), [q[4]], []),
        (CU1Gate(-np.pi / 2), [q[3], q[4]], []),
        (HGate(), [q[4]], []),
        (RC3XGate().inverse(), [q[0], q[1], q[2], q[3]], []),
        (C3SXGate(), [q[0], q[1], q[2], q[4]], []),
    ]
    for instr, qargs, cargs in rules:
        qc._append(instr, qargs, cargs)

    return qc
