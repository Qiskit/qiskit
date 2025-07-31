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
"""
Quantum Shannon Decomposition.

Method is described in arXiv:quant-ph/0406176.
"""
from __future__ import annotations
from typing import Callable
import scipy
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit, QuantumRegister
from qiskit.synthesis.two_qubit import (
    TwoQubitBasisDecomposer,
    two_qubit_decompose,
)
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.circuit.library.generalized_gates.uc_pauli_rot import UCPauliRotGate, _EPS
from qiskit._accelerate.two_qubit_decompose import two_qubit_decompose_up_to_diagonal

# pylint: disable=invalid-name


def qs_decomposition(
    mat: np.ndarray,
    opt_a1: bool | None = None,
    opt_a2: bool | None = None,
    decomposer_1q: Callable[[np.ndarray], QuantumCircuit] | None = None,
    decomposer_2q: Callable[[np.ndarray], QuantumCircuit] | None = None,
    *,
    _depth=0,
):
    r"""
    Decomposes a unitary matrix into one and two qubit gates using Quantum Shannon Decomposition,
    based on the Block ZXZ-Decomposition.

    This decomposition is described in Krol and Al-Ars [2] and improves the method of
    Shende et al. [1].

    .. code-block:: text

          ┌───┐              ┌───┐     ┌───┐
         ─┤   ├─      ────□──┤ H ├──□──┤ H ├──□──
          │   │    ≃    ┌─┴─┐└───┘┌─┴─┐└───┘┌─┴─┐
        /─┤   ├─      ──┤ C ├─────┤ B ├─────┤ A ├
          └───┘         └───┘     └───┘     └───┘

    The number of :class:`.CXGate`\ s generated with the decomposition without optimizations is
    the same as the unoptimized method in [1]:

    .. math::

        \frac{9}{16} 4^n - \frac{3}{2} 2^n

    If ``opt_a1 = True``, the CX count is reduced, improving [1], by:

    .. math::

        \frac{2}{3} (4^{n - 2} - 1).

    Saving two :class:`.CXGate`\ s instead of one in each step of the recursion.

    If ``opt_a2 = True``, the CX count is reduced, as in [1], by:

    .. math::

        4^{n-2} - 1.

    Hence, the number of :class:`.CXGate`\ s generated with the decomposition with optimizations is

    .. math::

        \frac{22}{48} 4^n - \frac{3}{2} 2^n + \frac{5}{3}.

    Args:
        mat: unitary matrix to decompose
        opt_a1: whether to try optimization A.1 from [1, 2].
            This should eliminate 2 ``cx`` per call.
        opt_a2: whether to try optimization A.2 from [1, 2].
            This decomposes two qubit unitaries into a diagonal gate and
            a two ``cx`` unitary and reduces overall ``cx`` count by :math:`4^{n-2} - 1`.
            This optimization should not be done if the original unitary is controlled.
        decomposer_1q: optional 1Q decomposer. If None, uses
            :class:`~qiskit.synthesis.OneQubitEulerDecomposer`.
        decomposer_2q: optional 2Q decomposer. If None, uses
            :class:`~qiskit.synthesis.TwoQubitBasisDecomposer`.

    Returns:
        QuantumCircuit: Decomposed quantum circuit.

    References:
        1. Shende, Bullock, Markov, *Synthesis of Quantum Logic Circuits*,
           `arXiv:0406176 [quant-ph] <https://arxiv.org/abs/quant-ph/0406176>`_
        2. Krol, Al-Ars, *Beyond Quantum Shannon: Circuit Construction for General
           n-Qubit Gates Based on Block ZXZ-Decomposition*,
           `arXiv:2403.13692 <https://arxiv.org/abs/2403.13692>`_
    """
    #  _depth (int): Internal use parameter to track recursion depth.
    dim = mat.shape[0]
    nqubits = dim.bit_length() - 1
    if opt_a1 is None:
        opt_a1 = True

    if np.allclose(np.identity(dim), mat):
        return QuantumCircuit(nqubits)
    # One-qubit unitary
    if dim == 2:
        if decomposer_1q is None:
            decomposer_1q = one_qubit_decompose.OneQubitEulerDecomposer()
        circ = decomposer_1q(mat)
        return circ
    # Two-qubit unitary
    if dim == 4:
        if decomposer_2q is None:
            if opt_a2 and _depth > 0:
                from qiskit.circuit.library.generalized_gates.unitary import (
                    UnitaryGate,
                )  # pylint: disable=cyclic-import

                def decomp_2q(mat):
                    ugate = UnitaryGate(mat)
                    qc = QuantumCircuit(2, name="qsd2q")
                    qc.append(ugate, [0, 1])
                    return qc

                decomposer_2q = decomp_2q
            else:
                decomposer_2q = TwoQubitBasisDecomposer(CXGate())
        mat = _closest_unitary(mat)
        circ = decomposer_2q(mat)
        return circ
    # check whether the matrix is equivalent to a block diagonal wrt ctrl_index
    if opt_a2 is None:
        opt_a2 = True
        # check if the unitary is controlled
        for ctrl_index in range(nqubits):
            um00, um11, um01, um10 = _extract_multiplex_blocks(mat, ctrl_index)
            if _off_diagonals_are_zero(um01, um10):
                opt_a2 = False
    if opt_a2 is False:
        for ctrl_index in range(nqubits):
            um00, um11, um01, um10 = _extract_multiplex_blocks(mat, ctrl_index)
            # the ctrl_index is reversed here
            if _off_diagonals_are_zero(um01, um10):
                decirc, _, _ = _demultiplex(
                    um00,
                    um11,
                    opt_a1=opt_a1,
                    opt_a2=opt_a2,
                    _vw_type="all",
                    _depth=_depth,
                    _ctrl_index=nqubits - 1 - ctrl_index,
                )
                return decirc
    qr = QuantumRegister(nqubits)

    # perform block ZXZ decomposition from [2]
    A1, A2, B, C = _block_zxz_decomp(np.asarray(mat, dtype=complex))
    iden = np.eye(2 ** (nqubits - 1))
    # left circ
    left_circ, vmatC, _wmatC = _demultiplex(
        iden, C, opt_a1=opt_a1, opt_a2=opt_a2, _vw_type="only_w", _depth=_depth
    )
    # right circ
    right_circ, _vmatA, wmatA = _demultiplex(
        A1, A2, opt_a1=opt_a1, opt_a2=opt_a2, _vw_type="only_v", _depth=_depth
    )

    # middle circ
    # zmat is needed in order to reduce two cz gates, and combine them into the B2 matrix
    zmat = np.diag([1] * (dim // 4) + [-1] * (dim // 4))
    # wmatA and vmatC are combined into B1 and B2
    B1 = wmatA @ vmatC
    if opt_a1:
        B2 = zmat @ wmatA @ B @ vmatC @ zmat
    else:
        B2 = wmatA @ B @ vmatC
    middle_circ, _, _ = _demultiplex(
        B1, B2, opt_a1=opt_a1, opt_a2=opt_a2, _vw_type="all", _depth=_depth
    )

    # the output circuit of the block ZXZ decomposition from [2]
    circ = QuantumCircuit(qr)
    circ.append(left_circ.to_instruction(), qr)
    circ.h(nqubits - 1)
    circ.append(middle_circ.to_instruction(), qr)
    circ.h(nqubits - 1)
    circ.append(right_circ.to_instruction(), qr)

    if opt_a2 and _depth == 0 and dim > 4:
        return _apply_a2(circ)
    return circ


def _block_zxz_decomp(Umat):
    """Block ZXZ decomposition method, by Krol and Al-Ars [2]."""
    dim = Umat.shape[0]
    n = dim // 2
    # from now on we keep the notations of [2]
    X = Umat[:n, :n]
    Y = Umat[:n, n:]
    U21 = Umat[n:, :n]
    U22 = Umat[n:, n:]

    VX, S, WXdg = scipy.linalg.svd(X)
    Sigma = np.diag(S)
    VXdg = VX.conj().T
    SX = VX @ Sigma @ VXdg
    UX = VX @ WXdg
    VY, S, WYdg = scipy.linalg.svd(Y)
    Sigma = np.diag(S)
    VYdg = VY.conj().T
    SY = VY @ Sigma @ VYdg
    UY = VY @ WYdg
    UYdg = UY.conj().T
    Cdg = 1j * UYdg @ UX
    C = Cdg.conj().T
    A1 = (SX + 1j * SY) @ UX
    A1dg = A1.conj().T
    A2 = U21 + U22 @ (1j * (UYdg @ UX))
    B = 2 * (A1dg @ X) - np.eye(n)
    return A1, A2, B, C


def _closest_unitary(mat):
    """Find the closest unitary matrix to a matrix mat."""

    V, _S, Wdg = scipy.linalg.svd(mat)
    mat = V @ Wdg
    return mat


def _demultiplex(
    um0, um1, opt_a1=False, opt_a2=False, *, _vw_type="all", _depth=0, _ctrl_index=None
):
    """Decompose a generic multiplexer.

          ────□────
           ┌──┴──┐
         /─┤     ├─
           └─────┘

    represented by the block diagonal matrix

            ┏         ┓
            ┃ um0     ┃
            ┃     um1 ┃
            ┗         ┛

    to
               ┌───┐
        ───────┤ Rz├──────
          ┌───┐└─┬─┘┌───┐
        /─┤ w ├──□──┤ v ├─
          └───┘     └───┘

    where v and w are general unitaries determined from decomposition.

    Args:
       um0 (ndarray): applied if MSB is 0
       um1 (ndarray): applied if MSB is 1
       opt_a1 (bool): whether to try optimization A.1 from [1, 2]. This should eliminate
          two ``cx`` gates per call.
       opt_a2 (bool): whether to try  optimization A.2 from [1, 2]. This decomposes two qubit
          unitaries into a diagonal gate and a two cx unitary and reduces overall ``cx`` count by
          4^(n-2) - 1. This optimization should not be done if the original unitary is controlled.
       _vw_type (string): "only_v", "only_w" or "all" for reductions.
          This is needed in order to combine the vmat or wmat into the B matrix in [2],
          instead of decomposing them.
       _depth (int): This is an internal variable to track the recursion depth.
       _ctrl_index (int): The index wrt which um0 and um1 are controlled.

    Returns:
        QuantumCircuit: decomposed circuit
    """
    dim = um0.shape[0] + um1.shape[0]  # these should be same dimension
    nqubits = dim.bit_length() - 1
    if _ctrl_index is None:
        _ctrl_index = nqubits - 1
    layout = list(range(0, _ctrl_index)) + list(range(_ctrl_index + 1, nqubits)) + [_ctrl_index]

    um0um1 = um0 @ um1.T.conjugate()
    if is_hermitian_matrix(um0um1):
        eigvals, vmat = scipy.linalg.eigh(um0um1)
    else:
        evals, vmat = scipy.linalg.schur(um0um1, output="complex")
        eigvals = evals.diagonal()
    dvals = np.emath.sqrt(eigvals)
    dmat = np.diag(dvals)
    wmat = dmat @ vmat.T.conjugate() @ um1

    circ = QuantumCircuit(nqubits)

    # left gate. In this case we decompose wmat.
    # Otherwise, it is combined with the B matrix.
    if _vw_type in ["only_w", "all"]:
        left_gate = qs_decomposition(
            wmat, opt_a1=opt_a1, opt_a2=opt_a2, _depth=_depth + 1
        ).to_instruction()
        circ.append(left_gate, layout[: nqubits - 1])

    # multiplexed Rz gate
    # If opt_a1 = ``True``, then we reduce 2 ``cx`` gates per call.
    angles = 2 * np.angle(np.conj(dvals))
    if _vw_type == "only_w" and opt_a1:
        ucrz = _get_ucrz(nqubits, angles)
    elif _vw_type == "only_v" and opt_a1:
        ucrz = _get_ucrz(nqubits, angles).reverse_ops()
    else:
        ucrz = _get_ucrz(nqubits, angles, _vw_type="all")
    circ.append(ucrz, [layout[-1]] + layout[: nqubits - 1])

    # right gate. In this case we decompose vmat.
    # Otherwise, it is combined with the B matrix.
    if _vw_type in ["only_v", "all"]:
        right_gate = qs_decomposition(
            vmat, opt_a1=opt_a1, opt_a2=opt_a2, _depth=_depth + 1
        ).to_instruction()
        circ.append(right_gate, layout[: nqubits - 1])

    return circ, vmat, wmat


def _get_ucrz(nqubits, angles, _vw_type=None):
    """This function synthesizes UCRZ without the final CX gate,
    unless _vw_type = ``all``."""
    circuit = QuantumCircuit(nqubits)
    q_controls = circuit.qubits[1:]
    q_target = circuit.qubits[0]

    UCPauliRotGate._dec_uc_rotations(angles, 0, len(angles), False)
    for i, angle in enumerate(angles):
        if np.abs(angle) > _EPS:
            circuit.rz(angle, q_target)
        if not i == len(angles) - 1:
            binary_rep = np.binary_repr(i + 1)
            q_contr_index = len(binary_rep) - len(binary_rep.rstrip("0"))
            circuit.cx(q_controls[q_contr_index], q_target)
        elif _vw_type == "all":
            q_contr_index = len(q_controls) - 1
            circuit.cx(q_controls[q_contr_index], q_target)

    return circuit


def _apply_a2(circ):
    """The optimization A.2 from [1, 2]. This decomposes two qubit unitaries into a
    diagonal gate and a two cx unitary and reduces overall ``cx`` count by
    4^(n-2) - 1. This optimization should not be done if the original unitary is controlled.
    """
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
    from qiskit.transpiler.passes.synthesis import HighLevelSynthesis

    decomposer = two_qubit_decompose_up_to_diagonal
    hls = HighLevelSynthesis(basis_gates=["u", "cx", "qsd2q"], qubits_initially_zero=False)
    ccirc = hls(circ)
    ind2q = []
    # collect 2q instrs
    for i, instruction in enumerate(ccirc.data):
        if instruction.operation.name == "qsd2q":
            ind2q.append(i)
    if len(ind2q) == 0:
        return ccirc
    elif len(ind2q) == 1:
        # No neighbors to merge diagonal into; revert name
        ccirc.data[ind2q[0]].operation.name = "Unitary"
        return ccirc
    # rolling over diagonals
    ind2 = None  # lint
    for ind1, ind2 in zip(ind2q[0:-1:], ind2q[1::]):
        # get neighboring 2q gates separated by controls
        instr1 = ccirc.data[ind1]
        mat1 = Operator(instr1.operation).data
        instr2 = ccirc.data[ind2]
        mat2 = Operator(instr2.operation).data
        # rollover
        dmat, qc2cx_data = decomposer(mat1)
        qc2cx = QuantumCircuit._from_circuit_data(qc2cx_data)
        ccirc.data[ind1] = instr1.replace(operation=qc2cx.to_gate())
        mat2 = mat2 @ dmat
        ccirc.data[ind2] = instr2.replace(UnitaryGate(mat2))
    qc3 = two_qubit_decompose.two_qubit_cnot_decompose(mat2)
    ccirc.data[ind2] = ccirc.data[ind2].replace(operation=qc3.to_gate())
    return ccirc


def _extract_multiplex_blocks(umat, k):
    """
    A block diagonal gate is represented as:
    [ um00 | um01 ]
    [ ---- | ---- ]
    [ um10 | um11 ]

    Args:
       umat (ndarray): unitary matrix
       k (integer): qubit which indicates the ctrl index

    Returns:
       um00 (ndarray): upper left block
       um01 (ndarray): upper right block
       um10 (ndarray): lower left block
       um11 (ndarray): lower right block
    """
    dim = umat.shape[0]
    nqubits = dim.bit_length() - 1
    halfdim = dim // 2

    utensor = umat.reshape((2,) * (2 * nqubits))

    # Move qubit k to top
    if k != 0:
        utensor = np.moveaxis(utensor, k, 0)
        utensor = np.moveaxis(utensor, k + nqubits, nqubits)

    # reshape for extraction
    ud4 = utensor.reshape(2, halfdim, 2, halfdim)
    # block for qubit k = |0>
    um00 = ud4[0, :, 0, :]
    # block for qubit k = |1>
    um11 = ud4[1, :, 1, :]
    # off diagonal blocks
    um01 = ud4[0, :, 1, :]
    um10 = ud4[1, :, 0, :]
    return um00, um11, um01, um10


def _off_diagonals_are_zero(um01, um10, atol=1e-12):
    """
    Checks whether off-diagonal blocks are zero.

    Args:
       um01 (ndarray): upper right block
       um10 (ndarray): lower left block
       atol (float): absolute tolerance

    Returns:
       bool: whether both blocks are zero within tolerance
    """
    return np.allclose(um01, 0, atol=atol) and np.allclose(um10, 0, atol=atol)
