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
from typing import Optional
import warnings
import scipy
import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit, QuantumRegister
from qiskit.synthesis.one_qubit.one_qubit_decompose import OneQubitEulerDecomposer
from qiskit.synthesis.two_qubit import (
    TwoQubitBasisDecomposer,
    two_qubit_decompose,
)
from qiskit.synthesis.one_qubit import one_qubit_decompose
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.circuit.library.generalized_gates.uc_pauli_rot import UCPauliRotGate, _EPS
from qiskit._accelerate.two_qubit_decompose import two_qubit_decompose_up_to_diagonal
from qiskit._accelerate import qsd


# pylint: disable=invalid-name
# pylint: disable=too-many-return-statements
def qs_decomposition(
    mat: np.ndarray,
    opt_a1: Optional[bool] = None,
    opt_a2: Optional[bool] = None,
    decomposer_1q: Optional[OneQubitEulerDecomposer] = None,
    decomposer_2q: Optional[TwoQubitBasisDecomposer] = None,
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

    If ``opt_a2 = True``, the CX count is reduced, as in [1], by:

    .. math::

        4^{n-2} - 1.

    Hence, the number of :class:`.CXGate`\ s generated with the decomposition with optimizations is

    .. math::

        \frac{22}{48} 4^n - \frac{3}{2} 2^n + \frac{5}{3}.

    Args:
        mat: unitary matrix to decompose
        opt_a1: (Deprecated) whether to try optimization A.1.
        opt_a2: (Deprecated) whether to try optimization A.2.
        decomposer_1q: optional 1Q decomposer instance.
        decomposer_2q: optional 2Q decomposer instance.

    .. deprecated:: 2.3.0
        The parameters ``opt_a1`` and ``opt_a2`` are deprecated and will be removed in Qiskit 3.0.
        The decomposition will automatically determine optimization settings.
        Passing custom callables for ``decomposer_1q`` or ``decomposer_2q`` is also deprecated;
        only :class:`.OneQubitEulerDecomposer` and :class:`.TwoQubitBasisDecomposer` instances
        will be supported.

    Returns:
        QuantumCircuit: Decomposed quantum circuit.

    References:
        1. Shende, Bullock, Markov, *Synthesis of Quantum Logic Circuits*,
           `arXiv:0406176 [quant-ph] <https://arxiv.org/abs/quant-ph/0406176>`_
        2. Krol, Al-Ars, *Beyond Quantum Shannon: Circuit Construction for General
           n-Qubit Gates Based on Block ZXZ-Decomposition*,
           `arXiv:2403.13692 <https://arxiv.org/abs/2403.13692>`_
    """
    # ---------------------------------------------------------------------
    # Deprecation warnings (Qiskit 2.3.0 → removal in 3.0)
    # ---------------------------------------------------------------------
    if opt_a1 is not None or opt_a2 is not None:
        warnings.warn(
            "The 'opt_a1' and 'opt_a2' parameters are deprecated as of Qiskit 2.3.0 "
            "and will be removed in Qiskit 3.0. The decomposition will automatically "
            "select the appropriate optimizations.",
            DeprecationWarning,
            stacklevel=2,
        )

    if decomposer_1q is not None and not isinstance(decomposer_1q, OneQubitEulerDecomposer):
        warnings.warn(
            "Passing a custom callable for 'decomposer_1q' is deprecated as of Qiskit 2.3.0 "
            "and will be removed in Qiskit 3.0. Only instances of "
            "OneQubitEulerDecomposer will be supported.",
            DeprecationWarning,
            stacklevel=2,
        )

    if decomposer_2q is not None and not isinstance(decomposer_2q, TwoQubitBasisDecomposer):
        warnings.warn(
            "Passing a custom callable for 'decomposer_2q' is deprecated as of Qiskit 2.3.0 "
            "and will be removed in Qiskit 3.0. Only instances of "
            "TwoQubitBasisDecomposer will be supported.",
            DeprecationWarning,
            stacklevel=2,
        )

    # ---------------------------------------------------------------------
    # Main function logic
    # ---------------------------------------------------------------------
    if (decomposer_1q is None or isinstance(decomposer_1q, OneQubitEulerDecomposer)) and (
        decomposer_2q is None or isinstance(decomposer_2q, TwoQubitBasisDecomposer)
    ):
        basis_1q = None
        if decomposer_1q is not None:
            basis_1q = decomposer_1q.basis
        two_q_decomp = None
        if decomposer_2q is not None:
            two_q_decomp = decomposer_2q._inner_decomposer
        array = np.asarray(mat, dtype=complex)
        return QuantumCircuit._from_circuit_data(
            qsd.qs_decomposition(array, opt_a1, opt_a2, basis_1q, two_q_decomp)
        )

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
    """Decompose a generic multiplexer."""
    dim = um0.shape[0] + um1.shape[0]
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

    if _vw_type in ["only_w", "all"]:
        left_gate = qs_decomposition(
            wmat, opt_a1=opt_a1, opt_a2=opt_a2, _depth=_depth + 1
        ).to_instruction()
        circ.append(left_gate, layout[: nqubits - 1])

    angles = 2 * np.angle(np.conj(dvals))
    if _vw_type == "only_w" and opt_a1:
        ucrz = _get_ucrz(nqubits, angles)
    elif _vw_type == "only_v" and opt_a1:
        ucrz = _get_ucrz(nqubits, angles).reverse_ops()
    else:
        ucrz = _get_ucrz(nqubits, angles, _vw_type="all")
    circ.append(ucrz, [layout[-1]] + layout[: nqubits - 1])

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
    """Optimization A.2 from [1, 2]."""
    from qiskit.quantum_info import Operator
    from qiskit.circuit.library.generalized_gates.unitary import UnitaryGate
    from qiskit.transpiler.passes.synthesis import HighLevelSynthesis

    decomposer = two_qubit_decompose_up_to_diagonal
    hls = HighLevelSynthesis(basis_gates=["u", "cx", "qsd2q"], qubits_initially_zero=False)
    ccirc = hls(circ)
    ind2q = []
    for i, instruction in enumerate(ccirc.data):
        if isinstance(instruction.operation, UnitaryGate):
            ind2q.append(i)
    i0 = ind2q[0]
    i1 = ind2q[1]
    qubits = ccirc.data[i0].qubits
    mat = Operator(ccirc.data[i0].operation).data
    mat2 = Operator(ccirc.data[i1].operation).data
    circ2 = decomposer(mat, mat2)
    ccirc.data.pop(i1)
    ccirc.data.pop(i0)
    ccirc.append(circ2.to_instruction(), qubits)
    return ccirc


def _extract_multiplex_blocks(mat, ctrl_index):
    """Return the 4 blocks of a unitary, assuming the ctrl_index is the control qubit."""
    dim = mat.shape[0]
    nqubits = dim.bit_length() - 1
    q_n = nqubits - 1 - ctrl_index
    indices = np.arange(0, dim, dtype=int)
    base = 2**q_n
    step = base * 2
    mask = (indices // base) % 2 == 1
    odd = np.where(mask)[0]
    even = np.where(mask == 0)[0]

    um00 = mat[np.ix_(even, even)]
    um01 = mat[np.ix_(even, odd)]
    um10 = mat[np.ix_(odd, even)]
    um11 = mat[np.ix_(odd, odd)]

    return um00, um11, um01, um10


def _off_diagonals_are_zero(um01, um10, tol=1e-10):
    """Check whether the off-diagonal blocks are close to zero."""
    return np.allclose(um01, np.zeros_like(um01), atol=tol) and np.allclose(
        um10, np.zeros_like(um10), atol=tol
    )
