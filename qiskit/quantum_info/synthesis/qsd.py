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
import scipy
import numpy as np
from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info.synthesis import two_qubit_decompose, one_qubit_decompose
from qiskit.circuit.library.standard_gates import CXGate
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.extensions.quantum_initializer.uc_pauli_rot import UCPauliRotGate, _EPS


def qs_decomposition(mat, opt_a1=True, decomposer_1q=None, decomposer_2q=None):
    """
    Decomposes unitary matrix into one and two qubit gates using Quantum Shannon Decomposition.

       ┌───┐               ┌───┐     ┌───┐     ┌───┐
      ─┤   ├─       ───────┤ Rz├─────┤ Ry├─────┤ Rz├─────
       │   │    ≃     ┌───┐└─┬─┘┌───┐└─┬─┘┌───┐└─┬─┘┌───┐
     /─┤   ├─       /─┤   ├──□──┤   ├──□──┤   ├──□──┤   ├
       └───┘          └───┘     └───┘     └───┘     └───┘

    The number of CX gates generated with the decomposition without optimizations is,

    .. math::

        \frac{9}{16} 4^n - frac{3}{2} 2^n

    If opt_a1=True, the CX count is further reduced by,

    .. math::

        \frac{1}{3} 4^{n - 2} - 1

    This decomposition is described in arXiv:quant-ph/0406176.

    Arguments:
       mat (ndarray): unitary matrix to decompose
       opt_a1 (bool): whether to try optimization A.1 from Shende. This should eliminate 1 cnot
          per call. If True CZ gates are left in the output. If desired these can be further decomposed
          to CX.
       decomposer_1q (None or Object): optional 1Q decomposer. If None, uses
          :class:`~qiskit.quantum_info.synthesis.one_qubit_decomposer.OneQubitEulerDecomser`
       decomposer_2q (None or Object): optional 2Q decomposer. If None, uses
          :class:`~qiskit.quantum_info.synthesis.two_qubit_decomposer.TwoQubitBasisDecomposer`
          with CXGate.

    Return:
       QuantumCircuit: Decomposed quantum circuit.
    """
    dim = mat.shape[0]
    nqubits = int(np.log2(dim))
    if np.allclose(np.identity(dim), mat):
        return QuantumCircuit(nqubits)
    if dim == 2:
        if decomposer_1q is None:
            decomposer_1q = one_qubit_decompose.OneQubitEulerDecomposer()
        circ = decomposer_1q(mat)
    elif dim == 4:
        if decomposer_2q is None:
            decomposer_2q = two_qubit_decompose.TwoQubitBasisDecomposer(CXGate())
        circ = decomposer_2q(mat)
    else:
        qr = QuantumRegister(nqubits)
        circ = QuantumCircuit(qr)
        dim_o2 = dim // 2
        # perform cosine-sine decomposition
        (u1, u2), vtheta, (v1h, v2h) = scipy.linalg.cossin(mat, separate=True, p=dim_o2, q=dim_o2)
        # left circ
        left_circ = _demultiplex(v1h, v2h, opt_a1=opt_a1)
        circ.append(left_circ.to_instruction(), qr)
        # middle circ
        if opt_a1:
            nangles = len(vtheta)
            half_size = nangles // 2
            # get UCG in terms of CZ
            circ_cz = _get_ucry_cz(nqubits, (2 * vtheta).tolist())
            circ.append(circ_cz.to_instruction(), range(nqubits))
            # merge final cz with right-side generic multiplexer
            u2[:, half_size:] = np.negative(u2[:, half_size:])
        else:
            circ.ucry((2 * vtheta).tolist(), qr[:-1], qr[-1])
        # right circ
        right_circ = _demultiplex(u1, u2, opt_a1=opt_a1)
        circ.append(right_circ.to_instruction(), qr)

    return circ


def _demultiplex(um0, um1, opt_a1=False):
    """decomposes a generic multiplexer.

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
       opt_a1 (bool): whether to try optimization A.1 from Shende. This should elliminate 1 cnot
          per call. If True CZ gates are left in the output. If desired these can be further decomposed

    Returns:
        QuantumCircuit: decomposed circuit
    """
    dim = um0.shape[0] + um1.shape[0]  # these should be same dimension
    nqubits = int(np.log2(dim))
    um0um1 = um0 @ um1.T.conjugate()
    if is_hermitian_matrix(um0um1):
        eigvals, vmat = scipy.linalg.eigh(um0um1)
    else:
        evals, vmat = scipy.linalg.schur(um0um1, output="complex")
        eigvals = evals.diagonal()
    dvals = np.lib.scimath.sqrt(eigvals)
    dmat = np.diag(dvals)
    wmat = dmat @ vmat.T.conjugate() @ um1

    circ = QuantumCircuit(nqubits)

    # left gate
    left_gate = qs_decomposition(wmat, opt_a1=opt_a1).to_instruction()
    circ.append(left_gate, range(nqubits - 1))

    # multiplexed Rz
    angles = 2 * np.angle(np.conj(dvals))
    circ.ucrz(angles.tolist(), list(range(nqubits - 1)), [nqubits - 1])

    # right gate
    right_gate = qs_decomposition(vmat, opt_a1=opt_a1).to_instruction()
    circ.append(right_gate, range(nqubits - 1))

    return circ


def _get_ucry_cz(nqubits, angles):
    """
    Get uniformally controlled Ry gate in in CZ-Ry as in UCPauliRotGate.
    """
    nangles = len(angles)
    qc = QuantumCircuit(nqubits)
    q_controls = qc.qubits[:-1]
    q_target = qc.qubits[-1]
    if not q_controls:
        if np.abs(angles[0]) > _EPS:
            qc.ry(angles[0], q_target)
    else:
        angles = angles.copy()
        UCPauliRotGate._dec_uc_rotations(angles, 0, len(angles), False)
        for (i, angle) in enumerate(angles):
            if np.abs(angle) > _EPS:
                qc.ry(angle, q_target)
            if not i == len(angles) - 1:
                binary_rep = np.binary_repr(i + 1)
                q_contr_index = len(binary_rep) - len(binary_rep.rstrip("0"))
            else:
                # Handle special case:
                q_contr_index = len(q_controls) - 1
            # leave off last CZ for merging with adjacent UCG
            if i < nangles - 1:
                qc.cz(q_controls[q_contr_index], q_target)
    return qc
