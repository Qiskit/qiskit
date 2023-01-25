# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Optimize the synthesis of an n-qubit circuit contains only CZ gates for
linear nearest neighbor (LNN) connectivity, using CX and phase (S, Sdg or Z) gates.
The 2-qubit depth of the circuit is bounded by 2*n+2.

References:
    [1]: Dmitri Maslov, Martin Roetteler,
         Shorter stabilizer circuits via Bruhat decomposition and quantum circuit transformations,
         `arXiv:1705.09176 <https://arxiv.org/abs/1705.09176>`_.
"""

from qiskit.circuit import QuantumCircuit


def _append_cx_stage1(qc, n):
    for i in range(int(n / 2)):
        qc.cx(2 * i, 2 * i + 1)
    for i in range(int((n + 1) / 2) - 1):
        qc.cx(2 * i + 2, 2 * i + 1)
    return qc


def _append_cx_stage2(qc, n):
    for i in range(int(n / 2)):
        qc.cx(2 * i + 1, 2 * i)
    for i in range(int((n + 1) / 2) - 1):
        qc.cx(2 * i + 1, 2 * i + 2)
    return qc


def _append_cx_stage(qc, n):
    """Append a depth 2 layer of CX gates"""
    qc.compose(_append_cx_stage1(qc, n))
    qc.compose(_append_cx_stage2(qc, n))
    return qc


def _odd_pattern1(n):
    pat = []
    pat.append(n - 2)
    for i in range(int((n - 3) / 2)):
        pat.append(n - 2 * i - 4)
        pat.append(n - 2 * i - 4)
    for i in range(int((n - 1) / 2)):
        pat.append(2 * i)
        pat.append(2 * i)
    return pat


def _odd_pattern2(n):
    pat = []
    for i in range(int((n - 1) / 2)):
        pat.append(2 * i + 2)
        pat.append(2 * i + 2)
    for i in range(int((n - 3) / 2)):
        pat.append(n - 2 * i - 2)
        pat.append(n - 2 * i - 2)
    pat.append(1)
    return pat


def _even_pattern1(n):
    pat = []
    pat.append(n - 1)
    for i in range(int((n - 2) / 2)):
        pat.append(n - 2 * i - 3)
        pat.append(n - 2 * i - 3)
    for i in range(int((n - 2) / 2)):
        pat.append(2 * i)
        pat.append(2 * i)
    pat.append(n - 2)
    return pat


def _even_pattern2(n):
    pat = []
    for i in range(int((n - 2) / 2)):
        pat.append(2 * (i + 1))
        pat.append(2 * (i + 1))
    for i in range(int(n / 2)):
        pat.append(n - 2 * i - 1)
        pat.append(n - 2 * i - 1)
    return pat


def _create_patterns(n):
    """Creating the patterns for the phase layer."""
    if (n % 2) == 0:
        pat1 = _even_pattern1(n)
        pat2 = _even_pattern2(n)
    else:
        pat1 = _odd_pattern1(n)
        pat2 = _odd_pattern2(n)
    pats = {}

    layer = 0
    for i in range(n):
        pats[(0, i)] = (i, i)

    if (n % 2) == 0:
        ind1 = int((2 * n - 4) / 2)
    else:
        ind1 = int((2 * n - 4) / 2 - 1)
    ind2 = 0
    while layer < int(n / 2):
        for i in range(n):
            pats[(layer + 1, i)] = (pat1[ind1 + i], pat2[ind2 + i])
        layer += 1
        ind1 -= 2
        ind2 += 2
    return pats


def synth_cz_depth_line_mr(mat):
    """Synthesis of a CZ circuit"""
    # A CZ circuit is represented by an upper-diagonal matrix mat:
    # mat[i][j]=1 for i<j represents a CZ(i,j) gate
    # The function return an n-qubits circuit of depth 2*n with LNN connectivity
    num_qubits = mat.shape[0]
    pats = _create_patterns(num_qubits)
    patlist = []

    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        for j in range(i + 1, num_qubits):
            if mat[i][j]:  # CZ(i,j) gate
                qc.z(i)
                qc.z(j)
                patlist.append((i, j - 1))
                patlist.append((i, j))
                patlist.append((i + 1, j - 1))
                patlist.append((i + 1, j))

    for i in range(int((num_qubits + 1) / 2)):
        for j in range(num_qubits):
            if pats[(i, j)] in patlist:
                patcnt = patlist.count(pats[(i, j)])
                for _ in range(patcnt):
                    qc.sdg(j)
        qc = _append_cx_stage(qc, num_qubits)

    if (num_qubits % 2) == 0:
        i = int(num_qubits / 2)
        for j in range(num_qubits):
            if pats[(i, j)] in patlist and pats[(i, j)][0] != pats[(i, j)][1]:
                patcnt = patlist.count(pats[(i, j)])
                for _ in range(patcnt):
                    qc.sdg(j)
        qc = _append_cx_stage1(qc, num_qubits)

    return qc
