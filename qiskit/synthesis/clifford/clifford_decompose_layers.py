# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for the Clifford class into layers.
"""
# pylint: disable=invalid-name

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.synthesis.linear import synth_cnot_count_full_pmh
from qiskit.synthesis.linear.linear_matrix_utils import (
    calc_inverse_matrix,
    check_invertible_binary_matrix,
    _compute_rank,
    _gauss_elimination,
    _gauss_elimination_with_perm,
)
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
    _append_h,
    _append_s,
    _append_cz,
)


def _default_cx_synth_func(mat, validate):
    """
    Construct the layer of CX gates from a boolean invertible matrix mat.
    """
    if validate:
        if not check_invertible_binary_matrix(mat):
            raise QiskitError("The matrix for CX circuit is not invertible.")

    CX_circ = synth_cnot_count_full_pmh(mat)
    CX_circ.name = "CX"

    if validate:
        _check_gates(CX_circ, ("cx", "swap"))

    return CX_circ


def _default_cz_synth_func(symmetric_mat, validate):
    """
    Construct the layer of CZ gates from a symmetric matrix.
    """
    nq = symmetric_mat.shape[0]
    qc = QuantumCircuit(nq, name="CZ")

    for j in range(nq):
        for i in range(0, j):
            if symmetric_mat[i][j]:
                qc.cz(i, j)

    if validate:
        _check_gates(qc, "cz")
    return qc


def synth_clifford_layers(
    cliff,
    cx_synth_func=_default_cx_synth_func,
    cz_synth_func=_default_cz_synth_func,
    cx_cz_synth_func=None,
    validate=False,
):
    """Synthesis of a Clifford into layers, it provides a similar decomposition to the synthesis
    described in Lemma 8 of [1].

    For example, a 5-qubit Clifford circuit is decomposed into the following layers:

    .. parsed-literal::
             ┌─────┐┌─────┐┌────────┐┌─────┐┌─────┐┌─────┐┌─────┐┌────────┐
        q_0: ┤0    ├┤0    ├┤0       ├┤0    ├┤0    ├┤0    ├┤0    ├┤0       ├
             │     ││     ││        ││     ││     ││     ││     ││        │
        q_1: ┤1    ├┤1    ├┤1       ├┤1    ├┤1    ├┤1    ├┤1    ├┤1       ├
             │     ││     ││        ││     ││     ││     ││     ││        │
        q_2: ┤2 S2 ├┤2 CZ ├┤2 CX_dg ├┤2 H2 ├┤2 S1 ├┤2 CZ ├┤2 H1 ├┤2 Pauli ├
             │     ││     ││        ││     ││     ││     ││     ││        │
        q_3: ┤3    ├┤3    ├┤3       ├┤3    ├┤3    ├┤3    ├┤3    ├┤3       ├
             │     ││     ││        ││     ││     ││     ││     ││        │
        q_4: ┤4    ├┤4    ├┤4       ├┤4    ├┤4    ├┤4    ├┤4    ├┤4       ├
             └─────┘└─────┘└────────┘└─────┘└─────┘└─────┘└─────┘└────────┘

    Args:
        cliff (Clifford): a clifford operator.
        cx_synth_func (Callable): a function to decompose the CX sub-circuit.
        cz_synth_func (Callable): a function to decompose the CZ sub-circuit.
        cx_cz_synth_func (Callable): optional, a function to decompose both sub-circuits CZ and CX.
        validate (Boolean): if True, validates the synthesis process.

    Return:
        QuantumCircuit: a circuit implementation of the Clifford.

    Reference:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
    """

    num_qubits = cliff.num_qubits

    qubit_list = list(range(num_qubits))
    layeredCircuit = QuantumCircuit(num_qubits)

    H1_circ, cliff1 = _create_graph_state(cliff, validate=validate)

    H2_circ, CZ1_circ, S1_circ, cliff2 = _decompose_graph_state(
        cliff1, validate=validate, cz_synth_func=cz_synth_func
    )

    S2_circ, CZ2_circ, CX_circ = _decompose_hadamard_free(
        cliff2.adjoint(),
        validate=validate,
        cz_synth_func=cz_synth_func,
        cx_synth_func=cx_synth_func,
        cx_cz_synth_func=cx_cz_synth_func,
    )

    layeredCircuit.append(S2_circ, qubit_list)
    layeredCircuit.append(CZ2_circ, qubit_list)

    CXinv = CX_circ.copy().inverse()
    layeredCircuit.append(CXinv, qubit_list)

    layeredCircuit.append(H2_circ, qubit_list)
    layeredCircuit.append(S1_circ, qubit_list)
    layeredCircuit.append(CZ1_circ, qubit_list)

    layeredCircuit.append(H1_circ, qubit_list)

    # Add Pauli layer to fix the Clifford phase signs
    # pylint: disable=cyclic-import
    from qiskit.quantum_info.operators.symplectic import Clifford

    clifford_target = Clifford(layeredCircuit)
    pauli_circ = _calc_pauli_diff(cliff, clifford_target)
    layeredCircuit.append(pauli_circ, qubit_list)

    return layeredCircuit


def _reverse_clifford(cliff):
    """Reverse qubit order of a Clifford cliff"""
    cliff_cpy = cliff.copy()
    cliff_cpy.stab_z = np.flip(cliff.stab_z, axis=1)
    cliff_cpy.destab_z = np.flip(cliff.destab_z, axis=1)
    cliff_cpy.stab_x = np.flip(cliff.stab_x, axis=1)
    cliff_cpy.destab_x = np.flip(cliff.destab_x, axis=1)
    return cliff_cpy


def _create_graph_state(cliff, validate=False):
    """Given a Clifford cliff (denoted by U) that induces a stabilizer state U |0>,
    apply a layer H1 of Hadamard gates to a subset of the qubits to make H1 U |0> into a graph state,
    namely to make cliff.stab_x matrix have full rank.
    Returns the QuantumCircuit H1_circ that includes the Hadamard gates and the updated Clifford
    that induces the graph state.
    The algorithm is based on Lemma 6 in [2].

    Args:
        cliff (Clifford): a clifford operator.
        validate (Boolean): if True, validates the synthesis process.

    Return:
        H1_circ: a circuit containing a layer of Hadamard gates.
        cliffh: cliffh.stab_x has full rank.

    Raises:
        QiskitError: if there are errors in the Gauss elimination process.

    Reference:
        2. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """

    num_qubits = cliff.num_qubits
    rank = _compute_rank(cliff.stab_x)
    H1_circ = QuantumCircuit(num_qubits, name="H1")
    cliffh = cliff.copy()

    if rank < num_qubits:
        stab = cliff.stab[:, :-1]
        stab = _gauss_elimination(stab, num_qubits)

        Cmat = stab[rank:num_qubits, num_qubits:]
        Cmat = np.transpose(Cmat)
        Cmat, perm = _gauss_elimination_with_perm(Cmat)
        perm = perm[0 : num_qubits - rank]

        # validate that the output matrix has the same rank
        if validate:
            if _compute_rank(Cmat) != num_qubits - rank:
                raise QiskitError("The matrix Cmat after Gauss elimination has wrong rank.")
            if _compute_rank(stab[:, 0:num_qubits]) != rank:
                raise QiskitError("The matrix after Gauss elimination has wrong rank.")
            # validate that we have a num_qubits - rank zero rows
            for i in range(rank, num_qubits):
                if stab[i, 0:num_qubits].any():
                    raise QiskitError(
                        "After Gauss elimination, the final num_qubits - rank rows"
                        "contain non-zero elements"
                    )

        for qubit in perm:
            H1_circ.h(qubit)
            _append_h(cliffh, qubit)

        # validate that a layer of Hadamard gates and then appending cliff, provides a graph state.
        if validate:
            stabh = cliffh.stab_x
            if _compute_rank(stabh) != num_qubits:
                raise QiskitError("The state is not a graph state.")

    return H1_circ, cliffh


def _decompose_graph_state(cliff, validate, cz_synth_func):
    """Assumes that a stabilizer state of the Clifford cliff (denoted by U) corresponds to a graph state.
    Decompose it into the layers S1 - CZ1 - H2, such that:
    S1 CZ1 H2 |0> = U |0>,
    where S1_circ is a circuit that can contain only S gates,
    CZ1_circ is a circuit that can contain only CZ gates, and
    H2_circ is a circuit that can contain H gates on all qubits.

    Args:
        cliff (Clifford): a clifford operator corresponding to a graph state, cliff.stab_x has full rank.
        validate (Boolean): if True, validates the synthesis process.
        cz_synth_func (Callable): a function to decompose the CZ sub-circuit.

    Return:
        S1_circ: a circuit that can contain only S gates.
        CZ1_circ: a circuit that can contain only CZ gates.
        H2_circ: a circuit containing a layer of Hadamard gates.
        cliff_cpy: a Hadamard-free Clifford.

    Raises:
        QiskitError: if cliff does not induce a graph state.
    """

    num_qubits = cliff.num_qubits
    rank = _compute_rank(cliff.stab_x)
    cliff_cpy = cliff.copy()
    if rank < num_qubits:
        raise QiskitError("The stabilizer state is not a graph state.")

    S1_circ = QuantumCircuit(num_qubits, name="S1")
    H2_circ = QuantumCircuit(num_qubits, name="H2")

    stabx = cliff.stab_x
    stabz = cliff.stab_z
    stabx_inv = calc_inverse_matrix(stabx, validate)
    stabz_update = np.matmul(stabx_inv, stabz) % 2

    # Assert that stabz_update is a symmetric matrix.
    if validate:
        if (stabz_update != stabz_update.T).any():
            raise QiskitError(
                "The multiplication of stabx_inv and stab_z is not a symmetric matrix."
            )

    CZ1_circ = cz_synth_func(stabz_update, validate=validate)

    for j in range(num_qubits):
        for i in range(0, j):
            if stabz_update[i][j]:
                _append_cz(cliff_cpy, i, j)

    for i in range(0, num_qubits):
        if stabz_update[i][i]:
            S1_circ.s(i)
            _append_s(cliff_cpy, i)

    for qubit in range(num_qubits):
        H2_circ.h(qubit)
        _append_h(cliff_cpy, qubit)

    return H2_circ, CZ1_circ, S1_circ, cliff_cpy


def _decompose_hadamard_free(cliff, validate, cz_synth_func, cx_synth_func, cx_cz_synth_func):
    """Assumes that the Clifford cliff is Hadamard free.
    Decompose it into the layers S2 - CZ2 - CX, where
    S2_circ is a circuit that can contain only S gates,
    CZ2_circ is a circuit that can contain only CZ gates, and
    CX_circ is a circuit that can contain CX gates.

    Args:
        cliff (Clifford): a Hadamard-free clifford operator.
        validate (Boolean): if True, validates the synthesis process.
        cz_synth_func (Callable): a function to decompose the CZ sub-circuit.
        cx_synth_func (Callable): a function to decompose the CX sub-circuit.
        cx_cz_synth_func (Callable): optional, a function to decompose both sub-circuits CZ and CX.

    Return:
        S2_circ: a circuit that can contain only S gates.
        CZ2_circ: a circuit that can contain only CZ gates.
        CX_circ: a circuit that can contain only CX gates.

    Raises:
        QiskitError: if cliff is not Hadamard free.
    """

    num_qubits = cliff.num_qubits
    destabx = cliff.destab_x
    destabz = cliff.destab_z
    stabx = cliff.stab_x

    if not (stabx == np.zeros((num_qubits, num_qubits))).all():
        raise QiskitError("The given Clifford is not Hadamard-free.")

    destabz_update = np.matmul(calc_inverse_matrix(destabx), destabz) % 2
    # Assert that destabz_update is a symmetric matrix.
    if validate:
        if (destabz_update != destabz_update.T).any():
            raise QiskitError(
                "The multiplication of the inverse of destabx and"
                "destabz is not a symmetric matrix."
            )

    S2_circ = QuantumCircuit(num_qubits, name="S2")
    for i in range(0, num_qubits):
        if destabz_update[i][i]:
            S2_circ.s(i)

    if cx_cz_synth_func is not None:
        CZ2_circ, CX_circ = cx_cz_synth_func(
            destabz_update, cliff.destab_x.transpose(), num_qubits=num_qubits
        )
        return S2_circ, CZ2_circ, CX_circ

    CZ2_circ = cz_synth_func(destabz_update, validate=validate)

    mat = destabx.transpose()
    CX_circ = cx_synth_func(mat, validate=validate)

    return S2_circ, CZ2_circ, CX_circ


def _calc_pauli_diff(cliff, cliff_target):
    """Given two Cliffords that differ by a Pauli, we find this Pauli."""

    num_qubits = cliff.num_qubits
    if cliff.num_qubits != cliff_target.num_qubits:
        raise QiskitError("num_qubits is not the same for the original clifford and the target.")

    # Compute the phase difference between the two Cliffords
    phase = [cliff.phase[k] ^ cliff_target.phase[k] for k in range(2 * num_qubits)]
    phase = np.array(phase, dtype=int)

    # compute inverse of our symplectic matrix
    A = cliff.symplectic_matrix
    Ainv = calc_inverse_matrix(A)

    # By carefully writing how X, Y, Z gates affect each qubit, all we need to compute
    # is A^{-1} * (phase)
    C = np.matmul(Ainv, phase) % 2

    # Create the Pauli
    pauli_circ = QuantumCircuit(num_qubits, name="Pauli")
    for k in range(num_qubits):
        destab = C[k]
        stab = C[k + num_qubits]
        if stab and destab:
            pauli_circ.y(k)
        elif stab:
            pauli_circ.x(k)
        elif destab:
            pauli_circ.z(k)

    return pauli_circ


def _check_gates(qc, allowed_gates):
    """Check that quantum circuit qc consists only of allowed_gates.
    qc - a QuantumCircuit
    allowed_gates - list of strings
    """
    for inst, _, _ in qc.data:
        if not inst.name in allowed_gates:
            raise QiskitError("The gate name is not in the allowed_gates list.")
