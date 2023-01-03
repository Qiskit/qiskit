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
    _compute_rank,
    _gauss_elimination,
    _gauss_elimination_with_perm,
)
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
    _append_h,
    _append_s,
    _append_cz,
)


class LayeredCircuit:
    """Stores layered decomposition of the QuantumCircuit
    layers is a list of QuantumCircuits
    """

    def __init__(self, num_qubits: int):
        """Stores layered decomposition of the QuantumCircuit

        Args:
            num_qubits: number of qubits of the QuantumCircuit.
        """
        self.num_qubits = num_qubits
        self.layers = []

    def append_layer(self, qc, qc_name=None):
        """Appends a new layer.
        qc is a quantum circuit
        qc_name is the optional label to assign to qc
        """
        if not isinstance(qc, QuantumCircuit):
            raise QiskitError("qc is not a QuantumCircuit.")
        if qc.num_qubits != self.num_qubits:
            raise QiskitError("num_qubits is not the same as qc.num_qubits.")

        if qc_name is not None:
            qc.name = qc_name
        self.layers.append(qc)

    def create_circuit(self):
        """Returns circuit obtained by appending all layers"""
        circ = QuantumCircuit(self.num_qubits)
        for _, qc in enumerate(list(self.layers)):
            circ.append(qc, list(range(self.num_qubits)))
        return circ

    def draw(self):
        """Print the layered circuit."""
        print(self.create_circuit())

    def draw_detailed(self):
        """Print the circuit layer-by-layer"""
        print()
        for i, qc in enumerate(list(self.layers)):
            print(f"Printing layer {i}")
            print(qc)
            print("")


def synth_clifford_layers(
    cliff,
    cx_synth_func,
    cz_synth_func,
    cx_cz_synth_func=None,
    reverse_cliff=False,
    validate=False,
):
    """Synthesis of a Clifford into layers, it provides a similar decomposition to the synthesis
    described in Lemma 8 of [1].

    Args:
        cliff (Clifford): a clifford operator.
        cx_synth_func (Callable): a function to decompose the CX sub-circuit.
        cz_synth_func (Callable): a function to decompose the CZ sub-circuit.
        cx_cz_synth_func (Callable): optional, a function to decompose the sub-circuits CZ and CX .
        reverse_cliff (Boolean): if True, reverse the order of the qubits of the Clifford.
        validate (Boolean): if True, validates the synthesis process.

    Return:
        QuantumCircuit: a circuit implementation of the Clifford.

    Reference:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
    """

    cliff_cpy = cliff.copy()
    num_qubits = cliff.num_qubits

    if reverse_cliff:
        # Reverse the order of qubits
        cliff_cpy = _reverse_clifford(cliff)

    layeredCircuit = LayeredCircuit(num_qubits)

    H1_circ, cliff1 = _create_graph_state(cliff_cpy, validate=validate)

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

    layeredCircuit.append_layer(S2_circ, "S2")
    layeredCircuit.append_layer(CZ2_circ, "CZ2")

    CXinv = CX_circ.copy().inverse()
    layeredCircuit.append_layer(CXinv, "CX")

    layeredCircuit.append_layer(H2_circ, "H2")
    layeredCircuit.append_layer(S1_circ, "S1")
    layeredCircuit.append_layer(CZ1_circ, "CZ1")

    layeredCircuit.append_layer(H1_circ, "H1")

    # Add Pauli layer to fix the Clifford phase signs
    # pylint: disable=cyclic-import
    from qiskit.quantum_info.operators.symplectic import Clifford

    clifford_target = Clifford(layeredCircuit.create_circuit())
    pauli_circ = _fix_pauli(cliff, clifford_target)
    layeredCircuit.append_layer(pauli_circ, "Pauli")

    return layeredCircuit.create_circuit()


def _reverse_clifford(cliff):
    """Reverse qubit order of a Clifford cliff"""
    cliff_cpy = cliff.copy()
    cliff_cpy.stab_z = np.flip(cliff.stab_z, axis=1)
    cliff_cpy.destab_z = np.flip(cliff.destab_z, axis=1)
    cliff_cpy.stab_x = np.flip(cliff.stab_x, axis=1)
    cliff_cpy.destab_x = np.flip(cliff.destab_x, axis=1)
    return cliff_cpy


def _create_graph_state(cliff, validate=False):
    """Apply Hadamard gates to a subset of the qubits to make cliff.stab_x matrix have full rank.
    Returns the QuantumCircuit H1_circ that includes the Hadamard gates.
    The algorithm is based on Lemma 6 in [2]."""

    num_qubits = cliff.num_qubits
    rank = _compute_rank(cliff.stab_x)
    H1_circ = QuantumCircuit(num_qubits)
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
    """
    num_qubits = cliff.num_qubits
    rank = _compute_rank(cliff.stab_x)
    cliff_cpy = cliff.copy()
    if rank < num_qubits:
        raise QiskitError("The stabilizer state is not a graph state.")

    S1_circ = QuantumCircuit(num_qubits)
    H2_circ = QuantumCircuit(num_qubits)

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
    CX_circ is a circuit that can contain CX gates on all qubits.
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

    S2_circ = QuantumCircuit(num_qubits)
    for i in range(0, num_qubits):
        if destabz_update[i][i]:
            S2_circ.s(i)

    if cx_cz_synth_func is not None:
        CZ2_circ, CX_circ = cx_cz_synth_func(
            destabz_update, cliff.destab_x.transpose(), num_qubits=num_qubits
        )
        return S2_circ, CZ2_circ, CX_circ

    CZ2_circ = cz_synth_func(destabz_update, validate=validate)

    cliff_cpy = cliff.copy()
    cliff_cpy.destab_z = np.zeros((num_qubits, num_qubits))
    cliff_cpy.phase = np.zeros(2 * num_qubits)
    CX_circ = cx_synth_func(cliff_cpy, validate=validate)

    return S2_circ, CZ2_circ, CX_circ


def _fix_pauli(cliff, cliff_target):
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
    pauli_circ = QuantumCircuit(num_qubits)
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


def _default_cx_synth_func(cliff, validate):
    """
    Assume that the Clifford is of the form [A, 0; 0, D].
    Construct the layer of CX gates for this Clifford."""

    from qiskit.quantum_info.operators.symplectic import Clifford

    if not isinstance(cliff, Clifford):
        raise QiskitError("cliff is not a Clifford element.")

    # Note: from the commutativity relations, we know that D = (A^T)^{-1},
    # and in fact we have already used this fact in _decompose_hadamard_free

    num_qubits = cliff.num_qubits
    if not (cliff.stab_x == np.zeros((num_qubits, num_qubits))).all():
        if not (cliff.destab_z == np.zeros((num_qubits, num_qubits))).all():
            raise QiskitError("The given Clifford is not linear reversible.")

    CX_circ = synth_cnot_count_full_pmh(cliff.destab_x.transpose())

    if validate:
        _check_gates(CX_circ, ("cx", "swap"))

    return CX_circ


def _default_cz_synth_func(symmetric_mat, validate):
    """
    Construct the layer of CZ gates from a symmetric matrix.
    """
    nq = symmetric_mat.shape[0]
    qc = QuantumCircuit(nq)

    for j in range(nq):
        for i in range(0, j):
            if symmetric_mat[i][j]:
                qc.cz(i, j)

    if validate:
        _check_gates(qc, "cz")
    return qc
