# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for the Clifford class.
"""
# pylint: disable=invalid-name

from itertools import product
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
    _append_z, _append_x, _append_h, _append_s, _append_v, _append_w, _append_cx, _append_swap)


def decompose_clifford(clifford):
    """Decompose a Clifford operator into a QuantumCircuit.

    For N <= 3 qubits this is based on optimal CX cost decomposition
    from reference [1]. For N > 3 qubits this is done using the general
    non-optimal compilation routine from reference [2].

    Args:
        clifford (Clifford): a clifford operator.

    Return:
        QuantumCircuit: a circuit implementation of the Clifford.

    References:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_

        2. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """
    num_qubits = clifford.num_qubits

    if num_qubits <= 3:
        return decompose_clifford_bm(clifford)

    return decompose_clifford_ag(clifford)


# ---------------------------------------------------------------------
# Synthesis based on Bravyi & Maslov decomposition
# ---------------------------------------------------------------------

def decompose_clifford_bm(clifford):
    """Decompose a clifford"""
    num_qubits = clifford.num_qubits

    if num_qubits == 1:
        return _decompose_clifford_1q(
            clifford.table.array, clifford.table.phase)

    # Inverse of final decomposed circuit
    inv_circuit = QuantumCircuit(num_qubits, name='inv_circ')

    # CNOT cost of clifford
    cost = _cx_cost(clifford)

    # Find composition of circuits with CX and (H.S)^a gates to reduce CNOT count
    while cost > 0:
        clifford, inv_circuit, cost = _reduce_cost(clifford, inv_circuit, cost)

    # Decompose the remaining product of 1-qubit cliffords
    ret_circ = QuantumCircuit(num_qubits, name=str(clifford))
    for qubit in range(num_qubits):
        pos = [qubit, qubit + num_qubits]
        table = clifford.table.array[pos][:, pos]
        phase = clifford.table.phase[pos]
        circ = _decompose_clifford_1q(table, phase)
        if len(circ) > 0:
            ret_circ.append(circ, [qubit])

    # Add the inverse of the 2-qubit reductions circuit
    if len(inv_circuit) > 0:
        ret_circ.append(inv_circuit.inverse(), range(num_qubits))
    return ret_circ.decompose()


# ---------------------------------------------------------------------
# Synthesis based on Aaronson & Gottesman decomposition
# ---------------------------------------------------------------------

def decompose_clifford_ag(clifford):
    """Decompose a Clifford operator into a QuantumCircuit.

    Args:
        clifford (Clifford): a clifford operator.

    Return:
        QuantumCircuit: a circuit implementation of the Clifford.
    """
    # Use 1-qubit decomposition method
    if clifford.num_qubits == 1:
        return _decompose_clifford_1q(
            clifford.table.array, clifford.table.phase)

    # Compose a circuit which we will convert to an instruction
    circuit = QuantumCircuit(clifford.num_qubits,
                             name=str(clifford))

    # Make a copy of Clifford as we are going to do row reduction to
    # reduce it to an identity
    clifford_cpy = clifford.copy()

    for i in range(clifford.num_qubits):
        # put a 1 one into position by permuting and using Hadamards(i,i)
        _set_qubit_x_true(clifford_cpy, circuit, i)
        # make all entries in row i except ith equal to 0
        # by using phase gate and CNOTS
        _set_row_x_zero(clifford_cpy, circuit, i)
        # treat Zs
        _set_row_z_zero(clifford_cpy, circuit, i)

    for i in range(clifford.num_qubits):
        if clifford_cpy.destabilizer.phase[i]:
            _append_z(clifford_cpy, i)
            circuit.z(i)
        if clifford_cpy.stabilizer.phase[i]:
            _append_x(clifford_cpy, i)
            circuit.x(i)
    # Next we invert the circuit to undo the row reduction and return the
    # result as a gate instruction
    return circuit.inverse()


# ---------------------------------------------------------------------
# 1-qubit Clifford decomposition
# ---------------------------------------------------------------------

def _decompose_clifford_1q(pauli, phase):
    """Decompose a single-qubit clifford"""
    circuit = QuantumCircuit(1, name='temp')

    # Add phase correction
    destab_phase, stab_phase = phase
    if destab_phase and not stab_phase:
        circuit.z(0)
    elif not destab_phase and stab_phase:
        circuit.x(0)
    elif destab_phase and stab_phase:
        circuit.y(0)
    destab_phase_label = '-' if destab_phase else '+'
    stab_phase_label = '-' if stab_phase else '+'

    destab_x, destab_z = pauli[0]
    stab_x, stab_z = pauli[1]

    # Z-stabilizer
    if stab_z and not stab_x:
        stab_label = 'Z'
        if destab_z:
            destab_label = 'Y'
            circuit.s(0)
        else:
            destab_label = 'X'

    # X-stabilizer
    elif not stab_z and stab_x:
        stab_label = 'X'
        if destab_x:
            destab_label = 'Y'
            circuit.sdg(0)
        else:
            destab_label = 'Z'
        circuit.h(0)

    # Y-stabilizer
    else:
        stab_label = 'Y'
        if destab_z:
            destab_label = 'Z'
        else:
            destab_label = 'X'
            circuit.s(0)
        circuit.h(0)
        circuit.s(0)

    # Add circuit name
    name_destab = "Destabilizer = ['{}{}']".format(destab_phase_label, destab_label)
    name_stab = "Stabilizer = ['{}{}']".format(stab_phase_label, stab_label)
    circuit.name = "Clifford: {}, {}".format(name_stab, name_destab)
    return circuit


# ---------------------------------------------------------------------
# Helper functions for Bravyi & Maslov decomposition
# ---------------------------------------------------------------------

def _reduce_cost(clifford, inv_circuit, cost):
    """Two-qubit cost reduction step"""
    num_qubits = clifford.num_qubits
    for qubit0 in range(num_qubits):
        for qubit1 in range(qubit0 + 1, num_qubits):
            for n0, n1 in product(range(3), repeat=2):

                # Apply a 2-qubit block
                reduced = clifford.copy()
                for qubit, n in [(qubit0, n0), (qubit1, n1)]:
                    if n == 1:
                        _append_v(reduced, qubit)
                    elif n == 2:
                        _append_w(reduced, qubit)
                _append_cx(reduced, qubit0, qubit1)

                # Compute new cost
                new_cost = _cx_cost(reduced)

                if new_cost == cost - 1:
                    # Add decomposition to inverse circuit
                    for qubit, n in [(qubit0, n0), (qubit1, n1)]:
                        if n == 1:
                            inv_circuit.sdg(qubit)
                            inv_circuit.h(qubit)
                        elif n == 2:
                            inv_circuit.h(qubit)
                            inv_circuit.s(qubit)
                    inv_circuit.cx(qubit0, qubit1)

                    return reduced, inv_circuit, new_cost

    # If we didn't reduce cost
    raise QiskitError("Failed to reduce Clifford CX cost.")


def _cx_cost(clifford):
    """Return the number of CX gates required for Clifford decomposition."""
    if clifford.num_qubits == 2:
        return _cx_cost2(clifford)
    if clifford.num_qubits == 3:
        return _cx_cost3(clifford)
    raise Exception("No Clifford CX cost function for num_qubits > 3.")


def _rank2(a, b, c, d):
    """Return rank of 2x2 boolean matrix."""
    if (a & d) ^ (b & c):
        return 2
    if a or b or c or d:
        return 1
    return 0


def _cx_cost2(clifford):
    """Return CX cost of a 2-qubit clifford."""
    U = clifford.table.array
    r00 = _rank2(U[0, 0], U[0, 2], U[2, 0], U[2, 2])
    r01 = _rank2(U[0, 1], U[0, 3], U[2, 1], U[2, 3])
    if r00 == 2:
        return r01
    return r01 + 1 - r00


def _cx_cost3(clifford):
    """Return CX cost of a 3-qubit clifford."""
    # pylint: disable=too-many-return-statements,too-many-boolean-expressions
    U = clifford.table.array
    n = 3
    # create information transfer matrices R1, R2
    R1 = np.zeros((n, n), dtype=int)
    R2 = np.zeros((n, n), dtype=int)
    for q1 in range(n):
        for q2 in range(n):
            R2[q1, q2] = _rank2(U[q1, q2], U[q1, q2 + n], U[q1 + n, q2], U[q1 + n, q2 + n])
            mask = np.zeros(2 * n, dtype=int)
            mask[[q2, q2 + n]] = 1
            isLocX = np.array_equal(U[q1, :] & mask, U[q1, :])
            isLocZ = np.array_equal(U[q1 + n, :] & mask, U[q1 + n, :])
            isLocY = np.array_equal((U[q1, :] ^ U[q1 + n, :]) & mask,
                                    (U[q1, :] ^ U[q1 + n, :]))
            R1[q1, q2] = 1 * (isLocX or isLocZ or isLocY) + 1 * (isLocX and isLocZ and isLocY)

    diag1 = np.sort(np.diag(R1)).tolist()
    diag2 = np.sort(np.diag(R2)).tolist()

    nz1 = np.count_nonzero(R1)
    nz2 = np.count_nonzero(R2)

    if diag1 == [2, 2, 2]:
        return 0

    if diag1 == [1, 1, 2]:
        return 1

    if (diag1 == [0, 1, 1]
            or (diag1 == [1, 1, 1] and nz2 < 9)
            or (diag1 == [0, 0, 2] and diag2 == [1, 1, 2])):
        return 2

    if ((diag1 == [1, 1, 1] and nz2 == 9)
            or (diag1 == [0, 0, 1] and (
                nz1 == 1 or diag2 == [2, 2, 2] or (diag2 == [1, 1, 2] and nz2 < 9)))
            or (diag1 == [0, 0, 2] and diag2 == [0, 0, 2])
            or (diag2 == [1, 2, 2] and nz1 == 0)):
        return 3

    if (diag2 == [0, 0, 1] or (diag1 == [0, 0, 0] and (
            (diag2 == [1, 1, 1] and nz2 == 9 and nz1 == 3)
            or (diag2 == [0, 1, 1] and nz2 == 8 and nz1 == 2)))):
        return 5

    if nz1 == 3 and nz2 == 3:
        return 6

    return 4


# ---------------------------------------------------------------------
# Helper functions for Aaronson & Gottesman decomposition
# ---------------------------------------------------------------------

def _set_qubit_x_true(clifford, circuit, qubit):
    """Set destabilizer.X[qubit, qubit] to be True.

    This is done by permuting columns l > qubit or if necessary applying
    a Hadamard
    """
    x = clifford.destabilizer.X[qubit]
    z = clifford.destabilizer.Z[qubit]

    if x[qubit]:
        return

    # Try to find non-zero element
    for i in range(qubit + 1, clifford.num_qubits):
        if x[i]:
            _append_swap(clifford, i, qubit)
            circuit.swap(i, qubit)
            return

    # no non-zero element found: need to apply Hadamard somewhere
    for i in range(qubit, clifford.num_qubits):
        if z[i]:
            _append_h(clifford, i)
            circuit.h(i)
            if i != qubit:
                _append_swap(clifford, i, qubit)
                circuit.swap(i, qubit)
            return


def _set_row_x_zero(clifford, circuit, qubit):
    """Set destabilizer.X[qubit, i] to False for all i > qubit.

    This is done by applying CNOTS assumes k<=N and A[k][k]=1
    """
    x = clifford.destabilizer.X[qubit]
    z = clifford.destabilizer.Z[qubit]

    # Check X first
    for i in range(qubit + 1, clifford.num_qubits):
        if x[i]:
            _append_cx(clifford, qubit, i)
            circuit.cx(qubit, i)

    # Check whether Zs need to be set to zero:
    if np.any(z[qubit:]):
        if not z[qubit]:
            # to treat Zs: make sure row.Z[k] to True
            _append_s(clifford, qubit)
            circuit.s(qubit)

        # reverse CNOTS
        for i in range(qubit + 1, clifford.num_qubits):
            if z[i]:
                _append_cx(clifford, i, qubit)
                circuit.cx(i, qubit)
        # set row.Z[qubit] to False
        _append_s(clifford, qubit)
        circuit.s(qubit)


def _set_row_z_zero(clifford, circuit, qubit):
    """Set stabilizer.Z[qubit, i] to False for all i > qubit.

    Implemented by applying (reverse) CNOTS assumes qubit < num_qubits
    and _set_row_x_zero has been called first
    """

    x = clifford.stabilizer.X[qubit]
    z = clifford.stabilizer.Z[qubit]

    # check whether Zs need to be set to zero:
    if np.any(z[qubit + 1:]):
        for i in range(qubit + 1, clifford.num_qubits):
            if z[i]:
                _append_cx(clifford, i, qubit)
                circuit.cx(i, qubit)

    # check whether Xs need to be set to zero:
    if np.any(x[qubit:]):
        _append_h(clifford, qubit)
        circuit.h(qubit)
        for i in range(qubit + 1, clifford.num_qubits):
            if x[i]:
                _append_cx(clifford, qubit, i)
                circuit.cx(qubit, i)
        if z[qubit]:
            _append_s(clifford, qubit)
            circuit.s(qubit)
        _append_h(clifford, qubit)
        circuit.h(qubit)
