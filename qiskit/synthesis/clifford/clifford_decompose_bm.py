# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Circuit synthesis for 2-qubit and 3-qubit Cliffords.
"""
# pylint: disable=invalid-name

# ---------------------------------------------------------------------
# Synthesis based on Bravyi & Maslov decomposition
# ---------------------------------------------------------------------


from itertools import product
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
    _append_cx,
    _append_v,
    _append_w,
)


def synth_clifford_bm(clifford):
    """Optimal CX-cost decomposition of a Clifford operator on 2-qubits or 3-qubits
    into a QuantumCircuit based on Bravyi-Maslov method.

    Args:
        clifford (Clifford): a clifford operator.

    Return:
        QuantumCircuit: a circuit implementation of the Clifford.

    Raises:
        QiskitError: if clifford is on more than 3 qubits.

    Reference:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
    """
    num_qubits = clifford.num_qubits

    if num_qubits > 3:
        raise QiskitError("Can only decompose up to 3-qubit Clifford circuits.")

    if num_qubits == 1:
        return _decompose_clifford_1q(clifford.tableau)

    clifford_name = str(clifford)

    # Inverse of final decomposed circuit
    inv_circuit = QuantumCircuit(num_qubits, name="inv_circ")

    # CNOT cost of clifford
    cost = _cx_cost(clifford)

    # Find composition of circuits with CX and (H.S)^a gates to reduce CNOT count
    while cost > 0:
        clifford, inv_circuit, cost = _reduce_cost(clifford, inv_circuit, cost)

    # Decompose the remaining product of 1-qubit cliffords
    ret_circ = QuantumCircuit(num_qubits, name=clifford_name)
    for qubit in range(num_qubits):
        pos = [qubit, qubit + num_qubits]
        circ = _decompose_clifford_1q(clifford.tableau[pos][:, pos + [-1]])
        if len(circ) > 0:
            ret_circ.append(circ, [qubit])

    # Add the inverse of the 2-qubit reductions circuit
    if len(inv_circuit) > 0:
        ret_circ.append(inv_circuit.inverse(), range(num_qubits))

    return ret_circ.decompose()


# ---------------------------------------------------------------------
# 1-qubit Clifford decomposition
# ---------------------------------------------------------------------


def _decompose_clifford_1q(tableau):
    """Decompose a single-qubit clifford"""
    circuit = QuantumCircuit(1, name="temp")

    # Add phase correction
    destab_phase, stab_phase = tableau[:, 2]
    if destab_phase and not stab_phase:
        circuit.z(0)
    elif not destab_phase and stab_phase:
        circuit.x(0)
    elif destab_phase and stab_phase:
        circuit.y(0)
    destab_phase_label = "-" if destab_phase else "+"
    stab_phase_label = "-" if stab_phase else "+"

    destab_x, destab_z = tableau[0, 0], tableau[0, 1]
    stab_x, stab_z = tableau[1, 0], tableau[1, 1]

    # Z-stabilizer
    if stab_z and not stab_x:
        stab_label = "Z"
        if destab_z:
            destab_label = "Y"
            circuit.s(0)
        else:
            destab_label = "X"

    # X-stabilizer
    elif not stab_z and stab_x:
        stab_label = "X"
        if destab_x:
            destab_label = "Y"
            circuit.sdg(0)
        else:
            destab_label = "Z"
        circuit.h(0)

    # Y-stabilizer
    else:
        stab_label = "Y"
        if destab_z:
            destab_label = "Z"
        else:
            destab_label = "X"
            circuit.s(0)
        circuit.h(0)
        circuit.s(0)

    # Add circuit name
    name_destab = f"Destabilizer = ['{destab_phase_label}{destab_label}']"
    name_stab = f"Stabilizer = ['{stab_phase_label}{stab_label}']"
    circuit.name = f"Clifford: {name_stab}, {name_destab}"
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
    U = clifford.tableau[:, :-1]
    r00 = _rank2(U[0, 0], U[0, 2], U[2, 0], U[2, 2])
    r01 = _rank2(U[0, 1], U[0, 3], U[2, 1], U[2, 3])
    if r00 == 2:
        return r01
    return r01 + 1 - r00


def _cx_cost3(clifford):
    """Return CX cost of a 3-qubit clifford."""
    # pylint: disable=too-many-return-statements,too-many-boolean-expressions
    U = clifford.tableau[:, :-1]
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
            isLocY = np.array_equal((U[q1, :] ^ U[q1 + n, :]) & mask, (U[q1, :] ^ U[q1 + n, :]))
            R1[q1, q2] = 1 * (isLocX or isLocZ or isLocY) + 1 * (isLocX and isLocZ and isLocY)

    diag1 = np.sort(np.diag(R1)).tolist()
    diag2 = np.sort(np.diag(R2)).tolist()

    nz1 = np.count_nonzero(R1)
    nz2 = np.count_nonzero(R2)

    if diag1 == [2, 2, 2]:
        return 0

    if diag1 == [1, 1, 2]:
        return 1

    if (
        diag1 == [0, 1, 1]
        or (diag1 == [1, 1, 1] and nz2 < 9)
        or (diag1 == [0, 0, 2] and diag2 == [1, 1, 2])
    ):
        return 2

    if (
        (diag1 == [1, 1, 1] and nz2 == 9)
        or (
            diag1 == [0, 0, 1]
            and (nz1 == 1 or diag2 == [2, 2, 2] or (diag2 == [1, 1, 2] and nz2 < 9))
        )
        or (diag1 == [0, 0, 2] and diag2 == [0, 0, 2])
        or (diag2 == [1, 2, 2] and nz1 == 0)
    ):
        return 3

    if diag2 == [0, 0, 1] or (
        diag1 == [0, 0, 0]
        and (
            (diag2 == [1, 1, 1] and nz2 == 9 and nz1 == 3)
            or (diag2 == [0, 1, 1] and nz2 == 8 and nz1 == 2)
        )
    ):
        return 5

    if nz1 == 3 and nz2 == 3:
        return 6

    return 4
