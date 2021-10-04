# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021
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
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
    _append_z,
    _append_x,
    _append_h,
    _append_s,
    _append_v,
    _append_w,
    _append_cx,
    _append_swap,
)


def decompose_clifford(clifford, method=None):
    """Decompose a Clifford operator into a QuantumCircuit.

    For N <= 3 qubits this is based on optimal CX cost decomposition
    from reference [1]. For N > 3 qubits this is done using the general
    non-optimal greedy compilation routine from reference [3],
    which typically yields better CX cost compared to the AG method in [2].

    Args:
        clifford (Clifford): a clifford operator.
        method (str):  Optional, a synthesis method ('AG' or 'greedy').
             If set this overrides optimal decomposition for N <=3 qubits.

    Return:
        QuantumCircuit: a circuit implementation of the Clifford.

    References:
        1. S. Bravyi, D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*,
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_

        2. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_

        3. Sergey Bravyi, Shaohan Hu, Dmitri Maslov, Ruslan Shaydulin,
           *Clifford Circuit Optimization with Templates and Symbolic Pauli Gates*,
           `arXiv:2105.02291 [quant-ph] <https://arxiv.org/abs/2105.02291>`_
    """
    num_qubits = clifford.num_qubits

    if method == "AG":
        return decompose_clifford_ag(clifford)

    if method == "greedy":
        return decompose_clifford_greedy(clifford)

    if num_qubits <= 3:
        return decompose_clifford_bm(clifford)

    return decompose_clifford_greedy(clifford)


# ---------------------------------------------------------------------
# Synthesis based on Bravyi & Maslov decomposition
# ---------------------------------------------------------------------


def decompose_clifford_bm(clifford):
    """Decompose a clifford"""
    num_qubits = clifford.num_qubits

    if num_qubits == 1:
        return _decompose_clifford_1q(clifford.table.array, clifford.table.phase)

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
        return _decompose_clifford_1q(clifford.table.array, clifford.table.phase)

    # Compose a circuit which we will convert to an instruction
    circuit = QuantumCircuit(clifford.num_qubits, name=str(clifford))

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
    circuit = QuantumCircuit(1, name="temp")

    # Add phase correction
    destab_phase, stab_phase = phase
    if destab_phase and not stab_phase:
        circuit.z(0)
    elif not destab_phase and stab_phase:
        circuit.x(0)
    elif destab_phase and stab_phase:
        circuit.y(0)
    destab_phase_label = "-" if destab_phase else "+"
    stab_phase_label = "-" if stab_phase else "+"

    destab_x, destab_z = pauli[0]
    stab_x, stab_z = pauli[1]

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
    if np.any(z[qubit + 1 :]):
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


# ---------------------------------------------------------------------
# Synthesis based on Bravyi et. al. greedy clifford compiler
# ---------------------------------------------------------------------


def decompose_clifford_greedy(clifford):
    """Decompose a Clifford operator into a QuantumCircuit.

    Args:
        clifford (Clifford): a clifford operator.

    Return:
        QuantumCircuit: a circuit implementation of the Clifford.

    Raises:
        QiskitError: if symplectic Gaussian elimination fails.
    """

    num_qubits = clifford.num_qubits
    circ = QuantumCircuit(num_qubits, name=str(clifford))
    qubit_list = list(range(num_qubits))
    clifford_cpy = clifford.copy()

    # Reducing the original Clifford to identity
    # via symplectic Gaussian elimination
    while len(qubit_list) > 0:
        list_greedy_cost = []
        for qubit in qubit_list:
            pauli_x = Pauli(num_qubits * "I")
            pauli_x[qubit] = "X"
            pauli_x = pauli_x.evolve(clifford_cpy)

            pauli_z = Pauli(num_qubits * "I")
            pauli_z[qubit] = "Z"
            pauli_z = pauli_z.evolve(clifford_cpy)
            list_pairs = []
            pauli_count = 0

            # Compute the CNOT cost in order to find the qubit with the minimal cost
            for i in qubit_list:
                typeq = _from_pair_paulis_to_type(pauli_x, pauli_z, i)
                list_pairs.append(typeq)
                pauli_count += 1
            cost = _compute_greedy_cost(list_pairs)
            list_greedy_cost.append([cost, qubit])

        _, min_qubit = (sorted(list_greedy_cost))[0]

        # Gaussian elimination step for the qubit with minimal CNOT cost
        pauli_x = Pauli(num_qubits * "I")
        pauli_x[min_qubit] = "X"
        pauli_x = pauli_x.evolve(clifford_cpy)

        pauli_z = Pauli(num_qubits * "I")
        pauli_z[min_qubit] = "Z"
        pauli_z = pauli_z.evolve(clifford_cpy)

        # Compute the decoupling operator of cliff_ox and cliff_oz
        decouple_circ, decouple_cliff = _calc_decoupling(
            pauli_x, pauli_z, qubit_list, min_qubit, num_qubits, clifford_cpy
        )
        circ = circ.compose(decouple_circ)

        # Now the clifford acts trivially on min_qubit
        clifford_cpy = decouple_cliff.adjoint().compose(clifford_cpy)
        qubit_list.remove(min_qubit)

    # Add the phases (Pauli gates) to the Clifford circuit
    for qubit in range(num_qubits):
        stab = clifford_cpy.stabilizer.phase[qubit]
        destab = clifford_cpy.destabilizer.phase[qubit]
        if destab and stab:
            circ.y(qubit)
        elif not destab and stab:
            circ.x(qubit)
        elif destab and not stab:
            circ.z(qubit)

    return circ


# ---------------------------------------------------------------------
# Helper functions for Bravyi et. al. greedy clifford compiler
# ---------------------------------------------------------------------

# Global arrays of the 16 pairs of Pauli operators
# divided into 5 equivalence classes under the action of single-qubit Cliffords

# Class A - canonical representative is 'XZ'
A_class = [
    [[False, True], [True, True]],  # 'XY'
    [[False, True], [True, False]],  # 'XZ'
    [[True, True], [False, True]],  # 'YX'
    [[True, True], [True, False]],  # 'YZ'
    [[True, False], [False, True]],  # 'ZX'
    [[True, False], [True, True]],
]  # 'ZY'

# Class B - canonical representative is 'XX'
B_class = [
    [[True, False], [True, False]],  # 'ZZ'
    [[False, True], [False, True]],  # 'XX'
    [[True, True], [True, True]],
]  # 'YY'

# Class C - canonical representative is 'XI'
C_class = [
    [[True, False], [False, False]],  # 'ZI'
    [[False, True], [False, False]],  # 'XI'
    [[True, True], [False, False]],
]  # 'YI'

# Class D - canonical representative is 'IZ'
D_class = [
    [[False, False], [False, True]],  # 'IX'
    [[False, False], [True, False]],  # 'IZ'
    [[False, False], [True, True]],
]  # 'IY'

# Class E - only 'II'
E_class = [[[False, False], [False, False]]]  # 'II'


def _from_pair_paulis_to_type(pauli_x, pauli_z, qubit):
    """Converts a pair of Paulis pauli_x and pauli_z into a type"""

    type_x = [pauli_x.z[qubit], pauli_x.x[qubit]]
    type_z = [pauli_z.z[qubit], pauli_z.x[qubit]]
    return [type_x, type_z]


def _compute_greedy_cost(list_pairs):
    """Compute the CNOT cost of one step of the algorithm"""

    A_num = 0
    B_num = 0
    C_num = 0
    D_num = 0

    for pair in list_pairs:
        if pair in A_class:
            A_num += 1
        elif pair in B_class:
            B_num += 1
        elif pair in C_class:
            C_num += 1
        elif pair in D_class:
            D_num += 1

    if (A_num % 2) == 0:
        raise QiskitError("Symplectic Gaussian elimination fails.")

    # Calculate the CNOT cost
    cost = 3 * (A_num - 1) / 2 + (B_num + 1) * (B_num > 0) + C_num + D_num
    if list_pairs[0] not in A_class:  # additional SWAP
        cost += 3

    return cost


def _calc_decoupling(pauli_x, pauli_z, qubit_list, min_qubit, num_qubits, cliff):
    """Calculate a decoupling operator D:
    D^{-1} * Ox * D = x1
    D^{-1} * Oz * D = z1
    and reduce the clifford such that it will act trivially on min_qubit
    """

    circ = QuantumCircuit(num_qubits)

    # decouple_cliff is initialized to an identity clifford
    decouple_cliff = cliff.copy()
    num_qubits = decouple_cliff.num_qubits
    decouple_cliff.table.phase = np.zeros(2 * num_qubits)
    decouple_cliff.table.array = np.eye(2 * num_qubits)

    qubit0 = min_qubit  # The qubit for the symplectic Gaussian elimination

    # Reduce the pair of Paulis to a representative in the equivalence class
    # ['XZ', 'XX', 'XI', 'IZ', 'II'] by adding single-qubit gates
    for qubit in qubit_list:

        typeq = _from_pair_paulis_to_type(pauli_x, pauli_z, qubit)

        if typeq in [
            [[True, True], [False, False]],  # 'YI'
            [[True, True], [True, True]],  # 'YY'
            [[True, True], [True, False]],
        ]:  # 'YZ':
            circ.s(qubit)
            _append_s(decouple_cliff, qubit)

        elif typeq in [
            [[True, False], [False, False]],  # 'ZI'
            [[True, False], [True, False]],  # 'ZZ'
            [[True, False], [False, True]],  # 'ZX'
            [[False, False], [False, True]],
        ]:  # 'IX'
            circ.h(qubit)
            _append_h(decouple_cliff, qubit)

        elif typeq in [
            [[False, False], [True, True]],  # 'IY'
            [[True, False], [True, True]],
        ]:  # 'ZY'
            circ.s(qubit)
            circ.h(qubit)
            _append_s(decouple_cliff, qubit)
            _append_h(decouple_cliff, qubit)

        elif typeq == [[True, True], [False, True]]:  # 'YX'
            circ.h(qubit)
            circ.s(qubit)
            _append_h(decouple_cliff, qubit)
            _append_s(decouple_cliff, qubit)

        elif typeq == [[False, True], [True, True]]:  # 'XY'
            circ.s(qubit)
            circ.h(qubit)
            circ.s(qubit)
            _append_s(decouple_cliff, qubit)
            _append_h(decouple_cliff, qubit)
            _append_s(decouple_cliff, qubit)

    # Reducing each pair of Paulis (except of qubit0) to 'II'
    # by adding two-qubit gates and single-qubit gates
    A_qubits = []
    B_qubits = []
    C_qubits = []
    D_qubits = []

    for qubit in qubit_list:
        typeq = _from_pair_paulis_to_type(pauli_x, pauli_z, qubit)
        if typeq in A_class:
            A_qubits.append(qubit)
        elif typeq in B_class:
            B_qubits.append(qubit)
        elif typeq in C_class:
            C_qubits.append(qubit)
        elif typeq in D_class:
            D_qubits.append(qubit)

    if len(A_qubits) % 2 != 1:
        raise QiskitError("Symplectic Gaussian elimination fails.")

    if qubit0 not in A_qubits:  # SWAP qubit0 and qubitA
        qubitA = A_qubits[0]
        circ.swap(qubit0, qubitA)
        _append_swap(decouple_cliff, qubit0, qubitA)
        if qubit0 in B_qubits:
            B_qubits.remove(qubit0)
            B_qubits.append(qubitA)
            A_qubits.remove(qubitA)
            A_qubits.append(qubit0)
        elif qubit0 in C_qubits:
            C_qubits.remove(qubit0)
            C_qubits.append(qubitA)
            A_qubits.remove(qubitA)
            A_qubits.append(qubit0)
        elif qubit0 in D_qubits:
            D_qubits.remove(qubit0)
            D_qubits.append(qubitA)
            A_qubits.remove(qubitA)
            A_qubits.append(qubit0)
        else:
            A_qubits.remove(qubitA)
            A_qubits.append(qubit0)

    # Reduce pairs in Class C to 'II'
    for qubit in C_qubits:
        circ.cx(qubit0, qubit)
        _append_cx(decouple_cliff, qubit0, qubit)

    # Reduce pairs in Class D to 'II'
    for qubit in D_qubits:
        circ.cx(qubit, qubit0)
        _append_cx(decouple_cliff, qubit, qubit0)

    # Reduce pairs in Class B to 'II'
    if len(B_qubits) > 1:
        for qubit in B_qubits[1:]:
            qubitB = B_qubits[0]
            circ.cx(qubitB, qubit)
            _append_cx(decouple_cliff, qubitB, qubit)

    if len(B_qubits) > 0:
        qubitB = B_qubits[0]
        circ.cx(qubit0, qubitB)
        circ.h(qubitB)
        circ.cx(qubitB, qubit0)
        _append_cx(decouple_cliff, qubit0, qubitB)
        _append_h(decouple_cliff, qubitB)
        _append_cx(decouple_cliff, qubitB, qubit0)

    # Reduce pairs in Class A (except of qubit0) to 'II'
    Alen = int((len(A_qubits) - 1) / 2)
    if Alen > 0:
        A_qubits.remove(qubit0)
    for qubit in range(Alen):
        circ.cx(A_qubits[2 * qubit + 1], A_qubits[2 * qubit])
        circ.cx(A_qubits[2 * qubit], qubit0)
        circ.cx(qubit0, A_qubits[2 * qubit + 1])
        _append_cx(decouple_cliff, A_qubits[2 * qubit + 1], A_qubits[2 * qubit])
        _append_cx(decouple_cliff, A_qubits[2 * qubit], qubit0)
        _append_cx(decouple_cliff, qubit0, A_qubits[2 * qubit + 1])

    return circ, decouple_cliff
