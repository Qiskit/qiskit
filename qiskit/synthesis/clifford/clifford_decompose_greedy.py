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
Circuit synthesis for the Clifford class.
"""
# pylint: disable=invalid-name

# ---------------------------------------------------------------------
# Synthesis based on Bravyi et. al. greedy clifford compiler
# ---------------------------------------------------------------------


import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
    _append_cx,
    _append_h,
    _append_s,
    _append_swap,
)
from qiskit.quantum_info.operators.symplectic.pauli import Pauli


def synth_clifford_greedy(clifford):
    """Decompose a Clifford operator into a QuantumCircuit based on the
    greedy Clifford compiler that is described in Appendix A of
    Bravyi, Hu, Maslov and Shaydulin.

    This method typically yields better CX cost compared to the Aaronson-Gottesma method.

    Args:
        clifford (Clifford): a clifford operator.

    Return:
        QuantumCircuit: a circuit implementation of the Clifford.

    Raises:
        QiskitError: if symplectic Gaussian elimination fails.

    Reference:
        1. Sergey Bravyi, Shaohan Hu, Dmitri Maslov, Ruslan Shaydulin,
           *Clifford Circuit Optimization with Templates and Symbolic Pauli Gates*,
           `arXiv:2105.02291 [quant-ph] <https://arxiv.org/abs/2105.02291>`_
    """

    num_qubits = clifford.num_qubits
    circ = QuantumCircuit(num_qubits, name=str(clifford))
    qubit_list = list(range(num_qubits))
    clifford_cpy = clifford.copy()

    # Reducing the original Clifford to identity
    # via symplectic Gaussian elimination
    while len(qubit_list) > 0:
        # Calculate the adjoint of clifford_cpy without the phase
        clifford_adj = clifford_cpy.copy()
        tmp = clifford_adj.destab_x.copy()
        clifford_adj.destab_x = clifford_adj.stab_z.T
        clifford_adj.destab_z = clifford_adj.destab_z.T
        clifford_adj.stab_x = clifford_adj.stab_x.T
        clifford_adj.stab_z = tmp.T

        list_greedy_cost = []
        for qubit in qubit_list:
            pauli_x = Pauli("I" * (num_qubits - qubit - 1) + "X" + "I" * qubit)
            pauli_x = pauli_x.evolve(clifford_adj, frame="s")

            pauli_z = Pauli("I" * (num_qubits - qubit - 1) + "Z" + "I" * qubit)
            pauli_z = pauli_z.evolve(clifford_adj, frame="s")
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
        pauli_x = Pauli("I" * (num_qubits - min_qubit - 1) + "X" + "I" * min_qubit)
        pauli_x = pauli_x.evolve(clifford_adj, frame="s")

        pauli_z = Pauli("I" * (num_qubits - min_qubit - 1) + "Z" + "I" * min_qubit)
        pauli_z = pauli_z.evolve(clifford_adj, frame="s")

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
        stab = clifford_cpy.stab_phase[qubit]
        destab = clifford_cpy.destab_phase[qubit]
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
    decouple_cliff.phase = np.zeros(2 * num_qubits)
    decouple_cliff.symplectic_matrix = np.eye(2 * num_qubits)

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
