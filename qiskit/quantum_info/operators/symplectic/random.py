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
Random symplectic operator functions
"""

import numpy as np

from qiskit.exceptions import QiskitError
from .clifford import Clifford
from .stabilizer_table import StabilizerTable
from .pauli_table import PauliTable


def random_pauli_table(num_qubits, size=1, random_state=None):
    """Return a random PauliTable.

    Args:
        num_qubits (int): the number of qubits.
        size (int): Optional. The number of rows of the table (Default: 1).
        random_state(np.RandomState or int): Optional. A specific random
                                             state for RNG.

    Returns:
        PauliTable: a random PauliTable.
    """
    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    table = random_state.randint(2, size=(size, 2 * num_qubits)).astype(np.bool)
    return PauliTable(table)


def random_stabilizer_table(num_qubits, size=1, random_state=None):
    """Return a random StabilizerTable.

    Args:
        num_qubits (int): the number of qubits.
        size (int): Optional. The number of rows of the table (Default: 1).
        random_state(np.RandomState or int): Optional. A specific random
                                             state for RNG.

    Returns:
        PauliTable: a random StabilizerTable.
    """
    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    table = random_state.randint(2, size=(size, 2 * num_qubits)).astype(np.bool)
    phase = random_state.randint(2, size=size).astype(np.bool)
    return StabilizerTable(table, phase)

def random_clifford(num_qubits, random_state=None):
    """Return a random Clifford operator.

    The Clifford is sampled using the method of Reference [1].

    Args:
        num_qubits (int): the number of qubits for the Clifford
        random_state (np.RandomState or int): Optional. A specific random
                                                        state for RNG.

    Returns:
        Clifford: a random Clifford operator.

    Reference:
        1. S. Bravyi and D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*.
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
    """
    if random_state is None:
        random_state = np.random
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)
    elif not isinstance(random_state, np.RandomState):
        raise QiskitError("Invalid random_state")

    had, perm = sample_qmallows(num_qubits, random_state)

    gamma1 = np.diag(random_state.randint(2, size=num_qubits))
    gamma2 = np.diag(random_state.randint(2, size=num_qubits))
    delta1 = np.eye(num_qubits, dtype=int)
    delta2 = delta1.copy()

    def fill_tril(mat, symmetric=False):
        """Add symmetric random ints to off diagonals"""
        vals = random_state.randint(2, size=num_qubits * (num_qubits - 1) // 2)
        rows, cols = np.tril_indices(num_qubits, -1)
        mat[(rows, cols)] = vals
        if symmetric:
            mat[(cols, rows)] = vals
        return mat

    fill_tril(gamma1, symmetric=True)
    fill_tril(gamma2, symmetric=True)
    fill_tril(delta1)
    fill_tril(delta2)

    # Compute stabilizer table
    zero = np.zeros((num_qubits, num_qubits), dtype=int)
    prod1 = np.matmul(gamma1, delta1)
    inv1 = np.linalg.inv(np.transpose(delta1)).astype(int)
    table1 = np.mod(np.block([[delta1, zero], [prod1, inv1]]), 2)

    prod2 = np.matmul(gamma2, delta2)
    inv2 = np.linalg.inv(np.transpose(delta2)).astype(int)
    table2 = np.mod(np.block([[delta2, zero], [prod2, inv2]]), 2)

    # Apply qubit permutation
    table = table2[np.concatenate([perm, num_qubits + perm])]

    # Apply layer of Hadamards
    inds = had * np.arange(1, num_qubits + 1)
    inds = inds[inds > 0] - 1
    lhs_inds = np.concatenate([inds, inds + num_qubits])
    rhs_inds = np.concatenate([inds + num_qubits, inds])
    table[lhs_inds, :] = table[rhs_inds, :]

    # Apply table
    table = np.mod(np.matmul(table1, table), 2).astype(np.bool)

    # Generate random phases
    phase = random_state.randint(2, size=2 * num_qubits).astype(np.bool)
    return Clifford(StabilizerTable(table, phase))


def sample_qmallows(n, rng=None):
    """Sample from the quantum Mallows distribution"""

    if rng is None:
        rng = np.random

    # Hadmard layer
    had = np.zeros(n, dtype=np.bool)

    # Permutation layer
    perm = np.zeros(n, dtype=int)

    inds = list(range(n))
    for i in range(n):
        m = n - i
        r = rng.uniform(0, 1)

        index = int(2 * m - np.ceil(np.log2(r * (4 ** m - 1) + 1)))
        had[i] = index < m
        if index < m:
            k = index
        else:
            k = 2 * m - index - 1
        perm[i] = inds[k]
        del inds[k]
    return had, perm
