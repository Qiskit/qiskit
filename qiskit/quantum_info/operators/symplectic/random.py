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
from numpy.random import default_rng

from .clifford import Clifford
from .pauli import Pauli
from .pauli_list import PauliList
from .pauli_table import PauliTable
from .stabilizer_table import StabilizerTable


def random_pauli(num_qubits, group_phase=False, seed=None):
    """Return a random Pauli.

    Args:
        num_qubits (int): the number of qubits.
        group_phase (bool): Optional. If True generate random phase.
                            Otherwise the phase will be set so that the
                            Pauli coefficient is +1 (default: False).
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        Pauli: a random Pauli
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)
    z = rng.integers(2, size=num_qubits, dtype=bool)
    x = rng.integers(2, size=num_qubits, dtype=bool)
    phase = rng.integers(4) if group_phase else 0
    pauli = Pauli((z, x, phase))
    return pauli


def random_pauli_list(num_qubits, size=1, seed=None, phase=True):
    """Return a random PauliList.

    Args:
        num_qubits (int): the number of qubits.
        size (int): Optional. The length of the Pauli list (Default: 1).
        seed (int or np.random.Generator): Optional. Set a fixed seed or generator for RNG.
        phase (bool): If True the Pauli phases are randomized, otherwise the phases are fixed to 0.
                     [Default: True]

    Returns:
        PauliList: a random PauliList.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)

    z = rng.integers(2, size=(size, num_qubits)).astype(bool)
    x = rng.integers(2, size=(size, num_qubits)).astype(bool)
    if phase:
        _phase = rng.integers(4, size=(size))
        return PauliList.from_symplectic(z, x, _phase)
    return PauliList.from_symplectic(z, x)


def random_pauli_table(num_qubits, size=1, seed=None):
    """Return a random PauliTable.

    Args:
        num_qubits (int): the number of qubits.
        size (int): Optional. The number of rows of the table (Default: 1).
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        PauliTable: a random PauliTable.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)

    table = rng.integers(2, size=(size, 2 * num_qubits)).astype(bool)
    return PauliTable(table)


def random_stabilizer_table(num_qubits, size=1, seed=None):
    """Return a random StabilizerTable.

    Args:
        num_qubits (int): the number of qubits.
        size (int): Optional. The number of rows of the table (Default: 1).
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        PauliTable: a random StabilizerTable.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)

    table = rng.integers(2, size=(size, 2 * num_qubits)).astype(bool)
    phase = rng.integers(2, size=size).astype(bool)
    return StabilizerTable(table, phase)


def random_clifford(num_qubits, seed=None):
    """Return a random Clifford operator.

    The Clifford is sampled using the method of Reference [1].

    Args:
        num_qubits (int): the number of qubits for the Clifford
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        Clifford: a random Clifford operator.

    Reference:
        1. S. Bravyi and D. Maslov, *Hadamard-free circuits expose the
           structure of the Clifford group*.
           `arXiv:2003.09412 [quant-ph] <https://arxiv.org/abs/2003.09412>`_
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)

    had, perm = _sample_qmallows(num_qubits, rng)

    gamma1 = np.diag(rng.integers(2, size=num_qubits, dtype=np.int8))
    gamma2 = np.diag(rng.integers(2, size=num_qubits, dtype=np.int8))
    delta1 = np.eye(num_qubits, dtype=np.int8)
    delta2 = delta1.copy()

    _fill_tril(gamma1, rng, symmetric=True)
    _fill_tril(gamma2, rng, symmetric=True)
    _fill_tril(delta1, rng)
    _fill_tril(delta2, rng)

    # For large num_qubits numpy.inv function called below can
    # return invalid output leading to a non-symplectic Clifford
    # being generated. This can be prevented by manually forcing
    # block inversion of the matrix.
    block_inverse_threshold = 50

    # Compute stabilizer table
    zero = np.zeros((num_qubits, num_qubits), dtype=np.int8)
    prod1 = np.matmul(gamma1, delta1) % 2
    prod2 = np.matmul(gamma2, delta2) % 2
    inv1 = _inverse_tril(delta1, block_inverse_threshold).transpose()
    inv2 = _inverse_tril(delta2, block_inverse_threshold).transpose()
    table1 = np.block([[delta1, zero], [prod1, inv1]])
    table2 = np.block([[delta2, zero], [prod2, inv2]])

    # Apply qubit permutation
    table = table2[np.concatenate([perm, num_qubits + perm])]

    # Apply layer of Hadamards
    inds = had * np.arange(1, num_qubits + 1)
    inds = inds[inds > 0] - 1
    lhs_inds = np.concatenate([inds, inds + num_qubits])
    rhs_inds = np.concatenate([inds + num_qubits, inds])
    table[lhs_inds, :] = table[rhs_inds, :]

    # Apply table
    table = np.mod(np.matmul(table1, table), 2).astype(bool)

    # Generate random phases
    phase = rng.integers(2, size=2 * num_qubits).astype(bool)
    return Clifford(StabilizerTable(table, phase))


def _sample_qmallows(n, rng=None):
    """Sample from the quantum Mallows distribution"""

    if rng is None:
        rng = np.random.default_rng()

    # Hadmard layer
    had = np.zeros(n, dtype=bool)

    # Permutation layer
    perm = np.zeros(n, dtype=int)

    inds = list(range(n))
    for i in range(n):
        m = n - i
        eps = 4 ** (-m)
        r = rng.uniform(0, 1)
        index = -int(np.ceil(np.log2(r + (1 - r) * eps)))
        had[i] = index < m
        if index < m:
            k = index
        else:
            k = 2 * m - index - 1
        perm[i] = inds[k]
        del inds[k]
    return had, perm


def _fill_tril(mat, rng, symmetric=False):
    """Add symmetric random ints to off diagonals"""
    dim = mat.shape[0]
    # Optimized for low dimensions
    if dim == 1:
        return

    if dim <= 4:
        mat[1, 0] = rng.integers(2, dtype=np.int8)
        if symmetric:
            mat[0, 1] = mat[1, 0]
        if dim > 2:
            mat[2, 0] = rng.integers(2, dtype=np.int8)
            mat[2, 1] = rng.integers(2, dtype=np.int8)
            if symmetric:
                mat[0, 2] = mat[2, 0]
                mat[1, 2] = mat[2, 1]
        if dim > 3:
            mat[3, 0] = rng.integers(2, dtype=np.int8)
            mat[3, 1] = rng.integers(2, dtype=np.int8)
            mat[3, 2] = rng.integers(2, dtype=np.int8)
            if symmetric:
                mat[0, 3] = mat[3, 0]
                mat[1, 3] = mat[3, 1]
                mat[2, 3] = mat[3, 2]
        return

    # Use numpy indices for larger dimensions
    rows, cols = np.tril_indices(dim, -1)
    vals = rng.integers(2, size=rows.size, dtype=np.int8)
    mat[(rows, cols)] = vals
    if symmetric:
        mat[(cols, rows)] = vals


def _inverse_tril(mat, block_inverse_threshold):
    """Invert a lower-triangular matrix with unit diagonal."""
    # Optimized inversion function for low dimensions
    dim = mat.shape[0]

    if dim <= 2:
        return mat

    if dim <= 5:
        inv = mat.copy()
        inv[2, 0] = mat[2, 0] ^ (mat[1, 0] & mat[2, 1])
        if dim > 3:
            inv[3, 1] = mat[3, 1] ^ (mat[2, 1] & mat[3, 2])
            inv[3, 0] = mat[3, 0] ^ (mat[3, 2] & mat[2, 0]) ^ (mat[1, 0] & inv[3, 1])
        if dim > 4:
            inv[4, 2] = (mat[4, 2] ^ (mat[3, 2] & mat[4, 3])) & 1
            inv[4, 1] = mat[4, 1] ^ (mat[4, 3] & mat[3, 1]) ^ (mat[2, 1] & inv[4, 2])
            inv[4, 0] = (
                mat[4, 0]
                ^ (mat[1, 0] & inv[4, 1])
                ^ (mat[2, 0] & inv[4, 2])
                ^ (mat[3, 0] & mat[4, 3])
            )
        return inv % 2

    # For higher dimensions we use Numpy's inverse function
    # however this function tends to fail and result in a non-symplectic
    # final matrix if n is too large.
    if dim <= block_inverse_threshold:
        return np.linalg.inv(mat).astype(np.int8) % 2

    # For very large matrices  we divide the matrix into 4 blocks of
    # roughly equal size and use the analytic formula for the inverse
    # of a block lower-triangular matrix:
    # inv([[A, 0],[C, D]]) = [[inv(A), 0], [inv(D).C.inv(A), inv(D)]]
    # call the inverse function recursively to compute inv(A) and invD

    dim1 = dim // 2
    mat_a = _inverse_tril(mat[0:dim1, 0:dim1], block_inverse_threshold)
    mat_d = _inverse_tril(mat[dim1:dim, dim1:dim], block_inverse_threshold)
    mat_c = np.matmul(np.matmul(mat_d, mat[dim1:dim, 0:dim1]), mat_a)
    inv = np.block([[mat_a, np.zeros((dim1, dim - dim1), dtype=int)], [mat_c, mat_d]])
    return inv % 2
