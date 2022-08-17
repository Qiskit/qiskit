# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for handling binary matrices."""

import numpy as np
from qiskit.exceptions import QiskitError


def _calc_rank(A):
    # This function takes as input a binary square matrix A
    # Returns the rank of A over the binary field F_2
    n = A.shape[1]
    X = np.identity(n, dtype=int)

    for i in range(n):
        y = np.dot(A[i, :], X) % 2
        not_y = (y + 1) % 2
        good = X[:, np.nonzero(not_y)]
        good = good[:, 0, :]
        bad = X[:, np.nonzero(y)]
        bad = bad[:, 0, :]
        if bad.shape[1] > 0:
            bad = np.add(bad, np.roll(bad, 1, axis=1))
            bad = bad % 2
            bad = np.delete(bad, 0, axis=1)
            X = np.concatenate((good, bad), axis=1)
    # now columns of X span the binary null-space of A
    return n - X.shape[1]


def random_invertible_binary_matrix(num_qubits, seed=None):
    """Generates a random invertible n x n binary matrix."""
    # This code is adapted from random_cnotdihedral
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    rank = 0
    while rank != num_qubits:
        mat = rng.integers(2, size=(num_qubits, num_qubits))
        rank = _calc_rank(mat)
    return mat


def _gauss_elimination(A, ncols=None, full_elim=False):
    """Gauss elimination of a matrix A with m rows and n columns.
    If full_elim = True, it allows full elimination of A[:, 0 : ncols]
    Returns the matrix A and the permutation perm that was done on the rows
    during the process."""

    # Treat the matrix A as containing integer values
    A = np.array(A, dtype=int, copy=False)

    m = A.shape[0]  # no. of rows
    n = A.shape[1]  # no. of columns
    if ncols is not None:
        n = min(n, ncols)  # no. of active columns

    perm = np.array(range(n))  # permutation on the rows

    r = 0  # current rank
    k = 0  # current pivot column
    while (r < m) and (k < n):
        isNZ = False
        new_r = r
        for j in range(k, n):
            for i in range(r, m):
                if A[i][j]:
                    isNZ = True
                    k = j
                    new_r = i
                    break
            if isNZ:
                break
        if not isNZ:
            return A, perm  # A is in the canonical form

        if new_r != r:
            tmp = A[new_r].copy()
            A[new_r] = A[r]
            A[r] = tmp
            perm[r], perm[new_r] = perm[new_r], perm[r]

        if full_elim:
            for i in range(0, r):
                if A[i][k]:
                    A[i] = A[i] ^ A[r]

        for i in range(r + 1, m):
            if A[i][k]:
                A[i] = A[i] ^ A[r]
        r += 1

    return A, perm


def calc_inverse_matrix(A, verify=False):
    """Given a square numpy(dtype=int) matrix A, tries to compute its inverse.
    Returns None if the matrix is not invertible, and the inverse otherwise."""

    if A.shape[0] != A.shape[1]:
        raise QiskitError("Matrix to invert is a non-square matrix.")

    n = A.shape[0]
    # concatenate the matrix and identity
    B = np.concatenate((A, np.eye(n, dtype=int)), axis=1)
    B, _ = _gauss_elimination(B, None, full_elim=True)

    r = _compute_rank_after_gauss_elim(B[:, 0:n])

    if r < n:
        raise QiskitError("The matrix is not invertible")

    else:
        C = B[:, n : 2 * n]

        if verify:
            D = np.dot(A, C) % 2
            assert np.array_equal(D, np.eye(n))
        return C


def _compute_rank_after_gauss_elim(A):
    """Given a matrix A after Gaussian elimination, computes its rank
    (i.e. simply the number of nonzero rows)"""
    return np.sum(A.any(axis=1))
