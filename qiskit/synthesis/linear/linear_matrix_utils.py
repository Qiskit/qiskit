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

from typing import Optional, Union
import numpy as np
from qiskit.exceptions import QiskitError


def check_invertible_binary_matrix(mat: np.ndarray):
    """Check that a binary matrix is invertible.

    Args:
        mat: a binary matrix.

    Returns:
        bool: True if mat in invertible and False otherwise.
    """
    if len(mat.shape) != 2 or mat.shape[0] != mat.shape[1]:
        return False

    rank = _compute_rank(mat)
    return rank == mat.shape[0]


def random_invertible_binary_matrix(
    num_qubits: int, seed: Optional[Union[np.random.Generator, int]] = None
):
    """Generates a random invertible n x n binary matrix.

    Args:
        num_qubits: the matrix size.
        seed: a random seed.

    Returns:
        np.ndarray: A random invertible binary matrix of size num_qubits.
    """
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    rank = 0
    while rank != num_qubits:
        mat = rng.integers(2, size=(num_qubits, num_qubits))
        rank = _compute_rank(mat)
    return mat


def _gauss_elimination(mat, ncols=None, full_elim=False):
    """Gauss elimination of a matrix mat with m rows and n columns.
    If full_elim = True, it allows full elimination of mat[:, 0 : ncols]
    Returns the matrix mat."""

    mat, _ = _gauss_elimination_with_perm(mat, ncols, full_elim)
    return mat


def _gauss_elimination_with_perm(mat, ncols=None, full_elim=False):
    """Gauss elimination of a matrix mat with m rows and n columns.
    If full_elim = True, it allows full elimination of mat[:, 0 : ncols]
    Returns the matrix mat, and the permutation perm that was done on the rows during the process.
    perm[0 : rank] represents the indices of linearly independent rows in the original matrix."""

    # Treat the matrix A as containing integer values
    mat = np.array(mat, dtype=int, copy=True)

    m = mat.shape[0]  # no. of rows
    n = mat.shape[1]  # no. of columns
    if ncols is not None:
        n = min(n, ncols)  # no. of active columns

    perm = np.array(range(m))  # permutation on the rows

    r = 0  # current rank
    k = 0  # current pivot column
    while (r < m) and (k < n):
        is_non_zero = False
        new_r = r
        for j in range(k, n):
            for i in range(r, m):
                if mat[i][j]:
                    is_non_zero = True
                    k = j
                    new_r = i
                    break
            if is_non_zero:
                break
        if not is_non_zero:
            return mat, perm  # A is in the canonical form

        if new_r != r:
            mat[[r, new_r]] = mat[[new_r, r]]
            perm[r], perm[new_r] = perm[new_r], perm[r]

        if full_elim:
            for i in range(0, r):
                if mat[i][k]:
                    mat[i] = mat[i] ^ mat[r]

        for i in range(r + 1, m):
            if mat[i][k]:
                mat[i] = mat[i] ^ mat[r]
        r += 1

    return mat, perm


def calc_inverse_matrix(mat: np.ndarray, verify: bool = False):
    """Given a square numpy(dtype=int) matrix mat, tries to compute its inverse.

    Args:
        mat: a boolean square matrix.
        verify: if True asserts that the multiplication of mat and its inverse is the identity matrix.

    Returns:
        np.ndarray: the inverse matrix.

    Raises:
         QiskitError: if the matrix is not square.
         QiskitError: if the matrix is not invertible.
    """

    if mat.shape[0] != mat.shape[1]:
        raise QiskitError("Matrix to invert is a non-square matrix.")

    n = mat.shape[0]
    # concatenate the matrix and identity
    mat1 = np.concatenate((mat, np.eye(n, dtype=int)), axis=1)
    mat1 = _gauss_elimination(mat1, None, full_elim=True)

    r = _compute_rank_after_gauss_elim(mat1[:, 0:n])

    if r < n:
        raise QiskitError("The matrix is not invertible.")

    matinv = mat1[:, n : 2 * n]

    if verify:
        mat2 = np.dot(mat, matinv) % 2
        assert np.array_equal(mat2, np.eye(n))

    return matinv


def _compute_rank_after_gauss_elim(mat):
    """Given a matrix A after Gaussian elimination, computes its rank
    (i.e. simply the number of nonzero rows)"""
    return np.sum(mat.any(axis=1))


def _compute_rank(mat):
    """Given a matrix A computes its rank"""
    mat = _gauss_elimination(mat)
    return np.sum(mat.any(axis=1))


def _row_op(mat, ctrl, trgt):
    # Perform ROW operation on a matrix mat
    mat[trgt] = mat[trgt] ^ mat[ctrl]


def _col_op(mat, ctrl, trgt):
    # Perform COL operation on a matrix mat
    mat[:, ctrl] = mat[:, trgt] ^ mat[:, ctrl]
