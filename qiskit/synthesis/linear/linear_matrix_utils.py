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

# pylint: disable=unused-import
from qiskit._accelerate.synthesis.linear import (
    gauss_elimination,
    gauss_elimination_with_perm,
    compute_rank_after_gauss_elim,
    calc_inverse_matrix,
    binary_matmul,
    row_op,
    col_op,
)


def check_invertible_binary_matrix(mat: np.ndarray):
    """Check that a binary matrix is invertible.

    Args:
        mat: a binary matrix.

    Returns:
        bool: True if mat in invertible and False otherwise.
    """
    if len(mat.shape) != 2 or mat.shape[0] != mat.shape[1]:
        return False

    rank = compute_rank(mat)
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
        rank = compute_rank(mat)
    return mat


def compute_rank(mat):
    """Given a boolean matrix A computes its rank"""
    mat = mat.astype(bool)
    gauss_elimination(mat)
    return np.sum(mat.any(axis=1))
