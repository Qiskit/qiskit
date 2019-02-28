# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import numpy as np

ATOL_DEFAULT = 1e-8


def is_square_matrix(op):
    """Test if an array is a square matrix."""
    mat = np.array(op)
    if mat.ndim != 2:
        return False
    shape = mat.shape
    return shape[0] == shape[1]


def is_diagonal_matrix(op, atol=ATOL_DEFAULT):
    """Test if an array is a diagonal matrix"""
    if atol is None:
        atol = ATOL_DEFAULT
    mat = np.array(op)
    if mat.ndim != 2:
        return False
    return np.allclose(mat, np.diag(np.diagonal(mat)), atol=atol)


def is_symmetric_matrix(op, atol=ATOL_DEFAULT):
    """Test if an array is a symmetrix matrix"""
    if atol is None:
        atol = ATOL_DEFAULT
    mat = np.array(op)
    if mat.ndim != 2:
        return False
    return np.allclose(mat, mat.T, atol=atol)


def is_hermitian_matrix(op, atol=ATOL_DEFAULT):
    """Test if an array is a Hermitian matrix"""
    if atol is None:
        atol = ATOL_DEFAULT
    mat = np.array(op)
    if mat.ndim != 2:
        return False
    return np.allclose(mat, np.conj(mat.T), atol=atol)


def is_identity_matrix(op, ignore_phase=False, atol=ATOL_DEFAULT):
    """Test if an array is an identity matrix."""
    if atol is None:
        atol = ATOL_DEFAULT
    mat = np.array(op)
    if mat.ndim != 2:
        return False
    if ignore_phase:
        # If the matrix is equal to an identity up to a phase, we can
        # remove the phase by multiplying each entry by the complex
        # conjugate of the [0, 0] entry.
        mat *= np.conj(mat[0, 0])
    # Check if square identity
    iden = np.eye(len(mat))
    return np.allclose(mat, iden, atol=atol)


def matrix_equal(a, b, ignore_phase=False, atol=ATOL_DEFAULT):
    """Test if two arrays are equal"""
    if atol is None:
        atol = ATOL_DEFAULT
    mat1 = np.array(a)
    mat2 = np.array(b)
    if mat1.shape != mat2.shape:
        return False
    if ignore_phase:
        # Get phase of first non-zero entry of mat1 and mat2
        # and multiply all entries by the conjugate
        phases1 = np.angle(mat1[mat1 != 0].ravel(order='F'))
        if len(phases1) > 0:
            mat1 *= np.exp(-1j * phases1[0])
        phases2 = np.angle(mat2[mat2 != 0].ravel(order='F'))
        if len(phases2) > 0:
            mat2 *= np.exp(-1j * phases2[0])
    return np.allclose(mat1, mat2, atol=atol)


def is_unitary_matrix(op, atol=ATOL_DEFAULT):
    """Test if an array is a unitary matrix."""
    if atol is None:
        atol = ATOL_DEFAULT
    mat = np.array(op)
    # Compute A^dagger.A and see if it is identity matrix
    mat = np.conj(mat.T).dot(mat)
    return is_identity_matrix(mat, ignore_phase=False, atol=atol)
