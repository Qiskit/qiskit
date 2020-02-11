# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=len-as-condition,unsubscriptable-object

"""
Predicates for operators.
"""

import numpy as np

ATOL_DEFAULT = 1e-8
RTOL_DEFAULT = 1e-5


def matrix_equal(mat1,
                 mat2,
                 ignore_phase=False,
                 rtol=RTOL_DEFAULT,
                 atol=ATOL_DEFAULT):
    """Test if two arrays are equal.

    The final comparison is implemented using Numpy.allclose. See its
    documentation for additional information on tolerance parameters.

    If ignore_phase is True both matrices will be multiplied by
    exp(-1j * theta) where `theta` is the first nphase for a
    first non-zero matrix element `|a| * exp(1j * theta)`.

    Args:
        mat1 (matrix_like): a matrix
        mat2 (matrix_like): a matrix
        ignore_phase (bool): ignore complex-phase differences between
            matrices [Default: False]
        rtol (double): the relative tolerance parameter [Default {}].
        atol (double): the absolute tolerance parameter [Default {}].

    Returns:
        bool: True if the matrices are equal or False otherwise.
    """.format(RTOL_DEFAULT, ATOL_DEFAULT)

    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    mat1 = np.array(mat1)
    mat2 = np.array(mat2)
    if mat1.shape != mat2.shape:
        return False
    if ignore_phase:
        # Get phase of first non-zero entry of mat1 and mat2
        # and multiply all entries by the conjugate
        phases1 = np.angle(mat1[abs(mat1) > atol].ravel(order='F'))
        if len(phases1) > 0:
            mat1 = np.exp(-1j * phases1[0]) * mat1
        phases2 = np.angle(mat2[abs(mat2) > atol].ravel(order='F'))
        if len(phases2) > 0:
            mat2 = np.exp(-1j * phases2[0]) * mat2
    return np.allclose(mat1, mat2, rtol=rtol, atol=atol)


def is_square_matrix(mat):
    """Test if an array is a square matrix."""
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    shape = mat.shape
    return shape[0] == shape[1]


def is_diagonal_matrix(mat, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT):
    """Test if an array is a diagonal matrix"""
    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    return np.allclose(mat, np.diag(np.diagonal(mat)), rtol=rtol, atol=atol)


def is_symmetric_matrix(op, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT):
    """Test if an array is a symmetric matrix"""
    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    mat = np.array(op)
    if mat.ndim != 2:
        return False
    return np.allclose(mat, mat.T, rtol=rtol, atol=atol)


def is_hermitian_matrix(mat, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT):
    """Test if an array is a Hermitian matrix"""
    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    return np.allclose(mat, np.conj(mat.T), rtol=rtol, atol=atol)


def is_positive_semidefinite_matrix(mat, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT):
    """Test if a matrix is positive semidefinite"""
    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    if not is_hermitian_matrix(mat, rtol=rtol, atol=atol):
        return False
    # Check eigenvalues are all positive
    vals = np.linalg.eigvalsh(mat)
    for v in vals:
        if v < -atol:
            return False
    return True


def is_identity_matrix(mat,
                       ignore_phase=False,
                       rtol=RTOL_DEFAULT,
                       atol=ATOL_DEFAULT):
    """Test if an array is an identity matrix."""
    if atol is None:
        atol = ATOL_DEFAULT
    if rtol is None:
        rtol = RTOL_DEFAULT
    mat = np.array(mat)
    if mat.ndim != 2:
        return False
    if ignore_phase:
        # If the matrix is equal to an identity up to a phase, we can
        # remove the phase by multiplying each entry by the complex
        # conjugate of the phase of the [0, 0] entry.
        theta = np.angle(mat[0, 0])
        mat = np.exp(-1j * theta) * mat
    # Check if square identity
    iden = np.eye(len(mat))
    return np.allclose(mat, iden, rtol=rtol, atol=atol)


def is_unitary_matrix(mat, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT):
    """Test if an array is a unitary matrix."""
    mat = np.array(mat)
    # Compute A^dagger.A and see if it is identity matrix
    mat = np.conj(mat.T).dot(mat)
    return is_identity_matrix(mat, ignore_phase=False, rtol=rtol, atol=atol)


def is_isometry(mat, rtol=RTOL_DEFAULT, atol=ATOL_DEFAULT):
    """Test if an array is an isometry."""
    mat = np.array(mat)
    # Compute A^dagger.A and see if it is identity matrix
    iden = np.eye(mat.shape[1])
    mat = np.conj(mat.T).dot(mat)
    return np.allclose(mat, iden, rtol=rtol, atol=atol)
