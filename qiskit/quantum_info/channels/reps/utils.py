# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from itertools import repeat
import numpy as np


def reshuffle(a, shape=None):
    """
    Reshuffle the indicies of a bipartite matrix.

    This is the transformation A[ij,kl] -> A[lj,ki].

    Args:
        a (matrix_like): a bipartite matrix
        shape (list): length four list of bipartite dimensions.

    Additional Information:
        If shape is none it is assumed bipartite dimensions are equal.

    Returns:
        np.array: the reshuffled matrix.

    Raises:
        ValueError: if shape is not length-4 or shape dimensions are invalid.
    """

    # Get matrix shape
    mat = np.array(a)
    dl, dr = mat.shape

    # Get bipartite dimensions
    if shape is None:
        dl0 = int(np.sqrt(dl))
        dl1 = dl0
        dr0 = int(np.sqrt(dr))
        dr1 = dr0
    else:
        if len(shape) != 4:
            raise ValueError("Invalid bipartite matrix shape. Must be length 4.")
        dl0, dl1, dr0, dr1 = tuple(shape)
    if dr0 * dr1 != dr or dl0 * dl1 != dl:
        raise ValueError('Invalid Dimensions')

    # Reshuffle indicies
    mat = mat.reshape(dl0, dl1, dr0, dr1)
    return mat.transpose((3, 1, 2, 0)).reshape(dl1 * dr1, dl0 * dr0)


def reravel(a, b, shape_a, shape_b):
    """
    shape_a and shape_b length-4 lists.
    """
    data = np.kron(a, b)
    # Reshuffle indicies
    left_dims = shape_a[:2] + shape_b[:2]
    right_dims = shape_a[2:] + shape_b[2:]
    tensor_shape = left_dims + right_dims
    final_shape = (np.product(left_dims), np.product(right_dims))
    data = np.reshape(np.transpose(np.reshape(data, tensor_shape),
                                   (0, 2, 1, 3, 4, 6, 5, 7)), final_shape)
    return data


def col_to_basis_matrix(basis, repeats=1):
    full_basis = basis
    for _ in repeat(None, repeats - 1):
        full_basis = [np.kron(a, b) for a in basis for b in full_basis]
    basis_length = len(full_basis)
    left_dim, right_dim = full_basis[0].shape
    return np.array(full_basis).conj().reshape(basis_length,
                                               left_dim * right_dim, order='F')


def pauli_basis(num_qubits=1, normalized=False):
    # TODO: replace pauli basis with Pauli class objects when they are stable
    single_basis = [
        np.eye(2, dtype=complex),
        np.array([[0, 1], [1, 0]], dtype=complex),
        np.array([[0, -1j], [1j, 0]], dtype=complex),
        np.array([[1, 0], [0, -1]], dtype=complex)
    ]
    if normalized is True:
        single_basis = [p / np.sqrt(2) for p in single_basis]
    if num_qubits == 1:
        return single_basis
    else:
        basis = single_basis
        for _ in repeat(None, num_qubits - 1):
            basis = [np.kron(a, b) for a in single_basis for b in basis]
        return basis
