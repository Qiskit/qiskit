# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Methods to create random operators.
"""

import numpy as np
from scipy import stats

from qiskit.quantum_info.operators import Operator, Stinespring
from qiskit.exceptions import QiskitError


def random_unitary(dims, seed=None):
    """Return a random unitary Operator.

    The operator is sampled from the unitary Haar measure.

    Args:
        dims (int or tuple): the input dimensions of the Operator.
        seed (int): Optional. To set a random seed.

    Returns:
        Operator: a unitary operator.
    """
    dim = np.product(dims)
    mat = stats.unitary_group.rvs(dim, random_state=seed)
    return Operator(mat, input_dims=dims, output_dims=dims)


def random_hermitian(dims, traceless=False, seed=None):
    """Return a random hermitian Operator.

    The operator is sampled from Gaussian Unitary Ensemble.

    Args:
        dims (int or tuple): the input dimension of the Operator.
        traceless (bool): Optional. If True subtract diagonal entries to
                          return a traceless hermitian operator
                          (Default: False).
        seed (int): Optional. To set a random seed.

    Returns:
        Operator: a Hermitian operator.
    """
    # Total dimension
    dim = np.product(dims)

    if traceless:
        mat = np.zeros((dim, dim), dtype=complex)
    else:
        # Generate diagonal part of matrix for Gaussian N(0, 1)
        mat = np.diag(stats.norm.rvs(scale=1, size=dim).astype(complex))

    # Generate off diagonal parts from Gaussian N(0, 0.5)
    # We will construct the upper triangular real part from
    # the upper triangular part of the generated matrix, and
    # the upper triangular imaginary part from the lower
    # triangular part of the generated matrix.
    offdiag = stats.norm.rvs(scale=0.5, size=(dim, dim))
    offd_re = np.triu(offdiag, 1)
    offd_im = np.tril(offdiag, -1)
    mat += offd_re + offd_re.T + 1j * (offd_im - offd_im.T)

    return Operator(mat, input_dims=dims, output_dims=dims)


def random_quantum_channel(input_dims=None,
                           output_dims=None,
                           rank=None,
                           seed=None):
    """Return a random CPTP quantum channel.

    This constructs the Stinespring operator for the quantum channel by
    sampling a random isometry from the unitary Haar measure.

    Args:
        input_dims (int or tuple): the input dimension of the channel.
        output_dims (int or tuple): the input dimension of the channel.
        rank (int): Optional. The rank of the quantum channel Choi-matrix.
        seed (int): Optional. To set a random seed.

    Returns:
        Stinespring: a quantum channel operator.
    """
    # Determine total input and output dimensions
    if input_dims is None and output_dims is None:
        raise QiskitError(
            'No dimensions specified: input_dims and output_dims cannot'
            ' both be None.')
    if input_dims is None:
        input_dims = output_dims
    elif output_dims is None:
        output_dims = input_dims

    d_in = np.product(input_dims)
    d_out = np.product(output_dims)

    # If rank is not specified set to the maximum rank for the
    # Choi matrix (input_dim * output_dim)
    if rank is None or rank > d_in * d_out:
        rank = d_in * d_out
    if rank < 1:
        raise Exception("Rank {} must be greater than 0.".format(rank))

    # Generate a random unitary matrix
    unitary = stats.unitary_group.rvs(
        max(rank * d_out, d_in), random_state=seed)

    # Truncate columns to produce an isometry
    return Stinespring(
        unitary[:, :d_in], input_dims=input_dims, output_dims=output_dims)
