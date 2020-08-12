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
from numpy.random import default_rng
from scipy import stats

from qiskit.quantum_info.operators import Operator, Stinespring
from qiskit.exceptions import QiskitError

# pylint: disable=unused-import
from .symplectic.random import random_clifford
from .symplectic.random import random_pauli_table
from .symplectic.random import random_stabilizer_table


def random_unitary(dims, seed=None):
    """Return a random unitary Operator.

    The operator is sampled from the unitary Haar measure.

    Args:
        dims (int or tuple): the input dimensions of the Operator.
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        Operator: a unitary operator.
    """
    if seed is None:
        random_state = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        random_state = seed
    else:
        random_state = default_rng(seed)

    dim = np.product(dims)
    mat = stats.unitary_group.rvs(dim, random_state=random_state)
    return Operator(mat, input_dims=dims, output_dims=dims)


def random_hermitian(dims, traceless=False, seed=None):
    """Return a random hermitian Operator.

    The operator is sampled from Gaussian Unitary Ensemble.

    Args:
        dims (int or tuple): the input dimension of the Operator.
        traceless (bool): Optional. If True subtract diagonal entries to
                          return a traceless hermitian operator
                          (Default: False).
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        Operator: a Hermitian operator.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)

    # Total dimension
    dim = np.product(dims)

    if traceless:
        mat = np.zeros((dim, dim), dtype=complex)
    else:
        # Generate diagonal part of matrix for Gaussian N(0, 1)
        mat = np.diag(stats.norm.rvs(
            scale=1, size=dim, random_state=rng).astype(complex))

    # Generate lower triangular values from Gaussian N(0, 0.5)
    num_tril = (dim * (dim - 1)) // 2
    real_tril = stats.norm.rvs(
        scale=0.5, size=num_tril, random_state=rng)
    imag_tril = stats.norm.rvs(
        scale=0.5, size=num_tril, random_state=rng)
    # Get lower triangular indicies
    rows, cols = np.tril_indices(dim, -1)
    mat[(rows, cols)] = real_tril + 1j * imag_tril
    mat[(cols, rows)] = real_tril - 1j * imag_tril
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
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        Stinespring: a quantum channel operator.

    Raises:
        QiskitError: if rank or dimensions are invalid.
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
        raise QiskitError("Rank {} must be greater than 0.".format(rank))

    # Generate a random unitary matrix
    unitary = stats.unitary_group.rvs(
        max(rank * d_out, d_in), random_state=seed)

    # Truncate columns to produce an isometry
    return Stinespring(
        unitary[:, :d_in], input_dims=input_dims, output_dims=output_dims)
