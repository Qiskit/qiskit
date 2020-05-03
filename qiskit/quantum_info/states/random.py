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

"""
Random state generation.
"""

import warnings
import numpy as np
from numpy.random import default_rng

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.random import random_unitary
from .statevector import Statevector
from .densitymatrix import DensityMatrix


def random_statevector(dims, seed=None):
    """Generator a random Statevector.

    The statevector is sampled from the uniform (Haar) measure.

    Args:
        dims (int or tuple): the dimensions of the state.
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        Statevector: the random statevector.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)

    dim = np.product(dims)

    # Random array over interval (0, 1]
    x = rng.random(dim)
    x += x == 0
    x = -np.log(x)
    sumx = sum(x)
    phases = rng.random(dim) * 2.0 * np.pi
    return Statevector(np.sqrt(x / sumx) * np.exp(1j * phases), dims=dims)


def random_state(dim, seed=None):
    """
    DEPRECATED Return a random quantum state.

    Args:
        dim (int): the dim of the state space
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        ndarray:  state(2**num) a random quantum state.
    """
    warnings.warn(
        'The `random_state` function is deprecated as of 0.13.0,'
        ' and will be removed no earlier than 3 months after that '
        'release date. You should use the `random_statevector`'
        ' function instead.', DeprecationWarning, stacklevel=2)
    return random_statevector(dim, seed=seed).data


def random_density_matrix(dims, rank=None, method='Hilbert-Schmidt',
                          seed=None):
    """Generator a random DensityMatrix.

    Args:
        dims (int or tuple): the dimensions of the DensityMatrix.
        rank (int or None): Optional, the rank of the density matrix.
                            The default value is full-rank.
        method (string): Optional. The method to use.
            'Hilbert-Schmidt': (Default) sample from the Hilbert-Schmidt metric.
            'Bures': sample from the Bures metric.
        seed (int or np.random.Generator): Optional. Set a fixed seed or
                                           generator for RNG.

    Returns:
        DensityMatrix: the random density matrix.

    Raises:
        QiskitError: if the method is not valid.
    """
    # Flatten dimensions
    dim = np.product(dims)
    if rank is None:
        rank = dim  # Use full rank

    if method == 'Hilbert-Schmidt':
        rho = _random_density_hs(dim, rank, seed)
    elif method == 'Bures':
        rho = _random_density_bures(dim, rank, seed)
    else:
        raise QiskitError('Error: unrecognized method {}'.format(method))
    return DensityMatrix(rho, dims=dims)


def _ginibre_matrix(nrow, ncol, seed):
    """Return a normally distributed complex random matrix.

    Args:
        nrow (int): number of rows in output matrix.
        ncol (int): number of columns in output matrix.
        seed(int or np.random.Generator): default rng.

    Returns:
        ndarray: A complex rectangular matrix where each real and imaginary
            entry is sampled from the normal distribution.
    """
    if seed is None:
        rng = np.random.default_rng()
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = default_rng(seed)

    ginibre = rng.normal(
        size=(nrow, ncol)) + rng.normal(size=(nrow, ncol)) * 1j
    return ginibre


def _random_density_hs(dim, rank, seed):
    """
    Generate a random density matrix from the Hilbert-Schmidt metric.

    Args:
        dim (int): the dimensions of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int or np.random.Generator): default rng.

    Returns:
        ndarray: rho (N,N)  a density matrix.
    """
    mat = _ginibre_matrix(dim, rank, seed)
    mat = mat.dot(mat.conj().T)
    return mat / np.trace(mat)


def _random_density_bures(dim, rank, seed):
    """Generate a random density matrix from the Bures metric.

    Args:
        dim (int): the length of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int or np.random.Generator): default rng.

    Returns:
        ndarray: rho (N,N) a density matrix.
    """
    density = np.eye(dim) + random_unitary(dim, seed=seed).data
    mat = density.dot(_ginibre_matrix(dim, rank, seed))
    mat = mat.dot(mat.conj().T)
    return mat / np.trace(mat)
