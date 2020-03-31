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
from qiskit.exceptions import QiskitError

from .statevector import Statevector
from .densitymatrix import DensityMatrix
from qiskit.quantum_info.operators.random import random_unitary


def random_statevector(dims, seed=None):
    """Generator a random Statevector.

    The statevector is sampled from the uniform (Haar) measure.

    Args:
        dims (int or tuple): the dimensions of the state.
        seed (int): Optional. Set a fixed seed for RNG.

    Returns:
        Statevector: the random statevector.
    """
    dim = np.product(dims)
    rng = np.random.RandomState(seed)
    # Random array over interval (0, 1]
    x = rng.rand(dim)
    x += x == 0
    x = -np.log(x)
    sumx = sum(x)
    phases = rng.rand(dim) * 2.0 * np.pi
    return Statevector(np.sqrt(x / sumx) * np.exp(1j * phases), dims=dims)


def random_state(dim, seed=None):
    """
    Return a random quantum state from the uniform (Haar) measure on
    state space.

    Args:
        dim (int): the dim of the state space
        seed (int): Optional. To set a random seed.

    Returns:
        ndarray:  state(2**num) a random quantum state.
    """
    warnings.warn(
        'The `random_state` function is deprecated as of 0.13.0,'
        ' and will be removed no earlier than 3 months after that '
        'release date. You should use the `random_statevector`'
        ' function instead.', DeprecationWarning, stacklevel=2)
    return random_statevector(dim).data


def random_density_matrix(dims, rank=None, method='Hilbert-Schmidt', seed=None):
    """Generator a random DensityMatrix

    Args:
        dims (int or tuple): the dimensions of the DensityMatrix.
        rank (int or None): Optional, the rank of the density matrix.
                            The default value is full-rank.
        method (string): the method to use.
            'Hilbert-Schmidt': sample from the Hilbert-Schmidt metric.
            'Bures': sample from the Bures metric.
        seed (int): Optional. Set a fixed seed for RNG.

    Returns:
        DensityMatrix: the random density matrix.

    Raises:
        QiskitError: if the method is not valid.
    """
    if method == 'Hilbert-Schmidt':
        return _random_density_hs(dims, rank, seed)
    elif method == 'Bures':
        return _random_density_bures(dims, rank, seed)
    raise QiskitError('Error: unrecognized method {}'.format(method))


def _ginibre_matrix(nrow, ncol=None, seed=None):
    """Return a normally distributed complex random matrix.

    Args:
        nrow (int): number of rows in output matrix.
        ncol (int): number of columns in output matrix.
        seed (int): Optional. To set a random seed.

    Returns:
        ndarray: A complex rectangular matrix where each real and imaginary
            entry is sampled from the normal distribution.
    """
    if ncol is None:
        ncol = nrow
    rng = np.random.RandomState(seed)

    ginibre = rng.normal(size=(nrow, ncol)) + rng.normal(size=(nrow, ncol)) * 1j
    return ginibre


def _random_density_hs(dims, rank=None, seed=None):
    """
    Generate a random density matrix from the Hilbert-Schmidt metric.

    Args:
        length (int or tuple): the dimensions of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: rho (N,N)  a density matrix.
    """
    ginibre = _ginibre_matrix(np.product(dims), rank, seed)
    ginibre = ginibre.dot(ginibre.conj().T)
    return DensityMatrix(ginibre / np.trace(ginibre), dims=dims)


def _random_density_bures(dims, rank=None, seed=None):
    """Generate a random density matrix from the Bures metric.

    Args:
        length (int or tuple): the length of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: rho (N,N) a density matrix.
    """
    dim = np.product(dims)
    density = np.eye(dim) + random_unitary(dim, seed=seed * 2).data
    ginibre = density.dot(_ginibre_matrix(dim, rank, seed))
    ginibre = ginibre.dot(ginibre.conj().T)
    return DensityMatrix(ginibre / np.trace(ginibre), dims=dims)
