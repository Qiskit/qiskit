# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,invalid-sequence-index
# pylint: disable=unsupported-assignment-operation

"""
Methods to create random unitaries, states, etc.
"""

import math
import numpy as np

from qiskit.quantum_info.operators import Unitary
from qiskit.exceptions import QiskitError


# TODO: return a QuantumState object
def random_state(dim, seed=None):
    """
    Return a random quantum state from the uniform (Haar) measure on
    state space.

    Args:
        dim (int): the dim of the state spaxe
        seed (int): Optional. To set a random seed.

    Returns:
        ndarray:  state(2**num) a random quantum state.
    """
    if seed is None:
        seed = np.random.randint(0, np.iinfo(np.int32).max)
    rng = np.random.RandomState(seed)
    # Random array over interval (0, 1]
    x = rng.rand(dim)
    x += x == 0
    x = -np.log(x)
    sumx = sum(x)
    phases = rng.rand(dim)*2.0*np.pi
    return np.sqrt(x/sumx)*np.exp(1j*phases)


def random_unitary(dim, seed=None):
    """
    Return a random dim x dim Unitary from the Haar measure.

    Args:
        dim (int): the dim of the state space.
        seed (int): Optional. To set a random seed.

    Returns:
        Unitary: (dim, dim) unitary operator.

    Raises:
        QiskitError: if dim is not a positive power of 2.
    """
    if dim == 0 or not math.log2(dim).is_integer():
        raise QiskitError("Desired statevector length not a positive power of 2.")
    matrix = np.zeros([dim, dim], dtype=complex)
    for j in range(dim):
        if j == 0:
            a = random_state(dim, seed)
        else:
            a = random_state(dim)
        matrix[:, j] = np.copy(a)
        # Grahm-Schmidt Orthogonalize
        i = j-1
        while i >= 0:
            dc = np.vdot(matrix[:, i], a)
            matrix[:, j] = matrix[:, j]-dc*matrix[:, i]
            i = i - 1
        # normalize
        matrix[:, j] = matrix[:, j] * (1.0 / np.sqrt(np.vdot(matrix[:, j], matrix[:, j])))
    return Unitary(matrix)


# TODO: return a DensityMatrix object.
def random_density_matrix(length, rank=None, method='Hilbert-Schmidt', seed=None):
    """
    Generate a random density matrix rho.

    Args:
        length (int): the length of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        method (string): the method to use.
            'Hilbert-Schmidt': sample rho from the Hilbert-Schmidt metric.
            'Bures': sample rho from the Bures metric.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: rho (length, length) a density matrix.
    Raises:
        QiskitError: if the method is not valid.
    """
    if method == 'Hilbert-Schmidt':
        return __random_density_hs(length, rank, seed)
    elif method == 'Bures':
        return __random_density_bures(length, rank, seed)
    else:
        raise QiskitError('Error: unrecognized method {}'.format(method))


def __ginibre_matrix(nrow, ncol=None, seed=None):
    """
    Return a normally distributed complex random matrix.

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
    if seed is not None:
        np.random.seed(seed)
    G = np.random.normal(size=(nrow, ncol)) + \
        np.random.normal(size=(nrow, ncol)) * 1j
    return G


def __random_density_hs(N, rank=None, seed=None):
    """
    Generate a random density matrix from the Hilbert-Schmidt metric.

    Args:
        N (int): the length of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: rho (N,N  a density matrix.
    """
    G = __ginibre_matrix(N, rank, seed)
    G = G.dot(G.conj().T)
    return G / np.trace(G)


def __random_density_bures(N, rank=None, seed=None):
    """
    Generate a random density matrix from the Bures metric.

    Args:
        N (int): the length of the density matrix.
        rank (int or None): the rank of the density matrix. The default
            value is full-rank.
        seed (int): Optional. To set a random seed.
    Returns:
        ndarray: rho (N,N) a density matrix.
    """
    P = np.eye(N) + random_unitary(N).representation
    G = P.dot(__ginibre_matrix(N, rank, seed))
    G = G.dot(G.conj().T)
    return G / np.trace(G)
