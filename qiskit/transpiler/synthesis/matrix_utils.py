# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Helper functions for creating random matrices"""

import numpy as np


def switch_random_rows(array: np.ndarray, rng) -> np.ndarray:
    """Switches two random rows of a given array

    Args:
        array (np.ndarray): target array
        rng: a numpy.random._generator.Generator

    Returns:
        np.ndarray: given array with two of its rows switched
    """
    n = array.shape[1]
    rows = np.zeros(2)
    while rows[0] == rows[1]:
        rows = rng.integers(0, n, 2)
    array[[rows[0], rows[1]]] = array[[rows[1], rows[0]]]

    return array


def add_random_rows(array: np.ndarray, rng) -> np.ndarray:
    """Adds two random rows of a given array

    Args:
        array (np.ndarray): target array
        rng: a numpy.random._generator.Generator

    Returns:
        np.ndarray: given array with two of its rows added
    """
    n = array.shape[1]
    rows = np.zeros(2)
    while rows[0] == rows[1]:
        rows = rng.integers(0, n, 2)

    addition = np.add(array[rows[0]], array[rows[1]])
    addition = np.fmod(addition, 2)

    array[rows[0]] = addition

    return array


def build_random_parity_matrix(
    seed: int, n: int, m: int = 0, identity_allowed: bool = False
) -> np.ndarray:
    """Builds an n*n-sized random parity matrix

    Args:
        seed (int): Seed for a random generator object
        n (int): Size of parity matrix.
        m (int, optional): Number of row operations done on the matrix. Defaults to 0.
        identity_allowed (bool, optional): Whether or not an identity matrix is an acceptable output. Defaults to False.

    Returns:
        np.ndarray: a random parity matrix the size n
    """
    matrix = np.identity(n)

    rng = np.random.default_rng(seed)

    for i in range(m):
        operation = rng.integers(0, 2, 1)[0]
        if operation == 0:
            matrix = switch_random_rows(matrix, rng)
        else:
            matrix = add_random_rows(matrix, rng)

    if identity_allowed == False and np.array_equal(matrix, np.identity(n)):
        operation = rng.integers(0, 2, 1)[0]
        if operation == 0:
            matrix = switch_random_rows(matrix, rng)
        else:
            matrix = add_random_rows(matrix, rng)

    return matrix
