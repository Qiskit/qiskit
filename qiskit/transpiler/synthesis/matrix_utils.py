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


def switch_random_rows(array: np.ndarray) -> np.ndarray:
    """Switches two random rows of a given array

    Args:
        array (np.ndarray): target array

    Returns:
        np.ndarray: given array with two of its rows switched
    """
    n = array.shape[1]
    print(n)
    print(array)
    rows = np.random.randint(0, n, 2)
    print(rows)
    print(rows[0])
    print(rows[1])
    array[[rows[0], rows[1]]] = array[[rows[1], rows[0]]]
    print(array)

    return array


def build_random_parity_matrix(n: int) -> np.ndarray:
    """Builds an n*n-sized random parity matrix

    Args:
        n (int): size of parity matrix

    Returns:
        np.ndarray: a random parity matrix the size n
    """
    matrix = np.identity(n)
    return matrix