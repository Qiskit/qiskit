# This code is part of Qiskit.
#
# (C) Copyright IBM 2019, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Arithmetic Utilities
"""

from typing import List, Tuple
import numpy as np


def normalize_vector(vector):
    """
    Normalize the input state vector.
    """
    return vector / np.linalg.norm(vector)


def is_power_of_2(num):
    """
    Check if the input number is a power of 2.
    """
    return num != 0 and ((num & (num - 1)) == 0)


def log2(num):
    """
    Compute the log2 of the input number. Use bit operation if the input is a power of 2.
    """
    if is_power_of_2(num):
        ret = 0
        while True:
            if num >> ret == 1:
                return ret
            else:
                ret += 1
    else:
        return np.log2(num)


def is_power(num, return_decomposition=False):
    """
    Check if num is a perfect power in O(n^3) time, n=ceil(logN)
    """
    # pylint: disable=invalid-name
    b = 2
    while (2 ** b) <= num:
        a = 1
        c = num
        while (c - a) >= 2:
            m = int((a + c) / 2)

            if (m ** b) < (num + 1):
                p = int((m ** b))
            else:
                p = int(num + 1)

            if int(p) == int(num):
                if return_decomposition:
                    return True, int(m), int(b)
                else:
                    return True

            if p < num:
                a = int(m)
            else:
                c = int(m)
        b = b + 1
    if return_decomposition:
        return False, num, 1
    else:
        return False


def next_power_of_2_base(n):
    """
    Return the base of the smallest power of 2 no less than the input number
    """
    base = 0
    if n and not (n & (n - 1)):  # pylint: disable=superfluous-parens
        return log2(n)

    while n != 0:
        n >>= 1
        base += 1

    return base


def transpositions(permutation: List[int]) -> List[Tuple[int, int]]:
    """Return a sequence of transpositions, corresponding to the permutation.

    Args:
        permutation: The ``List[int]`` defining the permutation. An element at index ``j`` should be
            permuted to index ``permutation[j]``.

    Returns:
        List of transpositions, corresponding to the permutation. For permutation = [3, 0, 2, 1],
        returns [(0,1), (0,3)]
    """
    unchecked = [True] * len(permutation)
    cyclic_form = []
    for i in range(len(permutation)):
        if unchecked[i]:
            cycle = [i]
            unchecked[i] = False
            j = i
            while unchecked[permutation[j]]:
                j = permutation[j]
                cycle.append(j)
                unchecked[j] = False
            if len(cycle) > 1:
                cyclic_form.append(cycle)
    cyclic_form.sort()
    res = []
    for x in cyclic_form:
        len_x = len(x)
        if len_x == 2:
            res.append((x[0], x[1]))
        elif len_x > 2:
            first = x[0]
            for y in x[len_x - 1:0:-1]:
                res.append((first, y))
    return res
