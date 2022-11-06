# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Utility functions for handling permutations."""


def _get_ordered_swap(permutation_in):
    """This attempts to sort the input permutation by iterating through the
    permutation list and swapping the element with where the actual index occurs and
    tracking the swaps.
    """
    permutation = list(permutation_in[:])
    swap_list = []
    for i, val in enumerate(permutation):
        if val != i:
            j = permutation.index(i)
            swap_list.append((i, j))
            permutation[i], permutation[j] = permutation[j], permutation[i]
    swap_list.reverse()
    return swap_list


def _inverse_pattern(pattern):
    """Finds inverse of a permutation pattern."""
    b_map = {pos: idx for idx, pos in enumerate(pattern)}
    return [b_map[pos] for pos in range(len(pattern))]
