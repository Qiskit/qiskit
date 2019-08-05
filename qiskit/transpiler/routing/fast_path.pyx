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

# Copyright 2019 Andrew M. Childs, Eddie Schoute, Cem M. Unsal
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementations for permuting on a line graph."""

from qiskit.transpiler.routing import util
from libc.stdlib cimport malloc, free


cpdef object permute_path(permutation):
    """List swaps that implement a permutation on a path.

    Assumes that every node is only connected to it's neighbors

    Based on the paper "Routing Permutations on Graphs via Matchings"
    by Noga Alon, F. R. K. Chung, and R. L. Graham,
    DOI: https://doi.org/10.1137/S0895480192236628

    :param permutation: A list of destination nodes
    :return: A list describing which matchings to swap at each step.
    """
    cdef int length = len(permutation)
    cdef int *current = <int *> malloc(length * sizeof(int))
    if not current:
        raise MemoryError()

    # Copy all elements into a cython array, in order.
    for i, j in permutation.items():
        current[i] = j
    swaps = _inner_permute_path(current, length)
    free(current)
    return swaps


cdef object _inner_permute_path(int *current, unsigned int length):
    cdef unsigned int i, j, remainder
    swaps = list()
    # We do a total of `length` iterations.
    for i in range((length+1) // 2):  # ⌈length/2⌉
        no_swaps = True
        for remainder in range(2):
            current_swaps = list()
            for j in range(remainder, length-1, 2):
                if current[j] > current[j+1]:
                    current_swaps.append((j, j+1))
                    current[j], current[j+1] = current[j+1], current[j]
            swaps.append(current_swaps)

            if current_swaps:
                no_swaps = False
        # When both even and odd swaps steps have not emitted any swaps, stop.
        if no_swaps:
            break
    return util.optimize_swaps(swaps)


cpdef object permute_path_partial(mapping, length = None):
    """Permute a partial mapping on the path.

    Fills a partial mapping up to a full permutation then calls the full permutation algorithm.
    """
    cdef unsigned int c_length
    if length is None:
        c_length = max(set(mapping.keys() | set(mapping.values()))) + 1
    else:
        c_length = length

    cdef int direction, origin, destination
    if len(mapping) == 1:
        # Handle common case quickly.
        origin, destination = next(iter(mapping.items()))
        if origin == destination:
            # trivial case
            return iter([])

        direction = 1 if origin < destination else -1
        # Range is exclusive of endpoint.
        nodes = range(origin, destination + direction, direction)
        return [[swap] for swap in zip(nodes[0:-1], nodes[1:])]

    used = set(mapping.values())
    available = iter([i for i in range(c_length) if i not in used])
    cdef int *full_mapping = <int *> malloc(c_length * sizeof(int))
    for i in range(c_length):
        if i in mapping:
            full_mapping[i] = mapping[i]
        else:
            full_mapping[i] = next(available)

    swaps = _inner_permute_path(full_mapping, c_length)
    free(full_mapping)
    return swaps
