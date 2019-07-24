"""Implementations for permuting on a line graph."""
#  arct performs circuit transformations of quantum circuit for architectures
#  Copyright (C) 2019  Andrew M. Childs, Eddie Schoute, Cem M. Unsal
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from typing import List, TypeVar, Iterator, Mapping, Optional

from qiskit.transpiler.routing import Permutation, Swap, util

_V = TypeVar('_V')


def permute_path(permutation: Permutation[int]) -> Iterator[List[Swap[int]]]:
    """List swaps that implement a permutation on a path.

    Assumes that every node is only connected to it's neighbors

    Based on the paper "Routing Permutations on Graphs via Matchings"
    by Noga Alon, F. R. K. Chung, and R. L. Graham,
    DOI: https://doi.org/10.1137/S0895480192236628

    :param permutation: A list of destination nodes
    :return: A list describing which matchings to swap at each step.
    """
    current = permutation.copy()
    return util.optimize_swaps(_emit_swap_steps(current))


def _emit_swap_steps(current: Permutation[int]) -> Iterator[List[Swap[int]]]:
    """A coroutine that emits simple swap steps for the path."""
    length = len(current)
    # We do a total of `length` iterations.
    for _ in range((length + 1) // 2):  # ⌈length/2⌉
        no_swaps = True
        for remainder in range(2):
            current_swaps: List[Swap[int]] = [(i, i - 1) for i in range(2 - remainder, length, 2)
                                              if current[i - 1] > current[i]]
            yield current_swaps
            util.swap_permutation([current_swaps], current)
            if current_swaps:
                no_swaps = False
        # When both even and odd swaps steps have not emitted any swaps, stop.
        if no_swaps:
            return


def permute_path_partial(mapping: Mapping[int, int],
                         length: Optional[int] = None) -> Iterator[List[Swap[int]]]:
    """Permute a partial mapping on the path.

    Fills a partial mapping up to a full permutation then calls the full permutation algorithm.
    """
    if length is None:
        if mapping:
            length = max(set(mapping.keys() | set(mapping.values())))
        else:  # The mapping is empty
            return iter(list())

    if len(mapping) == 1:
        # Handle common case quickly.
        origin, destination = next(iter(mapping.items()))
        if origin == destination:
            # trivial case
            return iter([])

        direction = 1 if origin < destination else -1
        # Range is exclusive of endpoint.
        nodes = range(origin, destination + direction, direction)
        return ([swap] for swap in zip(nodes[0:-1], nodes[1:]))

    used = set(mapping.values())
    available = (i for i in range(length + 1) if i not in used)
    full_mapping = dict(mapping)
    for i in range(length + 1):
        if i not in mapping:
            full_mapping[i] = next(available)

    return util.optimize_swaps(_emit_swap_steps(full_mapping))
