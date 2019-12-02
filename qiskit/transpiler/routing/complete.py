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

"""Implementations for permuting on a complete graph."""

from typing import List, TypeVar, Mapping, Iterable

from qiskit.transpiler.routing import Permutation, Swap, util

_V = TypeVar('_V')


def partial_permute(mapping: Mapping[_V, _V], nodes: List[_V]) -> Iterable[List[Swap[_V]]]:
    """List swaps that implement a partial permutation on a complete graph.

    Args:
      mapping: A list of destination nodes
      nodes: A collection of all nodes. We need this so we can complete the mapping.
        The right type here is Collection[_V] but that was introduced only in python 3.6.

    Returns:
      A list describing which matchings to swap at each step.

    """
    if not mapping:
        return []

    # Case 1: All mappings are disjoint, so we can implement in 1 time step.
    if len({el for kv in mapping.items() for el in kv}) == 2 * len(mapping):
        return [[(k, v) for k, v in mapping.items()]]

    # Case 2: we complete the mapping with arbitrary assignments. This will take 2 time steps.
    used = set(mapping.values())
    available = (node for node in nodes if node not in used)
    permutation = {}
    for node in nodes:
        if node not in mapping:
            permutation[node] = next(available)
        else:
            permutation[node] = mapping[node]
    assert len(permutation) == len(nodes), "The full permutation is not of the right size."
    return permute(permutation)


def permute(permutation: Permutation[_V]) -> Iterable[List[Swap[_V]]]:
    """List swaps that implement a permutation on a complete graph.

    Assumes full connectivity between all nodes in the given permutation.
    Implements the permutation in at most 2 depth.

    Based on the paper "Routing Permutations on Graphs via Matchings"
    by Noga Alon, F. R. K. Chung, and R. L. Graham,
    DOI: https://doi.org/10.1137/S0895480192236628

    Args:
      permutation: A list of destination nodes

    Returns:
      A list describing which matchings to swap at each step.

    """
    cyclic_perms = util.cycles(permutation)

    swaps = []  # type: List[List[List[Swap[_V]]]]
    for cycle in cyclic_perms:
        if len(cycle) <= 1:
            continue

        inverted_cycle = {v: k for k, v in cycle.items()}

        # get a starting point and put that into the right place.
        first, first_to = cycle.popitem()
        cycle[first] = first_to  # reinsert pop (no peek available)
        step1 = [(first, first_to)]

        # then swap according to the inverted cycle
        # We are at "current" and have to_current -> current -> current_to in the permutation
        # (after the first swap).
        # But because of the first swap we cannot just swap to_current and current,
        # so we put it into position for the next step.
        current = first
        current_to = cycle[first_to]
        for _ in range((len(cycle) - 2) // 2):
            to_current = inverted_cycle[current]
            step1.append((to_current, current_to))
            current = to_current
            # After the swap, the current_to of the new current is the one it was swapped with.
            current_to = cycle[current_to]

        util.swap_permutation([step1], cycle)
        # Now we have the permutation   (x0     x1  x2  x3  ... xm)
        #                               (x2     x1  x0  xm  ... x3)
        # and we can swap everything into place.
        assert cycle[cycle[first]] == first, "Current permutation is not as expected."
        assert cycle[first_to] == first_to, "Permutation first_to is not done."

        tiny_cycles = util.cycles(cycle)
        step2 = [tiny_cycle.popitem() for tiny_cycle in tiny_cycles if len(tiny_cycle) > 1]

        cycle_swaps = [step1, step2]
        swaps.append(cycle_swaps)

    return util.optimize_swaps(util.flatten_swaps(swaps))
