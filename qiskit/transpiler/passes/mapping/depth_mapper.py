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

"""mapping.depth computes good mappings to the architecture for a given circuit minimizing depth.

Given a circuit and an architecture graph we compute how to place that circuit on the architecture
so gates can be performed where we try to minimize the longest (most expensive) sequence of gates
on any qubit.
"""

from typing import Mapping, Dict, Callable, Iterable, List, TypeVar, \
    Tuple

import networkx as nx

from qiskit.transpiler import routing as pm
from qiskit.transpiler.passes.mapping.mapper import Mapper
from qiskit.transpiler.passes.mapping.placement import Placement
from qiskit.transpiler.routing import Swap, util  # pylint: disable=unused-import

Reg = TypeVar('Reg')
ArchNode = TypeVar('ArchNode')


class DepthMapper(Mapper[Reg, ArchNode]):
    """An abstract mapper class for optimizing for the circuit depth."""

    def __init__(self,
                 arch_graph: nx.Graph,
                 arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                         Iterable[List[Swap[ArchNode]]]],
                 max_placement_size: int = 4) -> None:
        """Construct a depth mapper.

        Args:
          arch_graph: The directed architecture graph.
          arch_permuter: The permuter on the architecture graph.
          max_placement_size: The maximum size of a placement to cache.
            This memory requires scale superexponentially in this parameter.
            There are O(n!^2/(n-k)!^2) arch_mappings for k the size of the mapping.
        """
        super().__init__(arch_graph)
        self.arch_permuter = arch_permuter
        self.placement_costs = {}  # type: Dict[Placement[Reg, ArchNode], Tuple[int, int]]
        self.max_placement_size = max_placement_size

    def placement_cost(self, placement: Placement[Reg, ArchNode]) -> Tuple[int, int]:
        """Find the cost of performing the placement in depth.
        
        Will cache results for given small placements to speed up future computations.
        
        Returns:
            A tuple with the placement depth cost and its size. Suitable for depth-first
            comparisons, and then break ties with the size.
        """
        if placement in self.placement_costs:
            return self.placement_costs[placement]

        # Count the depth of the number of swaps multiplied with the swap cost.
        permutation_swaps = self.arch_permuter(placement.arch_mapping)
        par_cost = pm.util.longest_path(permutation_swaps)
        seq_cost = sum(len(swap_step) for swap_step in permutation_swaps)

        # Cache the result if the placement is small enough.
        # This prevents memory overruns but still caches the common case.
        # There are O(n!^2/(n-k)!^2) arch_mappings for k the size of the mapping.
        if len(placement.arch_mapping) <= self.max_placement_size:
            self.placement_costs[placement] = (par_cost, seq_cost)
        return par_cost, seq_cost
