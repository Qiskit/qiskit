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

"""The depth-optimizing greedy mapper"""

import logging
from typing import Mapping, Set, FrozenSet, Tuple, Optional, Iterable

from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.passes.mapping.depth_mapper import DepthMapper, Reg, ArchNode
from qiskit.transpiler.passes.mapping.mapper import Mapper
from qiskit.transpiler.passes.mapping.placement import Placement

logger = logging.getLogger(__name__)


class GreedyDepthMapper(DepthMapper[Reg, ArchNode]):
    """This mapper tries to place as many gates as possible."""

    def map(self,
            circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode]) -> Mapping[Reg, ArchNode]:
        """Provides a permutation that maps the circuit to the architecture.
        
        The approach to mapping is a greedy algorithm that tries to minimize the maximum circuit
        depth in a way that is similar to minimizing the maximum makespan. We find the 2-qubit
        operation that is the most expensive to perform and place it as well as possible using a
        matching graph. Then we iterate until all operations in the layer were placed or no
        placements are left.
        
        If a chosen mapping has a cost increase associated to it, then we try to perform the
        operation locally instead.

        Args:
          circuit: A circuit to execute
          current_mapping:

        Raises:
          RuntimeError: When no suitable placement is found.

        """
        binops = Mapper._binops_circuit(circuit)
        if not binops:
            return {}

        # The maximum matching gives us the maximum number of edges
        # for use in two-qubit ("binary") operations.
        # Note: maximum matching assumes undirected graph.
        remaining_arch = self.arch_graph.copy()
        matching = Mapper.construct_matching(remaining_arch)  # type: Set[FrozenSet[ArchNode]]
        current_placement = Placement({}, {})  # type: Placement[Reg, ArchNode]

        def placement_cost(place: Tuple[Placement[Reg, ArchNode], DAGNode]
                           ) -> Tuple[int, int]:
            """Compute the cost of placing this placement with the current placement.

            Args:
              place: Tuple[Placement[Reg: 
              ArchNode]: 
              DAGNode]: 

            Returns:

            """
            return self.placement_cost(current_placement + place[0])

        # We wish to minimize the depth of the circuit. This is similar to minimizing the maximum
        # makespan in a Job Scheduling problem context.
        placed_gates = 0
        total_gates = len(binops)
        while binops and matching:
            # Find the most expensive binop to perform and minimize its cost.
            max_min_placement = None  # type: Optional[Tuple[Placement[Reg, ArchNode], DAGNode]]
            for binop in binops:
                binop_map = {
                    qarg: current_mapping[qarg]
                    for qarg in binop.qargs
                }
                # Try all matchings and find the minimum cost placement.
                placements = (
                    (Placement(binop_map, dict(zip(binop.qargs, node_ordering))), binop)
                    for node0, node1 in matching
                    for node_ordering in ((node0, node1), (node1, node0))
                )  # type: Iterable[Tuple[Placement[Reg, ArchNode], DAGNode]]

                min_placement = min(placements, key=placement_cost)

                ###
                #  Also try a local placement if the cost of minimum cost placement was not 0.
                # This way we try to take advantage of free permutations
                # even though the operation may be performed locally.
                ###
                # Are the current nodes available?
                if set(binop_map.values()).issubset(remaining_arch.nodes):
                    # Is the minimal placement on a matching "free"
                    # relative to the current placement?
                    if placement_cost(min_placement)[0] \
                            - self.placement_cost(current_placement)[0] > 0:
                        # Can we perform the operation locally? Then do it.
                        local_placement = Placement(binop_map, binop_map)
                        if local_placement.is_local(self.arch_graph):
                            logger.debug("Using local placement instead.")
                            min_placement = (local_placement, binop)
                    else:
                        logger.debug('Minimal placement was "free" so using that.')

                if max_min_placement is not None:
                    max_min_placement = max(max_min_placement, min_placement, key=placement_cost)
                else:
                    max_min_placement = min_placement

            if max_min_placement is None:
                raise RuntimeError("The max_min_placement was not set.")
            logger.debug("The current cost is: %d\nNew cost is: %d.",
                         self.placement_cost(current_placement),
                         placement_cost(max_min_placement))

            # Place the most expensive binop
            current_placement += max_min_placement[0]
            binops.remove(max_min_placement[1])
            placed_gates += 1

            # The nodes are now in use and can no longer be used for anything else.
            remaining_arch.remove_nodes_from(max_min_placement[0].mapped_to.values())
            if max_min_placement[0].is_local(self.arch_graph):
                # If we used a local placement outside of the matching,
                # recompute the matching to reflect that.
                matching = Mapper.construct_matching(remaining_arch)
            else:
                # Otherwise both directions of the matching are now used.
                matching.remove(frozenset(max_min_placement[0].mapped_to.values()))

        logger.debug("Number of gates placed: %d/%d", placed_gates, total_gates)

        return current_placement.mapped_to
