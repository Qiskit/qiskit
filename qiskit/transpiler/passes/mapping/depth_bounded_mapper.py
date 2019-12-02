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

"""The depth-optimizing bounded depth mapper"""

import logging
import random
import sys
from typing import Mapping, Set, FrozenSet, Tuple, Optional, Iterable

from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.passes.mapping.depth_mapper import DepthMapper, Reg, ArchNode
from qiskit.transpiler.passes.mapping.mapper import Mapper
from qiskit.transpiler.passes.mapping.placement import Placement

logger = logging.getLogger(__name__)


class BoundedDepthMapper(DepthMapper[Reg, ArchNode]):
    """A depth mapper that will place the cheapest gate and move the rest closer.

    After placing the most expensive gate, we place an upper bound on the movement cost
    of the remaining gates that is the twice that of the most expensive. The mapper will
    then move the remaining gates only as close as that upper bound lets it.

    Args:

    Returns:

    """

    def map(self,
            circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode]) -> Mapping[Reg, ArchNode]:
        """Provides a permutation that maps the circuit to the architecture.

        If a chosen mapping has a cost increase associated to it,
        then we try to perform the operation locally instead.

        Args:
          circuit: A circuit to execute
          current_mapping: The current mapping.

        Returns:
          The new mapping of qubits to architecture nodes.

        Raises:
          RuntimeError: If the extremal placement was not found.

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

        placed_gates = 0
        total_gates = len(binops)

        def placement_cost(place: Tuple[Placement[Reg, ArchNode], Tuple[Reg, Reg]]
                           ) -> Tuple[int, int]:
            """Compute the cost of placing this placement with the current placement."""
            return self.placement_cost(current_placement + place[0])

        minimal_cost = sys.maxsize
        while binops and matching:
            # Find the cheapest binop to perform and minimize its cost.
            min_min_placement = \
                None  # type: Optional[Tuple[Placement[Reg, ArchNode], Tuple[Reg, Reg]]]
            for binop in binops:
                binop_map = {qarg: current_mapping[qarg] for qarg in binop.qargs}
                # Try all matchings and find the minimum cost placement.
                placements = (
                    (Placement(binop_map, dict(zip(binop.qargs, node_ordering))), binop)
                    for node0, node1 in matching
                    for node_ordering in ((node0, node1), (node1, node0))
                )  # type: Iterable[Tuple[Placement[Reg, ArchNode], DAGNode]]

                min_placement = min(placements, key=placement_cost)

                if min_min_placement is not None:
                    min_min_placement = min(min_min_placement, min_placement, key=placement_cost)
                else:
                    min_min_placement = min_placement

            if min_min_placement is None:
                raise RuntimeError("The min_min_placement was not set.")

            min_place_cost = placement_cost(min_min_placement)[0]
            minimal_cost = min(minimal_cost, min_place_cost)  # Should only be set once.
            logger.debug("Cost changing from %d → %d / %d",
                         self.placement_cost(current_placement), min_place_cost, 2 * minimal_cost)

            # If the placement cost of this placement exceeds the threshold,
            # stop and go to next iteration.
            if min_place_cost > 2 * minimal_cost:
                logger.debug("Threshold exceeded!")
                break

            # Place the most expensive binop
            current_placement += min_min_placement[0]
            binops.remove(min_min_placement[1])
            placed_gates += 1

            # The nodes are now in use and can no longer be used for anything else.
            remaining_arch.remove_nodes_from(min_min_placement[0].mapped_to.values())
            # Both directions of the matching are now used.
            matching.remove(frozenset(min_min_placement[0].mapped_to.values()))

        logger.debug("Number of gates placed before: %d/%d", placed_gates, total_gates)

        ###
        # We now try to place the remaining gates on nodes
        # such that they are closer to their destination.
        # IDEA: apply some min-makespan instead of arbitrary ordering
        ###
        ordered_binops = list(binops)
        random.shuffle(ordered_binops)  # Ensure random ordering
        for binop in ordered_binops:
            logger.debug("Placing %s", binop)
            # Enumerate all nodes that do not exceed the cost threshold for both qargs
            eligible_nodes = [[
                node for node in remaining_arch.nodes
                # Rough filter of possible locations
                if self.distance[current_mapping[qarg]][node] <= 2 * minimal_cost
                # Exact filter
                and placement_cost((Placement({qarg: current_mapping[qarg]},
                                              {qarg: node}), binop.qargs))[0] <= 2 * minimal_cost
            ] for qarg in binop.qargs]
            # Find the pair of nodes such that the distance is minimized.
            node_pairs = (
                (node0, node1)
                for node0 in eligible_nodes[0]
                for node1 in eligible_nodes[1]
                if node0 != node1  # both qargs cannot have the same destination
            )
            try:
                # Find the pair of eligible nodes that minimizes the distance between the two.
                closest_nodes = min(node_pairs, key=lambda nodes: self.distance[nodes[0]][nodes[1]])
                # Then place the qargs at those nodes
                logger.debug("Placed %s at %s, old dist: %d, new: %d.",
                             {qarg: current_mapping[qarg] for qarg in binop.qargs},
                             closest_nodes,
                             self.distance[current_mapping[binop.qargs[0]]][
                                 current_mapping[binop.qargs[1]]],
                             self.distance[closest_nodes[0]][closest_nodes[1]])
                current_placement += Placement({qarg: current_mapping[qarg]
                                                for qarg in binop.qargs},
                                               dict(zip(binop.qargs, closest_nodes)))
                remaining_arch.remove_nodes_from(closest_nodes)
                placed_gates += 1
            except ValueError:
                logger.debug("No eligible node pairs")

        after_cost = self.placement_cost(current_placement)[0]
        logger.debug("Number of gates placed: %d/%d for a final cost of %d.",
                     placed_gates, total_gates, after_cost)
        if after_cost > 2 * minimal_cost:
            logger.debug("New cost exceeded the threshold.")

        return current_placement.mapped_to
