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

"""The depth-optimizing incremental mapper"""

import logging
from typing import Mapping, Tuple, Iterable

from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.passes.mapping.depth_mapper import DepthMapper, Reg, ArchNode
from qiskit.transpiler.passes.mapping.mapper import Mapper
from qiskit.transpiler.passes.mapping.placement import Placement

logger = logging.getLogger(__name__)


class IncrementalDepthMapper(DepthMapper[Reg, ArchNode]):
    """A depth mapper that will place the cheapest gate and move the rest closer.

    After placing the cheapest gate, we place an upper bound on the movement cost
    of the remaining gates that is the same as that of the cheapest. The mapper will
    then move the remaining gates only as close as that upper bound lets it.
    """

    def map(self,
            circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode]) -> Mapping[Reg, ArchNode]:
        """Map a given layer of the circuit to the architecture"""
        binops = Mapper._binops_circuit(circuit)
        if not binops:
            return {}
        remaining_arch = self.arch_graph.copy()
        current_placement = Placement({}, {})  # type: Placement[Reg, ArchNode]

        # First find the minimal cost binop to place
        def minimal_placement(binop_qargs: Tuple[Reg, Reg]) -> Placement[Reg, ArchNode]:
            """Find the placement that has minimal placement cost for the binop."""
            binop_map = {qarg: current_mapping[qarg] for qarg in binop_qargs}
            placements = (
                Placement(binop_map, dict(zip(binop_qargs, nodes)))
                for edge in self.arch_graph.edges
                # Also try the reverse of the edge.
                for nodes in (edge, reversed(edge))
            )  # type: Iterable[Placement[Reg, ArchNode]]

            return min(placements, key=self.placement_cost)

        min_placement = min(((minimal_placement(binop.qargs), binop)
                             for binop in binops),
                            key=lambda p: self.placement_cost(p[0]))
        logger.debug("Minimal placement is: %s.", min_placement[0])
        current_placement += min_placement[0]
        remaining_arch.remove_nodes_from(min_placement[0].mapped_to.values())
        binops.remove(min_placement[1])
        # We allow only swap circuits of equal size, but at least of 1 SWAP size.
        initial_minimal_cost = self.placement_cost(min_placement[0])[0]
        minimal_cost = max(initial_minimal_cost, 1)

        def placement_cost(place: Placement[Reg, ArchNode]) -> Tuple[int, int]:
            """Compute the cost of placing this placement with the current placement."""
            return self.placement_cost(current_placement + place)

        # For the remaining binops, place each as close to their partner as possible,
        # while not increasing the cost much further.
        for binop in binops:
            binop_map = {qubit: current_mapping[qubit] for qubit in binop.qargs}
            # Enumerate all nodes that do not exceed the cost threshold for both qargs
            eligible_nodes = [[
                node for node in remaining_arch.nodes
                # Rough filter of possible locations
                if self.distance[current_mapping[qubit]][node] <= minimal_cost
                # Exact filter.
                and placement_cost(Placement(binop_map, {qubit: node}))[0] <= minimal_cost
            ] for qubit in binop.qargs]
            # Find the pair of nodes such that the distance is minimized.
            node_pairs = [
                (node0, node1)
                for node0 in eligible_nodes[0]
                for node1 in eligible_nodes[1]
                if node0 != node1  # both qargs cannot have the same destination
            ]
            try:
                # Find the pair of eligible nodes that minimizes the distance between the two.
                minimal_distance = min(self.distance[node0][node1] for node0, node1 in node_pairs)
                closest_placements = [
                    Placement(binop_map, {binop.qargs[0]: node0, binop.qargs[1]: node1})
                    for node0, node1 in node_pairs
                    if self.distance[node0][node1] == minimal_distance]
                min_closest_placement = min(closest_placements, key=placement_cost)
                # Then place the qargs at those nodes
                logger.debug("Placed %s, old dist: %d, new: %d.",
                             min_closest_placement,
                             self.distance[current_mapping[binop.qargs[0]]][
                                 current_mapping[binop.qargs[1]]],
                             minimal_distance)
                current_placement += min_closest_placement
                remaining_arch.remove_nodes_from(min_closest_placement.mapped_to.values())
                # Update minimal cost, because the cost function is not stable.
                # Otherwise future nodes may not be able to be placed anywhere
                # (since it will always exceed the cost.)
                new_minimal_cost = self.placement_cost(current_placement)[0]
                logger.debug("Old minimal_cost: %d, new: %d", minimal_cost, new_minimal_cost)
                minimal_cost = max(new_minimal_cost, 1)
            except ValueError:
                logger.debug("No eligible node pairs for %s.", binop_map)

        logger.debug("Initial minimal cost set at: %d. Has finally become: %d.",
                     initial_minimal_cost, minimal_cost)
        return current_placement.mapped_to
