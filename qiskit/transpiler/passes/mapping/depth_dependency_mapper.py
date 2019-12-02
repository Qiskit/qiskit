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

"""The depth-optimizing mapper that orders gates by their number of dependencies."""

import logging
import operator
import sys
from collections import defaultdict
from typing import Callable, Mapping, Iterable, List, Optional, Tuple, Type, Dict, Iterator

import networkx as nx

from qiskit.circuit import Instruction
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.passes.mapping.depth_mapper import DepthMapper, Reg, ArchNode
from qiskit.transpiler.passes.mapping.mapper import Mapper
from qiskit.transpiler.passes.mapping.placement import Placement
from qiskit.transpiler.routing import Swap

logger = logging.getLogger(__name__)


class DepthDependencyMapper(DepthMapper[Reg, ArchNode]):
    """A mapper that uses the number of dependencies on a gate to decide the ordering."""

    def __init__(self,
                 arch_graph: nx.Graph,
                 arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                         Iterable[List[Swap[ArchNode]]]],
                 node_dependencies: Optional[Mapping[int, int]] = None) -> None:
        super().__init__(arch_graph, arch_permuter)
        self.node_dependencies = node_dependencies

    def map(self,
            circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode],
            node_dependencies: Optional[Mapping[int, int]] = None) -> Mapping[Reg, ArchNode]:
        """Maps nodes that have many dependents in the DAG first.

        Args:
          circuit: A circuit to execute
          node_dependencies: A mapping from node ids to number of dependents.
          current_mapping: The current mapping.

        Returns:
            A mapping of qubits to architecture nodes.
        """
        if node_dependencies is None:
            if self.node_dependencies is None:
                logger.debug("No node dependencies given. Calculating node dependencies.")
                node_dependencies = DepthDependencyMapper.dependents_map(circuit)
            else:
                node_dependencies = self.node_dependencies

        binops = Mapper._binops_circuit(circuit)
        # Reshape the nodes to their qargs.
        binops_dependents = [(binop, node_dependencies[binop])
                             for binop in binops]  # type: List[Tuple[DAGNode, int]]

        if not binops_dependents:
            return {}

        # After sorting by nr of descendents we discard the dependents field.
        binops_dependents = list(sorted(binops_dependents,
                                        key=operator.itemgetter(1),
                                        reverse=True))
        logger.debug("Max dependents: %s.", binops_dependents[0][1])
        binops = [el[0] for el in binops_dependents]

        remaining_arch = self.arch_graph.copy()
        current_placement = Placement({}, {})  # type: Placement[Reg, ArchNode]
        # The first iteration has unbounded cost for its placement.
        minimal_cost = sys.maxsize

        def placement_cost(place: Placement[Reg, ArchNode]) -> Tuple[int, int]:
            """Compute the cost of placing this placement with the current placement."""
            return self.placement_cost(current_placement + place)

        # Iterate over the binops sorted by nr of dependents and place them.
        # Place each as close to their partner as possible,
        # while not increasing the cost much further.
        for binop in binops:
            binop_map = {qarg: current_mapping[qarg] for qarg in binop.qargs}
            # Enumerate all nodes that do not exceed the cost threshold for both qargs
            eligible_nodes = [[
                node for node in remaining_arch.nodes
                # Rough filter of possible locations
                if self.distance[current_mapping[qarg]][node] <= minimal_cost
                # Exact filter.
                and placement_cost(Placement(binop_map, {qarg: node}))[0] <= minimal_cost
            ] for qarg in binop.qargs]
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
                logger.debug("Placed %s, old dist: %s, new: %s.",
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

        return current_placement.mapped_to

    @staticmethod
    def dependents_map(circuit: DAGCircuit,
                       gate_costs: Mapping[Type[Instruction], int] = None) -> Mapping[DAGNode, int]:
        """Compute a mapping from dag nodes to the weighted longest path length from that node
        to the end of the circuit.

        If gate_costs is not given, all gate costs are assumed to be 1.
        """
        if gate_costs is None:
            gate_costs = defaultdict(lambda: 1)

        max_lengths = {}  # type: Dict[DAGNode, int]
        reversed_layers = reversed(list(layer["graph"]
                                        for layer in
                                        circuit.layers()))  # type: Iterator[DAGCircuit]
        for layer in reversed_layers:
            for node in layer.op_nodes():
                node_cost = gate_costs[node.op]
                max_lengths[node] = max(
                    (max_lengths[successor] for successor in circuit.successors(node)),
                    default=0) + node_cost
        return max_lengths
