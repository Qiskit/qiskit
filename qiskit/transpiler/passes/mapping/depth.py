"""mapping.depth computes good mappings to the architecture for a given circuit minimizing depth.

Given a circuit and an architecture graph we compute how to place that circuit on the architecture
so gates can be performed where we try to minimize the longest (most expensive) sequence of gates
on any qubit.
"""
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

import logging
import operator
import random
import sys
import typing
from collections import defaultdict
from typing import Mapping, Dict, Set, Callable, Iterable, List, TypeVar, \
    Optional, Tuple, FrozenSet, Type, Iterator

import networkx as nx
from qiskit.circuit import Instruction
from qiskit.dagcircuit import DAGCircuit, DAGNode

from qiskit.transpiler import routing as pm
from qiskit.transpiler.passes.mapping.mapper import Mapper
from qiskit.transpiler.passes.mapping.placement import Placement
from qiskit.transpiler.passes.mapping.size import QiskitSizeMapper
from qiskit.transpiler.routing import Swap, util

Reg = TypeVar('Reg')
ArchNode = TypeVar('ArchNode')

logger = logging.getLogger(__name__)


class DepthMapper(Mapper[Reg, ArchNode]):
    """A mapper class for optimizing for the circuit depth."""

    def __init__(self,
                 arch_graph: nx.Graph,
                 arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                         Iterable[List[Swap[ArchNode]]]]
                 ) -> None:
        super().__init__(arch_graph)
        self.arch_permuter = arch_permuter
        self.placement_costs: Dict[Placement[Reg, ArchNode], Tuple[int, int]] = {}

    def placement_cost(self, placement: Placement[Reg, ArchNode]) -> Tuple[int, int]:
        """Find the cost of performing the placement in depth.

        Will cache results for given small placements to speed up future computations.

        :return: A tuple with the placement depth cost and its size. Suitable for depth-first
            comparisons, and then break ties with the size."""
        if placement in self.placement_costs:
            return self.placement_costs[placement]

        # Count the depth of the number of swaps multiplied with the swap cost.
        permutation_swaps = self.arch_permuter(placement.arch_mapping)
        par_cost = pm.util.longest_path(permutation_swaps)
        seq_cost = sum(len(swap_step) for swap_step in permutation_swaps)

        # Cache the result if the placement is small enough.
        # This prevents memory overruns but still caches the common case.
        # There are O(n!^2/(n-k)!^2) arch_mappings for k the size of the mapping.
        if len(placement.arch_mapping) <= 4:
            self.placement_costs[placement] = (par_cost, seq_cost)
        return par_cost, seq_cost


class SimpleDepthMapper(DepthMapper[Reg, ArchNode]):
    def map(self, circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode] = None) -> Mapping[Reg, ArchNode]:
        """
        Try to map as many two-qubit gates to a maximum matching as possible.

        Note: Does not take into account the scoring function, nor the weights on the graph.

        :param circuit: A circuit to execute
        :param arch_graph: The architecture graph,optionally with weights on edges.
            Default weights are 1.
        :param current_mapping: The current mapping of registers to archictecture nodes.
        :return:
        """
        binops = Mapper._binops_circuit(circuit)
        matching: Set[FrozenSet[ArchNode]] = Mapper.construct_matching(self.arch_graph)
        # First assign the two-qubit gates, because they are restricted by the architecture
        mapping: Dict[Reg, ArchNode] = {}
        for binop in binops:
            if matching:
                # pick an available matching and map this operation to that matching
                node0, node1 = matching.pop()
                mapping[binop.qargs[0]] = node0
                mapping[binop.qargs[1]] = node1
            else:
                # no more matchings
                break

        return mapping


class GreedyDepthMapper(DepthMapper[Reg, ArchNode]):
    def map(self,
            circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode]) -> Mapping[Reg, ArchNode]:
        """
        Provides a permutation that maps the circuit to the architecture.

        The approach to mapping is a greedy algorithm that tries to minimize the maximum circuit depth
        in a way that is similar to minimizing the maximum makespan. We find the 2-qubit operation
        that is the most expensive to perform and place it as well as possible using a matching graph.
        Then we iterate until all operations in the layer were placed or no placements are left.

        If a chosen mapping has a cost increase associated to it, then we try to perform the operation
        locally instead.

        :param circuit: A circuit to execute
        :param current_mapping:
        :return:
        """
        binops = Mapper._binops_circuit(circuit)
        if not binops:
            return {}

        # The maximum matching gives us the maximum number of edges
        # for use in two-qubit ("binary") operations.
        # Note: maximum matching assumes undirected graph.
        remaining_arch = self.arch_graph.copy()
        matching: Set[FrozenSet[ArchNode]] = Mapper.construct_matching(remaining_arch)
        current_placement: Placement[Reg, ArchNode] = Placement({}, {})

        def placement_cost(place: Tuple[Placement[Reg, ArchNode], DAGNode]) -> Tuple[
            int, int]:
            """Compute the cost of placing this placement with the current placement."""
            return self.placement_cost(current_placement + place[0])

        # We wish to minimize the depth of the circuit. This is similar to minimizing the maximum
        # makespan in a Job Scheduling problem context.
        placed_gates = 0
        total_gates = len(binops)
        while binops and matching:
            # Find the most expensive binop to perform and minimize its cost.
            max_min_placement: Optional[Tuple[Placement[Reg, ArchNode], DAGNode]] = None
            for binop in binops:
                binop_map = {
                    qarg: current_mapping[qarg]
                    for qarg in binop.qargs
                    }
                # Try all matchings and find the minimum cost placement.
                placements: Iterable[Tuple[Placement[Reg, ArchNode], DAGNode]] = (
                    (Placement(binop_map, dict(zip(binop.qargs, node_ordering))), binop)
                    for node0, node1 in matching
                    for node_ordering in ((node0, node1), (node1, node0))
                    )

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
            logger.debug(f"The current cost is: {self.placement_cost(current_placement)}\n"
                         f"New cost is: {placement_cost(max_min_placement)}.")

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

        logger.debug(f"Number of gates placed: {placed_gates}/{total_gates}")

        return current_placement.mapped_to


class IncrementalDepthMapper(DepthMapper[Reg, ArchNode]):
    """A depth mapper that will place the cheapest gate and move the rest closer.

    After placing the cheapest gate, we place an upper bound on the movement cost
    of the remaining gates that is the same as that of the cheapest. The mapper will
    then move the remaining gates only as close as that upper bound lets it."""

    def map(self,
            circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode]) -> Mapping[Reg, ArchNode]:
        binops = Mapper._binops_circuit(circuit)
        if not binops:
            return {}
        remaining_arch = self.arch_graph.copy()
        current_placement: Placement[Reg, ArchNode] = Placement({}, {})

        # First find the minimal cost binop to place
        def minimal_placement(binop_qargs: Tuple[Reg, Reg]) -> Placement[Reg, ArchNode]:
            """Find the placement that has minimal placement cost for the binop."""
            binop_map = {qarg: current_mapping[qarg] for qarg in binop_qargs}
            placements: Iterable[Placement[Reg, ArchNode]] = (
                Placement(binop_map, dict(zip(binop_qargs, nodes)))
                for edge in self.arch_graph.edges
                # Also try the reverse of the edge.
                for nodes in (edge, reversed(edge))
                )

            return min(placements, key=self.placement_cost)

        min_placement = min(((minimal_placement(binop.qargs), binop)
                             for binop in binops),
                            key=lambda p: self.placement_cost(p[0]))
        logger.debug(f"Minimal placement is: {min_placement[0]}.")
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
                ]
                for qubit in binop.qargs]
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
                logger.debug(f"Placed {min_closest_placement}, "
                             f"old dist: {self.distance[current_mapping[binop.qargs[0]]][current_mapping[binop.qargs[1]]]}, "
                             f"new: {minimal_distance}.")
                current_placement += min_closest_placement
                remaining_arch.remove_nodes_from(min_closest_placement.mapped_to.values())
                # Update minimal cost, because the cost function is not stable.
                # Otherwise future nodes may not be able to be placed anywhere
                # (since it will always exceed the cost.)
                new_minimal_cost = self.placement_cost(current_placement)[0]
                logger.debug(f"Old minimal_cost: {minimal_cost}, new: {new_minimal_cost}")
                minimal_cost = max(new_minimal_cost, 1)
            except ValueError:
                logger.debug(f"No eligible node pairs for {binop_map}.")

        logger.debug(f"Initial minimal cost set at: {initial_minimal_cost}. "
                     f"Has finally become: {minimal_cost}.")
        return current_placement.mapped_to


class BoundedDepthMapper(DepthMapper[Reg, ArchNode]):
    """A depth mapper that will place the cheapest gate and move the rest closer.

    After placing the most expensive gate, we place an upper bound on the movement cost
    of the remaining gates that is the twice that of the most expensive. The mapper will
    then move the remaining gates only as close as that upper bound lets it."""

    def map(self,
            circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode]) -> Mapping[Reg, ArchNode]:
        """
        Provides a permutation that maps the circuit to the architecture.

        If a chosen mapping has a cost increase associated to it, then we try to perform the operation
        locally instead.

        :param circuit: A circuit to execute
        :param current_mapping:
        :return:
        """
        binops = Mapper._binops_circuit(circuit)
        if not binops:
            return {}

        # The maximum matching gives us the maximum number of edges
        # for use in two-qubit ("binary") operations.
        # Note: maximum matching assumes undirected graph.
        remaining_arch = self.arch_graph.copy()
        matching: Set[FrozenSet[ArchNode]] = Mapper.construct_matching(remaining_arch)
        current_placement: Placement[Reg, ArchNode] = Placement({}, {})

        placed_gates = 0
        total_gates = len(binops)

        def placement_cost(place: Tuple[Placement[Reg, ArchNode], Tuple[Reg, Reg]]) -> Tuple[
            int, int]:
            """Compute the cost of placing this placement with the current placement."""
            return self.placement_cost(current_placement + place[0])

        minimal_cost = sys.maxsize
        while binops and matching:
            # Find the cheapest binop to perform and minimize its cost.
            min_min_placement: Optional[Tuple[Placement[Reg, ArchNode], Tuple[Reg, Reg]]] = None
            for binop in binops:
                binop_map = {qarg: current_mapping[qarg] for qarg in binop.qargs}
                # Try all matchings and find the minimum cost placement.
                placements: Iterable[Tuple[Placement[Reg, ArchNode], DAGNode]] = (
                    (Placement(binop_map, dict(zip(binop.qargs, node_ordering))), binop)
                    for node0, node1 in matching
                    for node_ordering in ((node0, node1), (node1, node0))
                    )

                min_placement = min(placements, key=placement_cost)

                if min_min_placement is not None:
                    min_min_placement = min(min_min_placement, min_placement, key=placement_cost)
                else:
                    min_min_placement = min_placement

            if min_min_placement is None:
                raise RuntimeError("The min_min_placement was not set.")

            min_place_cost = placement_cost(min_min_placement)[0]
            minimal_cost = min(minimal_cost, min_place_cost)  # Should only be set once.
            logger.debug(f"Cost changing from {self.placement_cost(current_placement)} "
                         f"→ {min_place_cost} / {2*minimal_cost}")

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

        logger.debug(f"Number of gates placed before: {placed_gates}/{total_gates}")

        ###
        # We now try to place the remaining gates on nodes
        # such that they are closer to their destination.
        # IDEA: apply some min-makespan instead of arbitrary ordering
        ###
        ordered_binops = list(binops)
        random.shuffle(ordered_binops)  # Ensure random ordering
        for binop in ordered_binops:
            logger.debug(f"Placing {binop}")
            # Enumerate all nodes that do not exceed the cost threshold for both qargs
            eligible_nodes = [[
                node for node in remaining_arch.nodes
                # Rough filter of possible locations
                if self.distance[current_mapping[qarg]][node] <= 2 * minimal_cost
                   # Exact filter
                   and placement_cost((Placement({qarg: current_mapping[qarg]},
                                                 {qarg: node}), binop.qargs))[0] <= 2 * minimal_cost
                ]
                for qarg in binop.qargs]
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
                logger.debug(
                    f"Placed { {qarg: current_mapping[qarg] for qarg in binop.qargs} } at {closest_nodes}, "
                    f"old dist: {self.distance[current_mapping[binop.qargs[0]]][current_mapping[binop.qargs[1]]]}, "
                    f"new: {self.distance[closest_nodes[0]][closest_nodes[1]]}.")
                current_placement += Placement({qarg: current_mapping[qarg]
                                                for qarg in binop.qargs},
                                               dict(zip(binop.qargs, closest_nodes)))
                remaining_arch.remove_nodes_from(closest_nodes)
                placed_gates += 1
            except ValueError:
                logger.debug("No eligible node pairs")

        after_cost = self.placement_cost(current_placement)[0]
        logger.debug(f"Number of gates placed: {placed_gates}/{total_gates} "
                     f"for a final cost of {after_cost}.")
        if after_cost > 2 * minimal_cost:
            logger.debug("New cost exceeded the threshold.")

        return current_placement.mapped_to


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
        """ Maps nodes that have many dependents in the DAG first.

        :param circuit: A circuit to execute
        :param node_dependencies: A mapping from node ids to number of dependents.
        :param current_mapping:
        :return:
        """
        if node_dependencies is None:
            if self.node_dependencies is None:
                logger.debug("No node dependencies given. Calculating node dependencies.")
                node_dependencies = DepthDependencyMapper.dependents_map(circuit)
            else:
                node_dependencies = self.node_dependencies

        binops = Mapper._binops_circuit(circuit)
        # Reshape the nodes to their qargs.
        binops_dependents: List[Tuple[DAGNode, int]] = [(binop, node_dependencies[binop])
                                                                for binop in binops]

        if not binops_dependents:
            return {}

        # After sorting by nr of descendents we discard the dependents field.
        binops_dependents = list(sorted(binops_dependents,
                                              key=operator.itemgetter(1),
                                              reverse=True))
        logger.debug(f"Max dependents: {binops_dependents[0][1]}.")
        binops = [el[0] for el in binops_dependents]

        remaining_arch = self.arch_graph.copy()
        current_placement: Placement[Reg, ArchNode] = Placement({}, {})
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
                ]
                for qarg in binop.qargs]
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
                logger.debug(f"Placed {min_closest_placement}, "
                             f"old dist: {self.distance[current_mapping[binop.qargs[0]]][current_mapping[binop.qargs[1]]]}, "
                             f"new: {minimal_distance}.")
                current_placement += min_closest_placement
                remaining_arch.remove_nodes_from(min_closest_placement.mapped_to.values())
                # Update minimal cost, because the cost function is not stable.
                # Otherwise future nodes may not be able to be placed anywhere
                # (since it will always exceed the cost.)
                new_minimal_cost = self.placement_cost(current_placement)[0]
                logger.debug(f"Old minimal_cost: {minimal_cost}, new: {new_minimal_cost}")
                minimal_cost = max(new_minimal_cost, 1)
            except ValueError:
                logger.debug(f"No eligible node pairs for {binop_map}.")

        return current_placement.mapped_to

    @staticmethod
    def dependents_map(circuit: DAGCircuit,
                       gate_costs: Mapping[Type[Instruction], int] = None) -> Mapping[DAGNode, int]:
        """Compute a mapping from dag nodes to the weighted longest path length from that node
        to the end of the circuit.

        If gate_costs is not given, all gate costs are assumed to be 1."""
        if gate_costs is None:
            gate_costs = defaultdict(lambda: 1)

        max_lengths: Dict[DAGNode, int] = {}
        reversed_layers: Iterator[DAGCircuit] = reversed(list(layer["graph"] for layer in circuit.layers()))
        for layer in reversed_layers:
            for node in layer.op_nodes():
                node_cost: int
                node_cost = gate_costs[node.op]
                max_lengths[node] = max((max_lengths[successor] for successor in circuit.successors(node)),
                                        default=0) + node_cost
        return max_lengths


class QiskitDepthMapper(DepthMapper[Reg, ArchNode]):

    def __init__(self,
                 arch_graph: nx.Graph,
                 arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                         Iterable[List[Swap[ArchNode]]]],
                 allow_swaps: bool = True) -> None:
        super().__init__(arch_graph, arch_permuter, allow_swaps)
        self.qiskit_size_mapper: QiskitSizeMapper[Reg, ArchNode] = \
            QiskitSizeMapper(arch_graph.to_directed(as_view=True),
                             # Convert depth permuter to size permuter
                             # This is only used for SimpleSizeMapper,
                             # so it's fine.
                             util.sequential_permuter(arch_permuter),
                             allow_swaps=allow_swaps)

    def map(self,
            circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode]) -> Mapping[Reg, ArchNode]:
        # We call the size_map function to avoid size-based preconditions.
        binops = Mapper._binops_circuit(circuit)
        return self.qiskit_size_mapper.size_map(circuit, current_mapping, binops)
