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