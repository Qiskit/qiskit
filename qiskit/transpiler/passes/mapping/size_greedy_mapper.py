from typing import Callable, Mapping, Iterable, List, Set, FrozenSet, Optional, Tuple

import networkx as nx

from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.passes.mapping.mapper import Mapper
from qiskit.transpiler.passes.mapping.placement import Placement
from qiskit.transpiler.passes.mapping.size import SizeMapper, Reg, ArchNode
from qiskit.transpiler.routing import Swap


class GreedySizeMapper(SizeMapper[Reg, ArchNode]):

    def __init__(self, arch_graph: nx.DiGraph,
                 arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                         Iterable[Swap[ArchNode]]],
                 max_first: bool = True) -> None:
        super().__init__(arch_graph, arch_permuter)
        self.max_first = max_first

    def size_map(self,
                 circuit: DAGCircuit,
                 current_mapping: Mapping[Reg, ArchNode],
                 binops: List[DAGNode]) -> Mapping[Reg, ArchNode]:
        """
        Provides a mapping that maps possibly multiple gates of the circuit to the architecture.

        If a chosen mapping has a cost increase associated to it, then we try to perform the operation
        locally instead.

        :param circuit: A circuit to execute
        :param current_mapping:
        :param binops: The binary operations to map
        :return: A partial mapping
        """

        # The maximum matching gives us the maximum number of edges
        # for use in two-qubit ("binary") operations.
        # Note: maximum matching assumes undirected graph.
        remaining_arch = self.arch_graph.copy()
        matching: Set[FrozenSet[ArchNode]] = Mapper.construct_matching(remaining_arch)
        current_placement: Placement[Reg, ArchNode] = Placement({}, {})

        current_placement_cost = lambda place: self.placement_cost(current_placement + place[0])

        # We wish to minimize the size of the circuit.
        # We try to find a good mapping of all binary ops to the matching,
        # such that the overhead cost of the mapping circuit is minimized.
        while binops and matching:
            # Find the most expensive or cheapest binop to perform (depending on max_first)
            # and minimize its cost.
            extremal_min_placement: Optional[Tuple[Placement[Reg, ArchNode], DAGNode]] = None
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

                min_placement = min(placements, key=current_placement_cost)

                if extremal_min_placement is None:
                    extremal_min_placement = min_placement
                else:
                    if self.max_first:
                        extremal_min_placement = max(extremal_min_placement, min_placement,
                                                     key=current_placement_cost)
                    else:
                        extremal_min_placement = min(extremal_min_placement, min_placement,
                                                     key=current_placement_cost)

            if extremal_min_placement is None:
                raise RuntimeError("The extremal_min_placement was not set.")

            # Place the most expensive binop
            current_placement += extremal_min_placement[0]
            binops.remove(extremal_min_placement[1])

            # The nodes are now in use and can no longer be used for anything else.
            remaining_arch.remove_nodes_from(extremal_min_placement[0].mapped_to.values())
            if extremal_min_placement[0].is_local(self.arch_graph):
                # If we used a local placement outside of the matching,
                # recompute the matching to reflect that.
                matching = Mapper.construct_matching(remaining_arch)
            else:
                # Otherwise both directions of the matching are now used.
                matching.remove(frozenset(extremal_min_placement[0].mapped_to.values()))

        return current_placement.mapped_to