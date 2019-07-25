"""mapping.size computes a good mapping to the architecture for a given circuit that minimizes size.

Given a circuit and an architecture graph we compute how to place that circuit on the architecture
so gates can be performed where we try to minimize the total number of gates.
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

import copy
import logging
from abc import abstractmethod
from typing import Dict, Set, Callable, Iterable, Mapping, TypeVar, Tuple, \
    Optional, FrozenSet, List

import networkx as nx
import numpy as np
from qiskit.dagcircuit import DAGCircuit, DAGNode

import arct.permutation as pm
import arct.permutation.util
from arct import scoring
from arct.mapping import util
from arct.mapping.mapper import Mapper
from arct.mapping.placement import Placement
from arct.permutation import Swap
from arct.util import first_layer

Reg = TypeVar('Reg')
ArchNode = TypeVar('ArchNode')

logger = logging.getLogger(__name__)

# Store the SWAP_COST as defined by scoring for quick access by _placement_cost
SWAP_COST = scoring.default_gate_costs()['swap']


class SizeMapper(Mapper[Reg, ArchNode]):
    """An abstract superclass for mappers that optimize for size.

    Internally caches outcomes of placement_cost for Placements in placement_costs."""

    def __init__(self, arch_graph: nx.DiGraph,
                 arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                         Iterable[Swap[ArchNode]]],
                 allow_swaps: bool = True) -> None:
        """Construct a SizeMapper object for mapping circuits to architectures, respecting size.

        :param arch_graph: The directed architecture graph.
        :param arch_permuter: The permuter on the architecture graph.
        :param allow_swaps: Whether to allow swaps or not. To increase performance it is recommended
            to set this to True and, if necessary, decompose SWAPs into CNOTs later."""
        super().__init__(arch_graph.to_undirected(as_view=True), allow_swaps=allow_swaps)
        self.arch_permuter = arch_permuter
        self.placement_costs: Dict[Placement[Reg, ArchNode], int] = {}

    def placement_cost(self, placement: Placement) -> int:
        """Find the cost of performing the placement in size.

        Will cache results for given small placements to speed up future computations."""
        if placement in self.placement_costs:
            return self.placement_costs[placement]

        # Swaps are symmetric so it is easy to compute the cost.
        if self.allow_swaps:
            # Count the number of swaps multiplied with the swap cost.
            return SWAP_COST * sum(1 for _ in self.arch_permuter(placement.arch_mapping))

        swaps = ([el] for el in self.arch_permuter(placement.arch_mapping))
        mapping_circuit = arct.permutation.util.circuit(swaps)
        inv_mapping = {v: k for k, v in mapping_circuit.inputmap.items()}
        # Fix CNOT directions.
        # util.direction_mapper(mapping_circuit.circuit, inv_mapping, self.arch_graph)
        # Then compute the cost of the mapping circuit in size.
        cost = scoring.cumulative_cost(mapping_circuit.circuit, inv_mapping, self.arch_graph)

        # Cache the result if the placement is small enough.
        # This prevents memory overruns but still caches the common case.
        # There are O(n!^2/(n-k)!^2) arch_mappings for k the size of the mapping.
        if len(placement.arch_mapping) <= 4:
            self.placement_costs[placement] = cost
        return cost

    @staticmethod
    def _inner_simple(binops: List[DAGNode],
                      current_mapping: Mapping[Reg, ArchNode],
                      remaining_arch: nx.DiGraph,
                      place_score: Callable[[Tuple[Placement[Reg, ArchNode], DAGNode]], int]) \
            -> Tuple[Placement[Reg, ArchNode], DAGNode]:
        """Internal function for computing a simple mapping of a set of gates.

        Will place the highest-scoring gate from binops_qargs
        at the location that maximize the score.

        Does not modify arguments."""
        max_max_placement: Optional[Tuple[Placement[Reg, ArchNode], DAGNode]] = None
        # Find the highest-scoring binop to perform and maximize the score.
        for binop in binops:
            binop_map = {
                qarg: current_mapping[qarg]
                for qarg in binop.qargs
                }
            # Try all edges and find the minimum cost placement.
            placements: Iterable[Tuple[Placement[Reg, ArchNode], DAGNode]] = \
                ((Placement(binop_map, dict(zip(binop.qargs, edge))), binop)
                 for edge_ab in remaining_arch.edges()
                 # Also check the reversed ordering.
                 for edge in [edge_ab, tuple(reversed(edge_ab))])
            max_placement = max(placements, key=place_score)

            # Maximize over all maximum scoring placements.
            if max_max_placement is not None:
                max_max_placement = max(max_max_placement, max_placement, key=place_score)
            else:
                max_max_placement = max_placement

        if max_max_placement is None:
            raise RuntimeError("No max_max_placement was produced. Were binops_qargs empty?")

        return max_max_placement

    def correction_cost(self, placement: Placement[Reg, ArchNode],
                        binops: Iterable[DAGNode]) -> int:
        """Compute the correction cost of the CNOT gates.

        :param placement: Asserts that mapped_to of this placement is of size 2.
        :param binops: The CNOT nodes."""
        # Hadamards to correct the CNOT
        return sum(4 for binop in binops
                   if
                   self.arch_graph.has_edge(placement.mapped_to[binop.qargs[0]], placement.mapped_to[binop.qargs[1]]))

    def saved_gates(self, place: Tuple[Placement[Reg, ArchNode], Iterable[DAGNode]],
                    current_placement: Placement[Reg, ArchNode] = None,
                    current_mapping: Mapping[Reg, ArchNode] = None) -> int:
        """Compute how many SWAP gates are saved by the placement.

        :param place: The placement to calculate the saved gates for.
        :param current_placement: The currently planned placement.
            Optional, but if used must also supply current_mapping.
        :param current_mapping: The current mapping of qubits to nodes.
        :return: The cost of gates saved. (>0 is good)"""
        placement, binops = place
        if current_placement is None or current_mapping is None:
            return -self.placement_cost(placement) - self.correction_cost(placement, binops)

        # Construct the mapping that will exist after applying the current placement.
        arch_mapping = {current_placement.current_mapping[k]: v
                        for k, v in current_placement.mapped_to.items()}
        swaps = self.arch_permuter(arch_mapping)
        # Compute what the generated swaps do with the current mapping.
        inv_new_mapping = {v: k for k, v in current_mapping.items()}
        pm.util.swap_permutation(([el] for el in swaps), inv_new_mapping,
                                 allow_missing_keys=True)
        new_mapping = {v: k for k, v in inv_new_mapping.items()}
        future_cost = 0
        for binop in binops:
            future_placement, future_qargs = self._inner_simple([binop],
                                                                new_mapping,
                                                                # The whole graph is available.
                                                                self.arch_graph,
                                                                # The first placement score is simple.
                                                                lambda t: self.saved_gates(
                                                                    (t[0], [t[1]])))
            future_cost += self.placement_cost(future_placement) \
                           + self.correction_cost(future_placement, [binop])

        return self.placement_cost(current_placement) \
               + future_cost \
               - self.placement_cost(current_placement + placement) \
               - self.correction_cost(placement, binops)

    def map(self,
            circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode]) -> Mapping[Reg, ArchNode]:
        """Map a (layer of the) given circuit to the architecture."""
        ###
        # We first check common preconditions for size mappers.
        # If all conditions are passed, then we call the size mapper.
        ###
        layer = first_layer(circuit)
        if not layer:
            return {}

        # If there are any 1-qubit gates left in the layer, do those first (and don't remap)
        if any(len(node.qargs) == 1 for node in layer.gate_nodes()):
            return {}

        binops = Mapper._binops_circuit(circuit)
        # No binops mean we don't care about the mapping.
        if not binops:
            return {}

        # If any binop can be mapped in place, return the trivial mapping
        for binop in binops:
            binop_map: Mapping[Reg, ArchNode] = {qarg: current_mapping[qarg]
                                                 for qarg in binop.qargs}
            if tuple(binop_map.values()) in self.arch_graph.to_undirected(as_view=True).edges:
                # Make sure that the qargs are mapped to their current location.
                return binop_map

        return self.size_map(circuit, current_mapping, binops)

    @abstractmethod
    def size_map(self,
                 circuit: DAGCircuit,
                 current_mapping: Mapping[Reg, ArchNode],
                 binops: List[DAGNode]) -> Mapping[Reg, ArchNode]:
        raise NotImplemented("Abstract method")


class SimpleSizeMapper(SizeMapper[Reg, ArchNode]):
    def size_map(self,
                 circuit: DAGCircuit,
                 current_mapping: Mapping[Reg, ArchNode],
                 binops: List[DAGNode]) -> Mapping[Reg, ArchNode]:
        """Perform a simple greedy mapping of the cheapest gate to the architecture."""

        # Peel off the first layer of operations for the circuit
        # so that we can assign operations to the architecture.

        def simple_saved_gates(place: Tuple[Placement[Reg, ArchNode], DAGNode]) -> int:
            """We have to repackage the second argument of place into an iterable."""
            return self.saved_gates((place[0], [place[1]]))

        return self._inner_simple(binops, current_mapping, self.arch_graph,
                                  simple_saved_gates)[0].mapped_to


class GreedySizeMapper(SizeMapper[Reg, ArchNode]):

    def __init__(self, arch_graph: nx.DiGraph,
                 arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                         Iterable[Swap[ArchNode]]],
                 allow_swaps: bool = True,
                 max_first: bool = True) -> None:
        super().__init__(arch_graph, arch_permuter, allow_swaps)
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


class ExtensionSizeMapper(SizeMapper[Reg, ArchNode]):
    def __init__(self, arch_graph: nx.DiGraph,
                 arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                         Iterable[Swap[ArchNode]]],
                 allow_swaps: bool = True,
                 lookahead: bool = False) -> None:
        super().__init__(arch_graph, arch_permuter, allow_swaps)
        self.lookahead = lookahead

    def size_map(self,
                 circuit: DAGCircuit,
                 current_mapping: Mapping[Reg, ArchNode],
                 binops: List[DAGNode]) -> Mapping[Reg, ArchNode]:
        """Place the cheapest gate and try to extend the placement with further good placements."""
        # Peel off the first layer of operations for the circuit
        # so that we can assign operations to the architecture.
        remaining_arch = self.arch_graph.copy()
        current_placement: Optional[Placement] = None

        def placement_score(place: Tuple[Placement[Reg, ArchNode], DAGNode]) -> int:
            """Returns a score for this placement, the higher the better."""
            placement, binop = place
            saved_now = self.saved_gates((placement, [binop]), current_placement, current_mapping)
            if not self.lookahead:
                return saved_now

            ###
            # Find the next gate that will be placed, and where.
            # See if the next placement will be improved by this placement.
            ###
            # TODO: This is now O(n)
            remaining_binops = binops[:]
            remaining_binops.remove(binop)
            new_remaining_arch = \
                remaining_arch.subgraph(node for node in remaining_arch.nodes()
                                        if node not in placement.mapped_to.values())
            if remaining_binops and len(new_remaining_arch.edges()) > 0:
                cur_place: Placement[Reg, ArchNode]
                if current_placement is None:
                    cur_place = Placement({}, {})
                else:
                    cur_place = current_placement

                # The extra cost incurred by placing 'placement'.
                place_cost_diff = self.placement_cost(placement + cur_place) \
                                  - self.placement_cost(cur_place)

                def placement_diff(place: Tuple[Placement[Reg, ArchNode], DAGNode]) -> int:
                    """Approximates saved_gates but it easier to compute.

                    It computes the cost of placing a given gate plus (separately) the 'placement'
                    gate versus the cost of placing everything together.
                    This does not require a lookahead."""
                    return self.placement_cost(place[0] + cur_place) + place_cost_diff \
                           - self.placement_cost(place[0] + placement + cur_place)

                next_placement = self._inner_simple(remaining_binops,
                                                    current_mapping,
                                                    new_remaining_arch,
                                                    placement_diff)
                diff_next = placement_diff(next_placement)
            else:
                diff_next = 0

            # Is it better to place the gate now or wait until the next iteration?
            # If the result is zero or less then it's not worse to place the gate now.
            return saved_now + diff_next

        placed_gates = 0
        total_gates = len(binops)
        while binops and remaining_arch.edges():
            max_max_placement: Optional[Tuple[Placement[Reg, ArchNode], DAGNode]] = None
            for binop in binops:
                binop_map: Mapping[Reg, ArchNode] = {
                    qarg: current_mapping[qarg]
                    for qarg in binop.qargs
                    }

                # Try all edges and find the minimum cost placement.
                all_edges = {e for directed_edge in remaining_arch.edges()
                             for e in [directed_edge, tuple(reversed(directed_edge))]}
                placements = ((Placement(binop_map, dict(zip(binop.qargs, edge))), binop)
                              for edge in all_edges)

                # Find the cost of placing this gate given the current placement,
                # versus a placement without the current placement.
                # If this is positive it means that placing this gate now is advantageous.
                max_placement = max(placements, key=placement_score)

                if max_max_placement is None:
                    max_max_placement = max_placement
                else:
                    max_max_placement = max(max_max_placement, max_placement, key=placement_score)

            if max_max_placement is None:
                raise RuntimeError("The max_max_placement is None. Was binops_qargs empty?")

            # Place the cheapest binops, but only if it is advantageous by the placement_score.
            if current_placement is None:
                # Always place at least one binop.
                current_placement = max_max_placement[0]
            else:
                score = placement_score(max_max_placement)
                if score < 0:
                    # There are no advantageous gates to place left.
                    break
                if score > 0:
                    logger.debug(f"Saved cost! Placement score: {score}")
                current_placement += max_max_placement[0]

            # Remove the placed binop from datastructure.
            binops.remove(max_max_placement[1])
            # The nodes are now in use and can no longer be used for anything else.
            remaining_arch.remove_nodes_from(max_max_placement[0].mapped_to.values())
            placed_gates += 1

        logger.debug(f"Number of gates placed: {placed_gates}/{total_gates}")
        if current_placement is None:
            raise RuntimeError("The current_placement is None. Somehow it did not get set.")
        return current_placement.mapped_to


class QiskitSizeMapper(SizeMapper[Reg, ArchNode]):
    """A mapper that combines the QISkit mapper and the extension size mapper."""

    def __init__(self, arch_graph: nx.DiGraph,
                 arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                         Iterable[Swap[ArchNode]]],
                 allow_swaps: bool = True,
                 trials: int = 40,
                 seed: Optional[int] = None) -> None:
        super().__init__(arch_graph, arch_permuter, allow_swaps)
        self.simple_mapper: SimpleSizeMapper[Reg, ArchNode] = \
            SimpleSizeMapper(arch_graph, arch_permuter, allow_swaps=allow_swaps)
        self.trials = trials
        self.seed = seed

    def size_map(self, circuit: DAGCircuit,
                 current_mapping: Mapping[Reg, ArchNode],
                 binops: List[DAGNode]) -> Mapping[Reg, ArchNode]:
        """A mapper based on qiskit.mapping.swap_mapper"""
        if self.seed is not None:
            np.random.seed(self.seed)

        # Filter out registers that are not used by binary operations from the current mapping.
        binop_regs = {qarg for binop in binops for qarg in binop.qargs}
        binop_current_mapping = {k: v for k, v in current_mapping.items() if k in binop_regs}

        # Try to map everything using the qiskit mapper.
        # Begin loop over trials of randomized algorithm
        trial_layouts = (self._qiskit_trial(binops, binop_current_mapping)
                         for _ in range(self.trials))
        # Filter out None results
        filtered_layouts = (trial for trial in trial_layouts if trial is not None)
        try:
            # Minimize over size
            best_layout = min(filtered_layouts, key=lambda t: t[0])
            logger.debug("qiskit mapper: done")
            return best_layout[1]
        except ValueError:
            logger.debug("qiskit mapper: failed!")
            # The qiskit mapper did not find a mapping so we just map a single gate.
            return self.simple_mapper.size_map(circuit, current_mapping, binops)

    def _qiskit_trial(self,
                      binops: List[DAGNode],
                      initial_layout: Mapping[Reg, ArchNode]) \
            -> Optional[Tuple[int, Mapping[Reg, ArchNode]]]:
        """One trial in computing a mapping as used in qiskit.

        Tries to swap edges that reduce the cost function up to a maximimum size."""
        trial_layout = copy.copy(initial_layout)
        inv_trial_layout: Mapping[ArchNode, Reg] = {v: k for k, v in trial_layout.items()}

        # Compute Sergey's randomized distance.
        # IDEA: Rewrite to numpy matrix
        xi: Dict[ArchNode, Dict[ArchNode, float]] = {}
        for i in self.arch_graph.nodes:
            xi[i] = {}
        for i in self.arch_graph.nodes:
            for j in self.arch_graph.nodes:
                scale = 1 + np.random.normal(0, 1 / self.arch_graph.number_of_nodes())
                xi[i][j] = scale * self.distance[i][j] ** 2
                xi[j][i] = xi[i][j]

        def cost(layout: Mapping[Reg, ArchNode]) -> float:
            """Compute the objective cost function."""
            return sum([xi[layout[binop.qargs[0]]][layout[binop.qargs[1]]] for binop in binops])

        def swap(node0: ArchNode, node1: ArchNode) \
                -> Tuple[Mapping[Reg, ArchNode], Mapping[ArchNode, Reg]]:
            """Swap qarg0 and qarg1 based on trial layout and inv_trial layout.

            Supports partial mappings."""
            inv_new_layout = dict(inv_trial_layout)
            qarg0: Optional[Reg] = inv_new_layout.pop(node0, None)
            qarg1: Optional[Reg] = inv_new_layout.pop(node1, None)
            if qarg1 is not None:
                inv_new_layout[node0] = qarg1
            if qarg0 is not None:
                inv_new_layout[node1] = qarg0

            return {v: k for k, v in inv_new_layout.items()}, inv_new_layout

        # Loop over sizes up to a max size (nr of swaps) of |V|^2
        size = 0
        for _ in range(len(self.arch_graph.nodes) ** 2):
            # Find the layout which minimize the objective function
            # by trying all possible swaps.
            new_layouts = (swap(*edge) for edge in self.arch_graph.edges)
            min_layout = min(new_layouts, key=lambda t: cost(t[0]))

            # Were there any good choices?
            if cost(min_layout[0]) < cost(trial_layout):
                trial_layout, inv_trial_layout = min_layout
                size += 1
            else:
                # If there weren't any good choices, there also won't be in the future. So abort.
                break

        # Compute the coupling graph distance
        # If all gates can be applied now, we have found a layout.
        dist = sum(self.distance[trial_layout[binop.qargs[0]]][trial_layout[binop.qargs[1]]]
                   for binop in binops)
        if dist == len(binops):
            # We have succeeded in finding a layout
            return size, trial_layout
        return None
