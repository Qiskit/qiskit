"""mapping.size computes a good mapping to the architecture for a given circuit that minimizes size.

Given a circuit and an architecture graph we compute how to place that circuit on the architecture
so gates can be performed where we try to minimize the total number of gates.
"""
import copy
import itertools
import logging
import typing
from typing import Dict, Set, Callable, Iterable, Mapping, TypeVar, List, Generic, Tuple, \
    Iterator, Optional

import networkx as nx
import numpy as np
from qiskit.dagcircuit import DAGCircuit

from .. import permutation as pm
from .. import scoring
from ..mapping import util
from ..mapping.placement import Placement
from ..permutation import Swap
from ..util import first_layer

Reg = TypeVar('Reg')
ArchNode = TypeVar('ArchNode')

logger = logging.getLogger(__name__)


class SizeMapper(Generic[Reg, ArchNode]):
    """A container for various mapping functions that optimize for size.

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
        self.arch_graph = arch_graph
        self.arch_permuter = arch_permuter
        self.allow_swaps = allow_swaps
        self.placement_costs: Dict[Placement[Reg, ArchNode], int] = {}
        self.graph_distance: Dict[ArchNode, Dict[ArchNode, int]] = None

    def simple(self,
               circuit: DAGCircuit,
               current_mapping: Mapping[Reg, ArchNode]) -> Mapping[Reg, ArchNode]:
        """Perform a simple greedy mapping of the cheapest gate to the architecture."""
        # Peel off the first layer of operations for the circuit
        # so that we can assign operations to the architecture.
        layer = first_layer(circuit)
        if not layer:
            return {}

        node_data = [n[1] for n in layer.multi_graph.nodes(data=True)]
        # If there are any 1-qubit gates left in the layer, do those first (and don't remap)
        if any(len(n["qargs"]) == 1 for n in node_data if n["type"] == "op"):
            return {}

        # Reshape the nodes to their qargs.
        binops_qargs: Set[Tuple[Reg, Reg]] = {
            # only keep qargs field as an immutable tuple.
            typing.cast(Tuple[Reg, Reg], tuple(n["qargs"]))
            for n in node_data
            # Filter out op nodes that are not barriers and are 2-qubit operations
            if n["type"] == "op" and n["name"] != "barrier" and len(n["qargs"]) == 2}

        # No binops mean we don't care about the mapping.
        if not binops_qargs:
            return {}

        def simple_saved_gates(place: Tuple[Placement[Reg, ArchNode], Tuple[Reg, Reg]]) -> int:
            """We have to repackage the second argument of place into an iterable."""
            return self.saved_gates((place[0], [place[1]]))

        return self._inner_simple(binops_qargs, current_mapping, self.arch_graph,
                                  simple_saved_gates)[0].mapped_to

    @staticmethod
    def _inner_simple(binops_qargs: Set[Tuple[Reg, Reg]],
                      current_mapping: Mapping[Reg, ArchNode],
                      remaining_arch: nx.DiGraph,
                      place_score: Callable[[Tuple[Placement[Reg, ArchNode], Tuple[Reg, Reg]]],
                                            int]) \
            -> Tuple[Placement[Reg, ArchNode], Tuple[Reg, Reg]]:
        """Internal function for computing a simple mapping of a set of gates.

        Does not modify arguments.

        :param binops_qargs: A set containing the quantum registers of 2-qubit operations involved
            in this layer of the DAG circuit.
        :param current_mapping: The current mapping of quantum registers to architecture nodes.
        :param place_score: A function that gives a score to a placement.
        :returns: A Placement of a gate together with the qargs of that gate."""
        max_max_placement: Tuple[Placement[Reg, ArchNode], Tuple[Reg, Reg]] = None
        # Find the cheapest binop to perform and minimize its cost.
        for binop_qargs in binops_qargs:
            binop_map = {
                qarg: current_mapping[qarg]
                for qarg in binop_qargs
                }
            # Try all edges and find the minimum cost placement.
            placements: Iterable[Tuple[Placement[Reg, ArchNode], Tuple[Reg, Reg]]] = \
                ((Placement(binop_map, dict(zip(binop_qargs, edge))), binop_qargs)
                 for edge_ab in remaining_arch.edges()
                 # Also check the reversed ordering.
                 for edge in [edge_ab, tuple(reversed(edge_ab))])
            max_placement = max(placements, key=place_score)

            # Maximize over all maximum scoring placements.
            if max_max_placement:
                max_max_placement = max(max_max_placement, max_placement, key=place_score)
            else:
                max_max_placement = max_placement
        return max_max_placement

    def greedy(self,
               circuit: DAGCircuit,
               current_mapping: Mapping[Reg, ArchNode],
               max_first: bool = True) -> Mapping[Reg, ArchNode]:
        """
        Provides a mapping that maps possibly multiple gates of the circuit to the architecture.

        If a chosen mapping has a cost increase associated to it,
        then we try to perform the operation locally instead.

        :param circuit: A circuit to execute
        :param arch_graph: The architecture graph,optionally with weights on edges.
            Default weights are 1.
        :param current_mapping:
        :param arch_permuter:
        :param max_first: Place the most expensive binop first.
        :return: A partial mapping
        """
        # Peel off the first layer of operations for the circuit
        # so that we can assign operations to the architecture.
        layer = first_layer(circuit)
        if not layer:
            return {}

        node_data = map(lambda n: n[1], layer.multi_graph.nodes(data=True))

        # Reshape the nodes to their qargs.
        binops_qargs: Set[Tuple[Reg, Reg]] = {
            # only keep qargs field as an immutable tuple.
            typing.cast(Tuple[Reg, Reg], tuple(n["qargs"]))
            for n in node_data
            # Filter out op nodes that are not barriers and are 2-qubit operations
            if n["type"] == "op" and n["name"] != "barrier" and len(n["qargs"]) == 2}

        # No binops mean we don't care about the mapping.
        if not binops_qargs:
            return {}

        # If any binop can be mapped in place, return the trivial mapping
        if any((current_mapping[binop_qargs[0]], current_mapping[binop_qargs[1]])
               in self.arch_graph.to_undirected(as_view=True).edges
               for binop_qargs in binops_qargs):
            return {}

        # The maximum matching gives us the maximum number of edges
        # for use in two-qubit ("binary") operations.
        # Note: maximum matching assumes undirected graph.
        matching: Dict[ArchNode, ArchNode] = nx.max_weight_matching(
            self.arch_graph.to_undirected(as_view=True), maxcardinality=True)
        remaining_arch = self.arch_graph.copy()
        current_placement: Placement[Reg, ArchNode] = Placement({}, {})

        current_placement_cost = lambda place: self.placement_cost(current_placement + place[0])

        # We wish to minimize the size of the circuit.
        # We try to find a good mapping of all binary ops to the matching,
        # such that the overhead cost of the mapping circuit is minimized.
        while binops_qargs and matching:
            # Find the most expensive or cheapest binop to perform (depending on max_first)
            # and minimize its cost.
            extremal_min_placement: Tuple[Placement[Reg, ArchNode], Tuple[Reg, Reg]] = None
            for binop_qargs in binops_qargs:
                binop_map = {
                    qarg: current_mapping[qarg]
                    for qarg in binop_qargs
                    }
                # Try all matchings and find the minimum cost placement.
                placements: Iterable[Tuple[Placement[Reg, ArchNode], Tuple[Reg, Reg]]] = [
                    (Placement(binop_map, dict(zip(binop_qargs, matching_nodes))), binop_qargs)
                    for matching_nodes in matching
                    ]

                min_placement = min(placements, key=current_placement_cost)

                if extremal_min_placement is None:
                    extremal_min_placement = min_placement
                else:
                    if max_first:
                        extremal_min_placement = max(extremal_min_placement, min_placement,
                                                     key=current_placement_cost)
                    else:
                        extremal_min_placement = min(extremal_min_placement, min_placement,
                                                     key=current_placement_cost)
            # Place the most expensive binop
            current_placement += extremal_min_placement[0]
            binops_qargs.remove(extremal_min_placement[1])

            # The nodes are now in use and can no longer be used for anything else.
            remaining_arch.remove_nodes_from(extremal_min_placement[0].mapped_to.values())
            if extremal_min_placement[0].is_local(self.arch_graph):
                # If we used a local placement outside of the matching,
                # recompute the matching to reflect that.
                matching = nx.max_weight_matching(remaining_arch.to_undirected(as_view=True),
                                                  maxcardinality=True)
            else:
                # Otherwise both directions of the matching are now used.
                for mapped_to in extremal_min_placement[0].mapped_to.values():
                    matching = { m for m in matching if m[0] != mapped_to }

        return current_placement.mapped_to

    def simple_extend(self,
                      circuit: DAGCircuit,
                      current_mapping: Mapping[Reg, ArchNode],
                      lookahead: bool = False) -> Mapping[Reg, ArchNode]:
        """Place the cheapest gate and try to extend the placement with further good placements.

        :param circuit: The DAGCircuit to place the first layer of.
        :param current_mapping: The current placement of quantum registers on the architecture
            graph.
        :param lookahead: Enables lookahead in the extension mapper.
            Warning: Very slow!
        :return: A (partial) mapping that places at least one gate in this layer
            on the architecture.
        """
        # Peel off the first layer of operations for the circuit
        # so that we can assign operations to the architecture.
        layer = first_layer(circuit)
        if not layer:
            return {}

        node_data = [n[1] for n in layer.multi_graph.nodes(data=True)]
        # If there are any 1-qubit gates left in the layer, do those first (and don't remap)
        if any(len(n["qargs"]) == 1 for n in node_data if n["type"] == "op"):
            return {}

        # Reshape the nodes to their qargs.
        binops_qargs: Set[Tuple[Reg, Reg]] = {
            # only keep qargs field as an immutable tuple.
            typing.cast(Tuple[Reg, Reg], tuple(n["qargs"]))
            for n in node_data
            # Filter out op nodes that are not barriers and are 2-qubit operations
            if n["type"] == "op" and n["name"] != "barrier" and len(n["qargs"]) == 2}

        # No binops mean we don't care about the mapping.
        if not binops_qargs:
            return {}

        # If any binop can be mapped in place, return the trivial mapping
        if any(self.arch_graph.to_undirected(as_view=True).has_edge(current_mapping[binop_qargs[0]],
                                                                    current_mapping[binop_qargs[1]])
               for binop_qargs in binops_qargs):
            return {}

        remaining_arch = self.arch_graph.copy()
        current_placement: Placement = None

        def placement_score(place: Tuple[Placement[Reg, ArchNode], Tuple[Reg, Reg]]) -> int:
            """Returns a score for this placement, the higher the better."""
            placement, binop_qargs = place
            saved_now = self.saved_gates((placement, [binop_qargs]),
                                         current_placement, current_mapping)
            if not lookahead:
                return saved_now

            ###
            # Find the next gate that will be placed, and where.
            # See if the next placement will be improved by this placement.
            ###
            remaining_binops = binops_qargs.difference({binop_qargs})
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

                def placement_diff(place: Tuple[Placement[Reg, ArchNode], Tuple[Reg, Reg]]) -> int:
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
        total_gates = len(binops_qargs)
        while binops_qargs and remaining_arch.edges():
            max_max_placement: Tuple[Placement[Reg, ArchNode], Tuple[Reg, Reg]] = None
            for binop_qargs in binops_qargs:
                binop_map: Mapping[Reg, ArchNode] = {
                    qarg: current_mapping[qarg]
                    for qarg in binop_qargs
                    }

                # Try all edges and find the minimum cost placement.
                placements = ((Placement(binop_map, dict(zip(binop_qargs, edge))), binop_qargs)
                              for directed_edge in remaining_arch.edges()
                              for edge in [directed_edge, tuple(reversed(directed_edge))])

                # Find the cost of placing this gate given the current placement,
                # versus a placement without the current placement.
                # If this is positive it means that placing this gate now is advantageous.
                max_placement = max(placements, key=placement_score)

                if max_max_placement is None:
                    max_max_placement = max_placement
                else:
                    max_max_placement = max(max_max_placement, max_placement, key=placement_score)
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
            binops_qargs.remove(max_max_placement[1])
            # The nodes are now in use and can no longer be used for anything else.
            remaining_arch.remove_nodes_from(max_max_placement[0].mapped_to.values())
            placed_gates += 1

        logger.debug(f"Number of gates placed: {placed_gates}/{total_gates}")
        return current_placement.mapped_to

    def qiskit_mapper(self,
                      circuit: DAGCircuit,
                      current_mapping: Mapping[Reg, ArchNode],
                      trials: int = 40,
                      seed: int = None,
                      lookahead: bool = False) -> Mapping[Reg, ArchNode]:
        """This mapper tries to minimize the size of the mapping circuit by an objective function.

        The main loop is in _qiskit_trial which tries swaps that lower the objective function,
        until eventually all gates within the first layer of the given circuit are adjacent.

        If this fails we use the extension mapper to try to find an alternative solution.
        If that fails we just place a single gate and proceed.

        :param circuit: The circuit to place the first layer of.
        :param current_mapping: The current placement of quantum registers on the architecture
            graph.
        :param trials: The number of trials to optimize over in the randomized subroutine.
        :param seed: The random seed for the trials.
        :param lookahead: Enables lookahead in the extension mapper.
            Warning: Very slow!
        :return: A (partial) mapping that places at least one gate in this layer on the
            architecture.
        """
        if seed is not None:
            np.random.seed(seed)

        layer = first_layer(circuit)
        if not layer:
            return {}

        node_data = [n[1] for n in layer.multi_graph.nodes(data=True)]
        # If there are any 1-qubit gates left in the layer, do those first (and don't remap)
        if any(len(n["qargs"]) == 1 for n in node_data if n["type"] == "op"):
            return {}

        # Reshape the nodes to their qargs.
        binops_qargs: Set[Tuple[Reg, Reg]] = {
            # only keep qargs field as an immutable tuple.
            typing.cast(Tuple[Reg, Reg], tuple(n["qargs"]))
            for n in node_data
            # Filter out op nodes that are not barriers and are 2-qubit operations
            if n["type"] == "op" and n["name"] != "barrier" and len(n["qargs"]) == 2}

        # No binops mean we don't care about the mapping.
        if not binops_qargs:
            return {}

        if self.graph_distance is None:
            # Compute the distance
            self.graph_distance = dict(nx.all_pairs_shortest_path_length(
                self.arch_graph.to_undirected(as_view=True)))

        # If any binop can be mapped in place, return the trivial mapping
        if any(self.arch_graph.to_undirected(as_view=True)
                       .has_edge(current_mapping[binop_qargs[0]],
                                 current_mapping[binop_qargs[1]])
               for binop_qargs in binops_qargs):
            return {}

        # Filter out registers that are not used by binary operations from the current mapping.
        binop_regs = {qarg for qargs in binops_qargs for qarg in qargs}
        current_mapping = {k: v for k, v in current_mapping.items() if k in binop_regs}

        # Try to map everything using the qiskit mapper.
        # Begin loop over trials of randomized algorithm
        trial_layouts = (self._qiskit_trial(binops_qargs, copy.copy(current_mapping))
                         for _ in range(trials))
        # Filter out None results
        trial_layouts = (trial for trial in trial_layouts if trial is not None)
        try:
            # Minimize over size
            best_layout = min(trial_layouts, key=lambda t: t[0])
            logger.debug("qiskit mapper: done")
            return best_layout[1]
        except ValueError:
            logger.debug("qiskit mapper: failed!")
            # The qiskit mapper did not find a mapping,
            # as an alternative use our extension algorithm without lookahead.
            extension_layout = self.simple_extend(circuit, current_mapping, lookahead=lookahead)
            mapped_qargs = [(qarg0, qarg1) for qarg0, qarg1 in binops_qargs
                            if qarg0 in extension_layout and qarg1 in extension_layout]
            # Only use the extension result if it actually saved gates.
            extension_saved = self.saved_gates((Placement(current_mapping, extension_layout),
                                                mapped_qargs),
                                               Placement({}, {}),
                                               current_mapping)
            logger.debug(f"Extension saved {extension_saved} cost.")
            # Saved half a swap per binop after the first one.
            if extension_saved // max(1, len(mapped_qargs) // 2 - 1) >= 17:
                logger.debug(f"Using extension layout")
                return extension_layout

            # Otherwise just map a single gate.
            return self.simple(circuit, current_mapping)

    def _qiskit_trial(self,
                      binops_qargs: Set[Tuple[Reg, Reg]],
                      trial_layout: Mapping[Reg, ArchNode]) \
            -> Optional[Tuple[int, Mapping[Reg, ArchNode]]]:
        """One trial in computing a mapping according to a cost function.

        The cost function is the same as used by qiskit's swap_mapper algorithm. Tries to swap edges
        that reduce the cost function up to a maximimum size. But instead of minimizing the depth
        of the circuit, we only minimize for the size.

        :param binops_qargs: A set containing the quantum registers of 2-qubit operations involved
            in this layer of the DAG circuit.
        :param trial_layout: The starting mapping of registers to architecture nodes for this trial.
        :returns: Either the number of swaps with a mapping, or None.
        """
        inv_trial_layout: Mapping[ArchNode, Reg] = {v: k for k, v in trial_layout.items()}

        # Compute Sergey's randomized distance.
        nr_nodes = len(self.arch_graph.nodes)
        # IDEA: Rewrite to numpy matrix
        xi: Dict[ArchNode, Dict[ArchNode, float]] = {}
        for i in self.arch_graph.nodes:
            xi[i] = {}
        for i in self.arch_graph.nodes:
            for j in self.arch_graph.nodes:
                scale = 1 + np.random.normal(0, 1 / nr_nodes)
                xi[i][j] = scale * self.graph_distance[i][j] ** 2
                xi[j][i] = xi[i][j]

        def cost(layout: Mapping[Reg, ArchNode]) -> float:
            """Compute the objective cost function."""
            return sum([xi[layout[qarg0]][layout[qarg1]] for qarg0, qarg1 in binops_qargs])

        def swap(node0: ArchNode, node1: ArchNode) \
                -> Tuple[Mapping[Reg, ArchNode], Mapping[ArchNode, Reg]]:
            """Swap qarg0 and qarg1 based on trial layout and inv_trial layout.

            Supports partial mappings."""
            inv_new_layout = dict(inv_trial_layout)
            qarg0 = inv_new_layout.pop(node0, None)
            qarg1 = inv_new_layout.pop(node1, None)
            if qarg1 is not None:
                inv_new_layout[node0] = qarg1
            if qarg0 is not None:
                inv_new_layout[node1] = qarg0

            return {v: k for k, v in inv_new_layout.items()}, inv_new_layout

        # Loop over depths sizes up to a max size of |V|^2/8
        size = 0
        for _ in range(len(self.arch_graph.nodes) ** 2 // 8):
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
        dist = sum(self.graph_distance[trial_layout[qarg0]][trial_layout[qarg1]]
                   for qarg0, qarg1 in binops_qargs)
        if dist == len(binops_qargs):
            # We have succeeded in finding a layout
            return size, trial_layout
        return None

    def correction_cost(self,
                        placement: Placement[Reg, ArchNode],
                        qargs: Iterable[Tuple[Reg, Reg]]) -> int:
        """Compute the correction cost of this CNOT gate.

        :param placement: Asserts that mapped_to of this placement is of size 2.
        :param qargs: The registers of this CNOT."""
        # Hadamards to correct the CNOT
        return sum(4 for qarg0, qarg1 in qargs
                   if
                   self.arch_graph.has_edge(placement.mapped_to[qarg0], placement.mapped_to[qarg1]))

    def saved_gates(self, place: Tuple[Placement[Reg, ArchNode], Iterable[Tuple[Reg, Reg]]],
                    current_placement: Placement[Reg, ArchNode] = None,
                    current_mapping: Mapping[Reg, ArchNode] = None) -> int:
        """Compute how many SWAP gates are saved by the placement.

        :param place: The placement to calculate the saved gates for.
        :param current_placement: The currently planned placement.
            Optional, but if used must also supply current_mapping.
        :param current_mapping: The current mapping of qubits to nodes.
        :return: The cost of gates saved. (>0 is good)"""
        placement, binops_qargs = place
        if current_placement is None or current_mapping is None:
            return -self.placement_cost(placement) - self.correction_cost(placement, binops_qargs)

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
        for binop_qargs in binops_qargs:
            future_placement = self._inner_simple({binop_qargs}, new_mapping,
                                                  # The whole graph is available.
                                                  self.arch_graph,
                                                  # The first placement score is simple.
                                                  lambda t: self.saved_gates((t[0], [t[1]])))[0]
            future_cost += self.placement_cost(future_placement) \
                           + self.correction_cost(future_placement, [binop_qargs])

        return self.placement_cost(current_placement) \
               + future_cost \
               - self.placement_cost(current_placement + placement) \
               - self.correction_cost(placement, binops_qargs)

    def placement_cost(self, placement: Placement) -> int:
        """Find the cost of performing the placement in size.

        Will cache results for given small placements to speed up future computations."""
        if placement in self.placement_costs:
            return self.placement_costs[placement]

        cost = _placement_cost(placement, self.arch_graph, self.arch_permuter,
                               allow_swaps=self.allow_swaps)
        # Cache the result if the placement is small enough.
        # This prevents memory overruns but still caches the common case.
        # There are O(n!^2/(n-k)!^2) arch_mappings for k the size of the mapping.
        if len(placement.arch_mapping) <= 4:
            self.placement_costs[placement] = cost
        return cost


# Store the SWAP_COST as defined by scoring for quick access by _placement_cost
SWAP_COST = scoring.default_gate_costs()['swap']


def _placement_cost(placement: Placement[Reg, ArchNode],
                    arch_graph: nx.DiGraph,
                    arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                            Iterable[Swap[ArchNode]]],
                    allow_swaps: bool = True) -> int:
    """Compute the cost of performing the placement in size."""
    # Swaps are symmetric so it is easy to compute the cost.
    if allow_swaps:
        swaps = list(arch_permuter(placement.arch_mapping))
        return SWAP_COST * len(swaps)

    def par_permuter(mapping: Mapping[ArchNode, ArchNode]) -> Iterable[List[Swap[ArchNode]]]:
        """Reshape the sequential permuter to a parallel permuter."""
        return map(lambda el: [el], arch_permuter(mapping))

    _, mapping_circuit = placement.mapping_circuit(par_permuter, allow_swaps=allow_swaps)
    inv_mapping = {v: k for k, v in mapping_circuit.inputmap.items()}
    # Fix CNOT directions.
    util.direction_mapper(mapping_circuit.circuit, inv_mapping, arch_graph)
    # Then compute the cost of the mapping circuit in size.
    return scoring.cumulative_cost(mapping_circuit.circuit,
                                   inv_mapping,
                                   arch_graph)


def matching_sets(edges: Iterable[Tuple[ArchNode, ArchNode]], size: int) \
        -> Iterator[List[Tuple[ArchNode, ArchNode]]]:
    """Compute all sets of edges that are a matching of the given size."""

    def is_matching(edges: Iterable[Tuple]) -> bool:
        """Checks if the nodes of the matching are unique"""
        nodes = [node for edge in edges for node in edge]
        return len(set(nodes)) == len(nodes)

    matchings_edges: Iterator[Tuple[Tuple[ArchNode, ArchNode], ...]] = \
        (permutation_edges for permutation_edges in itertools.permutations(edges, size)
         if is_matching(permutation_edges))
    # Now also generate all possible subsets with edges reversed.
    for matching_edges in matchings_edges:
        # Pick edges to reverse
        reversed_edge_indices = (set(index)
                                 for i in range(len(matching_edges) + 1)
                                 for index in
                                 itertools.combinations(range(len(matching_edges)), i))
        for index in reversed_edge_indices:
            yield [typing.cast(Tuple[ArchNode, ArchNode], tuple(reversed(edge)))
                   if i in index else edge
                   for i, edge in enumerate(matching_edges)]
