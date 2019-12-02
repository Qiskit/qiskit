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

"""mapping.size computes a good mapping to the architecture for a given circuit that minimizes size.

Given a circuit and an architecture graph we compute how to place that circuit on the architecture
so gates can be performed where we try to minimize the total number of gates.
"""

import logging
from abc import abstractmethod
from typing import Dict, Callable, Iterable, Mapping, TypeVar, Tuple, \
    Optional, List

import networkx as nx

import qiskit.transpiler.routing as pm
import qiskit.transpiler.routing.util  # pylint: disable=unused-import
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler.passes.mapping.mapper import Mapper
from qiskit.transpiler.passes.mapping.placement import Placement
from qiskit.transpiler.routing import Swap

Reg = TypeVar('Reg')
ArchNode = TypeVar('ArchNode')

logger = logging.getLogger(__name__)


class SizeMapper(Mapper[Reg, ArchNode]):
    """An abstract superclass for mappers that optimize for size.

    Internally caches outcomes of placement_cost for Placements in placement_costs.
    This may consume quite a bit of memory.
    """

    def __init__(self, arch_graph: nx.DiGraph,
                 arch_permuter: Callable[[Mapping[ArchNode, ArchNode]],
                                         Iterable[Swap[ArchNode]]],
                 max_placement_size: int = 4) -> None:
        """Construct a SizeMapper object for mapping circuits to architectures, respecting size.

        Args:
            arch_graph: The directed architecture graph.
            arch_permuter: The permuter on the architecture graph.
            max_placement_size: The maximum size of a placement to cache.
               This memory requires scale superexponentially in this parameter.
               There are O(n!^2/(n-k)!^2) arch_mappings for k the size of the mapping.
        """
        super().__init__(arch_graph.to_undirected(as_view=True))
        self.arch_permuter = arch_permuter
        self.placement_costs = {}  # type: Dict[Placement[Reg, ArchNode], int]
        self.max_placement_size = max_placement_size

    def placement_cost(self, placement: Placement) -> int:
        """Find the cost of performing the placement in size.

        Will cache results for given small placements to speed up future computations."""
        if placement in self.placement_costs:
            return self.placement_costs[placement]

        # Count the number of swaps
        seq_cost = sum(1 for _ in self.arch_permuter(placement.arch_mapping))

        # Cache the result if the placement is small enough.
        # This prevents memory overruns but still caches the common case.
        # There are O(n!^2/(n-k)!^2) arch_mappings for k the size of the mapping.
        if len(placement.arch_mapping) <= self.max_placement_size:
            self.placement_costs[placement] = seq_cost
        return seq_cost

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
        max_max_placement = None  # type: Optional[Tuple[Placement[Reg, ArchNode], DAGNode]]
        # Find the highest-scoring binop to perform and maximize the score.
        for binop in binops:
            binop_map = {
                qarg: current_mapping[qarg]
                for qarg in binop.qargs
            }
            # Try all edges and find the minimum cost placement.
            placements = ((Placement(binop_map, dict(zip(binop.qargs, edge))), binop)
                          for edge_ab in remaining_arch.edges()
                          # Also check the reversed ordering.
                          for edge in [edge_ab, tuple(reversed(edge_ab))]
                          )  # type: Iterable[Tuple[Placement[Reg, ArchNode], DAGNode]]
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

        Args:
          placement: Asserts that mapped_to of this placement is of size 2.
          binops: The CNOT nodes.
          placement:
          binops:

        Returns:
          The (integer) cost of correcting
        """
        # Hadamards to correct the CNOT
        return sum(4 for binop in binops
                   if
                   self.arch_graph.has_edge(placement.mapped_to[binop.qargs[0]],
                                            placement.mapped_to[binop.qargs[1]]))

    def saved_gates(self, place: Tuple[Placement[Reg, ArchNode], Iterable[DAGNode]],
                    current_placement: Placement[Reg, ArchNode] = None,
                    current_mapping: Mapping[Reg, ArchNode] = None) -> int:
        """Compute how many SWAP gates are saved by the placement.

        Args:
          place: The placement to calculate the saved gates for.
          current_placement: The currently planned placement.
            Optional, but if used must also supply current_mapping.
          current_mapping: The current mapping of qubits to nodes.

        Returns:
          The cost of gates saved. (>0 is good)
        """
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
            future_placement = \
                self._inner_simple([binop], new_mapping,  # The whole graph is available.
                                   self.arch_graph,  # The first placement score is simple.
                                   lambda t: self.saved_gates((t[0], [t[1]])))[0]
            future_cost += self.placement_cost(future_placement) + \
                self.correction_cost(future_placement, [binop])

        return self.placement_cost(current_placement) + \
            future_cost - \
            self.placement_cost(current_placement + placement) - \
            self.correction_cost(placement, binops)

    def map(self,
            circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode]) -> Mapping[Reg, ArchNode]:
        """Map a (layer of the) given circuit to the architecture."""
        ###
        # We first check common preconditions for size mappers.
        # If all conditions are passed, then we call the size mapper.
        ###
        layer = Mapper.first_layer(circuit)
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
            binop_map = {qarg: current_mapping[qarg]
                         for qarg in binop.qargs}  # type: Mapping[Reg, ArchNode]
            if tuple(binop_map.values()) in self.arch_graph.to_undirected(as_view=True).edges:
                # Make sure that the qargs are mapped to their current location.
                return binop_map

        return self.size_map(circuit, current_mapping, binops)

    @abstractmethod
    def size_map(self,
                 circuit: DAGCircuit,
                 current_mapping: Mapping[Reg, ArchNode],
                 binops: List[DAGNode]) -> Mapping[Reg, ArchNode]:
        """Implement the given mapping while optimizing for size"""
        raise NotImplementedError("Abstract method")
