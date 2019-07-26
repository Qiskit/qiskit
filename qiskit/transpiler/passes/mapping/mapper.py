"""Defines superclass for all mappers"""
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

import sys
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Mapping, Set, FrozenSet, List, Optional

import networkx as nx
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler import AnalysisPass, Layout

ArchNode = TypeVar('ArchNode')
Reg = TypeVar('Reg')


class Mapper(Generic[Reg, ArchNode], AnalysisPass):
    """The abstract mapper class has a mapper method that maps a circuit (layer) to an architecture."""

    def __init__(self,
                 arch_graph: nx.Graph
                 ) -> None:
        super().__init__()
        self.arch_graph = arch_graph

        float_distances = nx.floyd_warshall(arch_graph)
        # Round distances to integers for stability. (All weights are 1)
        self.distance = {
            origin: {destination: sys.maxsize if distance == float('inf') else round(distance)
                     for destination, distance in float_distances[origin].items()}
            for origin in float_distances}

    def __call__(self, circuit: DAGCircuit, current_mapping: Mapping[Reg, ArchNode]) \
            -> Mapping[Reg, ArchNode]:
        return self.map(circuit, current_mapping)

    def run(self, dag: DAGCircuit) -> None:
        new_mapping = self.map(circuit=dag, current_mapping=self.property_set['layout'])
        self.property_set['new_mapping'] = Layout(input_dict=new_mapping)

    @abstractmethod
    def map(self, circuit: DAGCircuit,
            current_mapping: Mapping[Reg, ArchNode]) -> Mapping[Reg, ArchNode]:
        """Map (the layer of) the circuit to the architecture."""
        pass

    @staticmethod
    def construct_matching(arch_graph: nx.Graph) -> Set[FrozenSet[ArchNode]]:
        """Construct a maximum matching as a set of sets of endnodes."""
        return {frozenset(matching)
                for matching in nx.max_weight_matching(arch_graph, maxcardinality=True)}

    @staticmethod
    def _binops_circuit(circuit: DAGCircuit) -> List[DAGNode]:
        """Given the circuit, find the binary operations in the first layer of the circuit."""
        layer = Mapper.first_layer(circuit)
        # Layer can be None
        if layer:
            return layer.twoQ_gates()
        else:
            return list()

    @staticmethod
    def first_layer(circuit: DAGCircuit) -> Optional[DAGCircuit]:
        """Take the first layer of the DAGCircuit and return it."""
        try:
            return next(circuit.layers())["graph"]
        except StopIteration:
            return None
