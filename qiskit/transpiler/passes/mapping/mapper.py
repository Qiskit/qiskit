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

"""Defines superclass for all mappers"""

import sys
from abc import abstractmethod
from typing import Generic, TypeVar, Mapping, Set, FrozenSet, List, Optional

import networkx as nx
from qiskit.dagcircuit import DAGCircuit, DAGNode
from qiskit.transpiler import AnalysisPass, Layout
from qiskit.transpiler.basepasses import MetaPass

ArchNode = TypeVar('ArchNode')
Reg = TypeVar('Reg')


class GenericMetaPass(MetaPass, type(Generic)):
    """A superclass for the Mapper class to be able to use Generic.
    
    The metaclass of Generic was removed in 3.7 so we resolve it dynamically.
    
    The need for this workaround was fixed in 3.7+: https://github.com/python/typing/issues/449
    """
    pass


class Mapper(Generic[Reg, ArchNode], AnalysisPass, metaclass=GenericMetaPass):
    """The abstract mapper class has a mapper method that maps a circuitlayer to an architecture."""

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
        """Map (the layer of) the circuit to the architecture.

        Args:
          circuit: The circuit to map to the architecture
          current_mapping: The currently active mapping of qubits to architecture nodes.

        Returns:
            A partial mapping of qubits to architecture nodes.
        """
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
