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

"""Test cases for the mapping.depth package"""

from typing import Tuple, List, Mapping, Iterator
from unittest import TestCase, mock

import networkx as nx

from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions import CnotGate, HGate
from qiskit.transpiler import routing as rt
from qiskit.transpiler.passes.mapping.depth_bounded_mapper import BoundedDepthMapper
from qiskit.transpiler.passes.mapping.depth_greedy_mapper import GreedyDepthMapper
from qiskit.transpiler.passes.mapping.depth_incremental_mapper import IncrementalDepthMapper
from qiskit.transpiler.passes.mapping.depth_simple_mapper import SimpleDepthMapper
from qiskit.transpiler.routing import modular, path  # pylint: disable=unused-import


def dummy_permuter(
        perm: Mapping  # pylint: disable=unused-argument
) -> Iterator[List]:
    """Dummy permuter that returns trivial swaps"""
    return iter([])


Reg = Tuple[str, int]


class TestDepthMapper(TestCase):
    """The test cases."""

    def setUp(self) -> None:
        self.circuit = DAGCircuit()
        self.arch_graph = nx.Graph()

    def test_map_simple_empty(self) -> None:
        """Test the mapping of an empty circuit on 2 interacting qubits."""
        q = QuantumRegister(2)
        self.circuit.add_qreg(q)
        self.arch_graph.add_edge(0, 1)
        mapper = SimpleDepthMapper(self.arch_graph, dummy_permuter)

        out = mapper.map(self.circuit)
        self.assertEqual({}, out)

    def test_map_simple_small(self) -> None:
        """Test the mapping of a single CNOT onto a 2-qubit path"""
        q = QuantumRegister(2)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.arch_graph.add_edge(0, 1, weight=2)
        mapper = SimpleDepthMapper(self.arch_graph, dummy_permuter)

        out = mapper.map(self.circuit)
        self.assertEqual({q[0]: 0, q[1]: 1}, out)

    def test_map_simple_small_2(self) -> None:
        """Test the mapping of a parallel Hadamard and CNOT on 0, 1<->2 (weight=2)"""
        q = QuantumRegister(3)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(HGate(), [q[2]])  # Hadamard
        # G: 0, 1<->2
        self.arch_graph.add_node(0)
        self.arch_graph.add_edge(1, 2, weight=2)
        mapper = SimpleDepthMapper(self.arch_graph, dummy_permuter)

        out = mapper.map(self.circuit)
        # q0 -> 1, q1 -> 2, q2 -> 0
        self.assertEqual({q[0]: 1, q[1]: 2}, out)

    def test_map_simple_two_layers(self) -> None:
        """Test mapping two sequential CNOTs"""
        q = QuantumRegister(3)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[1], q[2]])
        self.arch_graph.add_edge(1, 2)
        mapper = SimpleDepthMapper(self.arch_graph, dummy_permuter)

        out = mapper.map(self.circuit)
        # must be mapped to qubits 1 and 2
        self.assertEqual({1, 2}, set(out.values()))

    def test_map_incremental_empty(self) -> None:
        """Test mapping an empty circuit."""
        q = QuantumRegister(2)
        self.circuit.add_qreg(q)
        self.arch_graph.add_edge(0, 1)
        current_mapping = {q[i]: 1 - i for i in [0, 1]}

        mapper = IncrementalDepthMapper(self.arch_graph, lambda p: rt.modular.permute(p, 2, 2))

        out = mapper(self.circuit, current_mapping)
        self.assertEqual({}, out)

    def test_map_incremental_small(self) -> None:
        """Test mapping a single CNOT to the architecture (weight=2)"""
        q = QuantumRegister(2)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.arch_graph.add_edge(0, 1, weight=2)
        current_mapping0 = {q[i]: i for i in [0, 1]}
        current_mapping1 = {q[i]: 1 - i for i in [0, 1]}

        mapper = IncrementalDepthMapper(self.arch_graph, lambda p: rt.modular.permute(p, 2, 2))

        # Qubits should not be moved needlessly.
        out0 = mapper.map(self.circuit, current_mapping0)
        self.assertEqual({q[0]: 0, q[1]: 1}, out0)
        out1 = mapper.map(self.circuit, current_mapping1)
        self.assertEqual({q[0]: 1, q[1]: 0}, out1)

    def test_map_incremental_two_layers_move(self) -> None:
        """Test mapping two sequential CNOTs that require a shuffling of qubits."""
        q = QuantumRegister(3)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[1], q[2]])
        # Modular graph, m = n = 2
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(0, 2)
        self.arch_graph.add_edge(2, 3)

        mapper = IncrementalDepthMapper(self.arch_graph, lambda p: rt.modular.permute(p, 2, 2))

        current_mapping = {q[0]: 1, q[1]: 2, q[2]: 3}

        out = mapper.map(self.circuit, current_mapping)
        self.assertIn(frozenset(out.items()), {frozenset({q[0]: 1, q[1]: 0}.items()),
                                               frozenset({q[0]: 0, q[1]: 2}.items())})

    def test_incremental_partial_move(self) -> None:
        """Test if the incremental mapper moves outer nodes incrementally closer."""
        q = QuantumRegister(4)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[2], q[3]])

        self.arch_graph = nx.path_graph(8)
        current_mapping = {q[0]: 2, q[1]: 5,
                           q[2]: 0, q[3]: 7}

        mapper = IncrementalDepthMapper(self.arch_graph,
                                        lambda p: rt.path.permute_path_partial(p, length=8))
        out = mapper.map(self.circuit, current_mapping)
        # The mapper should have moved q0 and q1 adjacent, and q2 and q3 both one step closer.
        self.assertEqual({q[0]: 3, q[1]: 4,
                          q[2]: 1, q[3]: 6}, out)

    def test_incremental_partial_nomove(self) -> None:
        """Test if the incremental mapper moves outer nodes incrementally closer."""
        q = QuantumRegister(4)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[2], q[3]])

        self.arch_graph = nx.path_graph(6)
        current_mapping = {q[0]: 1, q[1]: 4,
                           q[2]: 0, q[3]: 5}

        mapper = IncrementalDepthMapper(self.arch_graph,
                                        lambda p: rt.path.permute_path_partial(p, length=8))
        out = mapper.map(self.circuit, current_mapping)
        # The mapper should have moved q0 and q1 adjacent, and q2 and q3 both one step closer.
        self.assertEqual({q[0]: 2, q[1]: 3,
                          q[2]: 0, q[3]: 5}, out)

    def test_map_bounded_small(self) -> None:
        """Test mapping a single CNOT to the architecture (weight=2)"""
        q = QuantumRegister(2)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.arch_graph.add_edge(0, 1, weight=2)
        current_mapping0 = {q[i]: i for i in [0, 1]}
        current_mapping1 = {q[i]: 1 - i for i in [0, 1]}

        mapper = BoundedDepthMapper(self.arch_graph, lambda p: rt.modular.permute(p, 2, 2))

        # Qubits should not be moved needlessly.
        out0 = mapper.map(self.circuit, current_mapping0)
        self.assertEqual({q[0]: 0, q[1]: 1}, out0)
        out1 = mapper.map(self.circuit, current_mapping1)
        self.assertEqual({q[0]: 1, q[1]: 0}, out1)

    def test_map_bounded_two_layers_move(self) -> None:
        """Test mapping two sequential CNOTs that require a shuffling of qubits."""
        q = QuantumRegister(3)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[1], q[2]])
        # Modular graph, m = n = 2
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(0, 2)
        self.arch_graph.add_edge(2, 3)

        mapper = BoundedDepthMapper(self.arch_graph, lambda p: rt.modular.permute(p, 2, 2))

        current_mapping = {q[0]: 1, q[1]: 2, q[2]: 3}

        out = mapper.map(self.circuit, current_mapping)
        self.assertIn(frozenset(out.items()), {frozenset({q[0]: 1, q[1]: 0}.items()),
                                               frozenset({q[0]: 0, q[1]: 2}.items())})

    def test_map_greedy_empty(self) -> None:
        """Test mapping an empty circuit."""
        q = QuantumRegister(2)
        self.circuit.add_qreg(q)
        self.arch_graph.add_edge(0, 1)
        current_mapping = {q[i]: 1 - i for i in [0, 1]}

        mapper = GreedyDepthMapper(self.arch_graph, lambda p: rt.modular.permute(p, 2, 2))

        out = mapper.map(self.circuit, current_mapping)
        self.assertEqual({}, out)

    def test_map_greedy_small(self) -> None:
        """Test mapping a single CNOT to the architecture (weight=2)"""
        q = QuantumRegister(2)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.arch_graph.add_edge(0, 1, weight=2)
        current_mapping0 = {q[i]: i for i in [0, 1]}
        current_mapping1 = {q[i]: 1 - i for i in [0, 1]}

        mapper = GreedyDepthMapper(self.arch_graph, lambda p: rt.modular.permute(p, 2, 2))

        # Qubits should not be moved needlessly.
        out0 = mapper.map(self.circuit, current_mapping0)
        self.assertEqual({q[0]: 0, q[1]: 1}, out0)
        out1 = mapper.map(self.circuit, current_mapping1)
        self.assertEqual({q[0]: 1, q[1]: 0}, out1)

    def test_map_greedy_small_2(self) -> None:
        """Test mapping a parallel Hadamard and CNOT to 0, 1<->2 (weight=2)"""
        q = QuantumRegister(3)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(HGate(), [q[2]])  # Hadamard
        # G: 0, 1<->2
        self.arch_graph.add_node(0)
        self.arch_graph.add_edge(1, 2, weight=2)
        current_mapping = {q[i]: i for i in range(3)}

        # Make sure that the matching does not contain the edge 1<->2
        matching_mock = mock.create_autospec(nx.max_weight_matching, return_value={(0, 1)})
        with mock.patch('networkx.max_weight_matching', matching_mock):
            mapper = GreedyDepthMapper(self.arch_graph,
                                       lambda p: rt.path.permute_path_partial(p, length=3))
            out = mapper.map(self.circuit, current_mapping)
        expected = current_mapping.copy()
        del expected[q[2]]
        self.assertEqual(expected, out)

    def test_map_greedy_cnots_modular(self) -> None:
        """Test mapping two sequential CNOTs on modular graph."""
        q = QuantumRegister(3)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[1], q[2]])
        # Modular graph, m = n = 2
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(0, 2)
        self.arch_graph.add_edge(2, 3)

        mapper = GreedyDepthMapper(self.arch_graph, lambda p: rt.modular.permute(p, 2, 2))

        current_mapping = {q[0]: 1, q[1]: 0, q[2]: 2}

        out = mapper.map(self.circuit, current_mapping)
        # Identity mapping, because first op can be performed in-place.
        self.assertEqual({q[0]: 1, q[1]: 0}, out)

    def test_map_greedy_two_layers_move(self) -> None:
        """Test mapping two sequential CNOTs that require a shuffling of qubits."""
        q = QuantumRegister(3)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[1], q[2]])
        # Modular graph, m = n = 2
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(0, 2)
        self.arch_graph.add_edge(2, 3)

        mapper = GreedyDepthMapper(self.arch_graph, lambda p: rt.modular.permute(p, 2, 2))

        current_mapping = {q[0]: 1, q[1]: 2, q[2]: 3}

        out = mapper.map(self.circuit, current_mapping)
        # Because of matching limitations q1 should be moved to 0 or to 3.
        self.assertIn(frozenset(out.items()),
                      {frozenset({q[0]: 1, q[1]: 0}.items()),
                       frozenset({q[0]: 2, q[1]: 3}.items()), })

    def test_map_greedy_nomatching(self) -> None:
        """Test mapping that should use an edge not inside the matching graph."""
        q = QuantumRegister(3)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[1], q[2]])
        # Modular graph, m = n = 2
        self.arch_graph = nx.complete_graph(4)

        current_mapping = {q[0]: 0, q[1]: 1, q[2]: 2}
        mapper = GreedyDepthMapper(self.arch_graph, lambda p: rt.modular.permute(p, 2, 2))

        out = mapper.map(self.circuit, current_mapping)
        expected = current_mapping.copy()
        del expected[q[0]]
        self.assertEqual(expected, out)
