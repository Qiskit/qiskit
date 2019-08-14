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

"""Test cases for the mapping.size package"""

import random
from typing import Tuple, List, Any, Mapping, TypeVar, Type
from unittest import TestCase

import networkx as nx
import numpy as np

from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit
from qiskit.extensions import CnotGate, HGate
from qiskit.transpiler.passes.mapping.mapper import Mapper
from qiskit.transpiler.passes.mapping.size import SizeMapper
from qiskit.transpiler.passes.mapping.size_extension_mapper import ExtensionSizeMapper
from qiskit.transpiler.passes.mapping.size_greedy_mapper import GreedySizeMapper
from qiskit.transpiler.passes.mapping.size_qiskit_mapper import QiskitSizeMapper
from qiskit.transpiler.passes.mapping.size_simple_mapper import SimpleSizeMapper
from qiskit.transpiler.routing import complete, modular, path
from qiskit.transpiler.routing.general import ApproximateTokenSwapper
from qiskit.transpiler.routing.util import sequential_permuter

ArchNode = TypeVar('ArchNode')
_V = TypeVar('_V')
Reg = Tuple[_V, int]


def dummy_permuter(mapping: Mapping[Any, Any]) -> List[Any]:
    """A permuter that always returns the empty list"""
    # pylint: disable=unused-argument
    return []


class TestSizeMapper(TestCase):
    """The test cases."""

    def setUp(self) -> None:
        random.seed(0)
        np.random.seed(0)

        self.circuit = DAGCircuit()
        self.arch_graph = nx.DiGraph()

    def empty_circuit_test(self, mapper: Type[SizeMapper]) -> None:
        """Check if the empty circuit is mapped correctly"""
        q = QuantumRegister(2)
        self.circuit.add_qreg(q)
        self.arch_graph.add_edge(0, 1)
        current_mapping = {q[i]: 1 - i for i in [0, 1]}
        permuter = sequential_permuter(lambda m: complete.partial_permute(m, [0, 1]))
        out = mapper(self.arch_graph, permuter).map(self.circuit, current_mapping)
        self.assertEqual({}, out)

    def test_map_simple_empty(self) -> None:
        """Test the mapping of an empty circuit on 2 interacting qubits."""
        self.empty_circuit_test(SimpleSizeMapper)

    def test_map_greedy_empty(self) -> None:
        """Test mapping an empty circuit."""
        self.empty_circuit_test(GreedySizeMapper)

    def test_map_extend_empty(self) -> None:
        """Test the mapping of an empty circuit on 2 interacting qubits."""
        self.empty_circuit_test(ExtensionSizeMapper)

    def test_qiskit_map_empty(self) -> None:
        """Test the mapping of an empty circuit on 2 interacting qubits."""
        self.empty_circuit_test(QiskitSizeMapper)

    def test_map_simple_small(self) -> None:
        """Test the mapping of a single CNOT onto a 2-qubit path"""
        q = QuantumRegister(2)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.arch_graph.add_edge(0, 1, weight=2)
        current_mapping = {q[i]: i for i in range(2)}

        seq_path_permuter = sequential_permuter(path.permute_path_partial)
        mapper = SimpleSizeMapper(self.arch_graph, seq_path_permuter)

        out = mapper.map(self.circuit, current_mapping)
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
        current_mapping = {q[i]: i for i in range(3)}
        mapper = SimpleSizeMapper(self.arch_graph, dummy_permuter)

        out = mapper.map(self.circuit, current_mapping)
        # The output is empty because there is a 1-qubit gate.
        self.assertEqual({}, out)

    def test_map_simple_two_layers(self) -> None:
        """Test mapping two sequential CNOTs"""
        q = QuantumRegister(3)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[1], q[2]])
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(1, 2)
        current_mapping = {q[i]: i for i in range(3)}
        mapper = SimpleSizeMapper(self.arch_graph,
                                  sequential_permuter(path.permute_path_partial))

        out = mapper.map(self.circuit, current_mapping)
        self.assertEqual({q[0]: 0, q[1]: 1}, out)

    def test_map_simple_choice(self) -> None:
        """Given two gates map the cheapest one."""
        q = QuantumRegister(4)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[2], q[3]])
        self.arch_graph = nx.path_graph(8)
        current_mapping = {q[0]: 1, q[1]: 6, q[2]: 0, q[3]: 7}
        mapper = SimpleSizeMapper(self.arch_graph, sequential_permuter(path.permute_path_partial))

        out = mapper.map(self.circuit, current_mapping)
        # Only q0 and q1 should be mapped since they are closer.
        self.assertEqual({q[0], q[1]}, set(out.keys()))

    def test_map_greedy_small(self) -> None:
        """Test mapping a single CNOT to the architecture (weight=2)"""
        q = QuantumRegister(2)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.arch_graph.add_edge(0, 1, weight=2)
        current_mapping0 = {q[i]: i for i in [0, 1]}
        current_mapping1 = {q[i]: 1 - i for i in [0, 1]}
        mapper = GreedySizeMapper(self.arch_graph,
                                  sequential_permuter(
                                      lambda m: complete.partial_permute(m, [0, 1])))

        # Qubits should not be moved needlessly.
        out0 = mapper.map(self.circuit, current_mapping0)
        self.assertEqual(current_mapping0, out0)
        out1 = mapper.map(self.circuit, current_mapping1)
        self.assertEqual(current_mapping1, out1)

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
        mapper = GreedySizeMapper(self.arch_graph, dummy_permuter)

        out = mapper.map(self.circuit, current_mapping)
        # The mapper does nothing, because there is a single-qubit gate that can be performed.
        self.assertEqual({}, out)

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

        current_mapping = {q[0]: 1, q[1]: 0, q[2]: 2}
        mapper = GreedySizeMapper(self.arch_graph,
                                  sequential_permuter(lambda p: modular.permute(p, 2, 2)))

        out = mapper.map(self.circuit, current_mapping)
        # The qubits for the first gate should not be moved.
        del current_mapping[q[2]]
        self.assertEqual(current_mapping, out)

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
        swapper = ApproximateTokenSwapper(
            self.arch_graph.to_undirected(as_view=True))  # type: ApproximateTokenSwapper[int]

        current_mapping = {q[0]: 1, q[1]: 2, q[2]: 3}
        mapper = GreedySizeMapper(self.arch_graph, swapper.map)

        out = mapper.map(self.circuit, current_mapping)
        # Because of matching limitations q1 should be moved to 0.
        output_mapping = {q[0]: 1, q[1]: 0}
        self.assertEqual(output_mapping, out)

    def test_map_extend_small(self) -> None:
        """Test the mapping of a single CNOT onto a 2-qubit path"""
        q = QuantumRegister(2)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.arch_graph.add_edge(0, 1, weight=2)
        current_mapping = {q[i]: i for i in range(2)}

        par_path_permuter = sequential_permuter(path.permute_path_partial)
        mapper = ExtensionSizeMapper(self.arch_graph, par_path_permuter)

        out = mapper.map(self.circuit, current_mapping)
        self.assertEqual(current_mapping, out)

    def test_map_extend_small_2(self) -> None:
        """Test the mapping of a parallel Hadamard and CNOT on 0, 1<->2 (weight=2)"""
        q = QuantumRegister(3)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(HGate(), [q[2]])  # Hadamard
        # G: 0, 1<->2
        self.arch_graph.add_node(0)
        self.arch_graph.add_edge(1, 2, weight=2)
        current_mapping = {q[i]: i for i in range(3)}
        mapper = ExtensionSizeMapper(self.arch_graph, dummy_permuter)

        out = mapper.map(self.circuit, current_mapping)
        # The output is empty because there is a 1-qubit gate.
        self.assertEqual({}, out)

    def test_map_extend_two_layers(self) -> None:
        """Test mapping two sequential CNOTs"""
        q = QuantumRegister(3)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[1], q[2]])
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(1, 2)
        current_mapping = {q[i]: i for i in range(3)}
        mapper = ExtensionSizeMapper(self.arch_graph,
                                     sequential_permuter(
                                         path.permute_path_partial))

        out = mapper.map(self.circuit, current_mapping)
        del current_mapping[q[2]]
        self.assertEqual(current_mapping, out)

    def test_map_extend_choice(self) -> None:
        """Given two gates map both."""
        q = QuantumRegister(4)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[2], q[3]])
        self.arch_graph = nx.path_graph(8).to_directed()
        current_mapping = {q[0]: 1, q[1]: 6, q[2]: 0, q[3]: 7}
        swapper = ApproximateTokenSwapper(
            self.arch_graph.to_undirected(as_view=True))  # type: ApproximateTokenSwapper[int]
        mapper = ExtensionSizeMapper(self.arch_graph, swapper.map)

        out = mapper.map(self.circuit, current_mapping)
        # All qubits are mapped, since it takes 10 SWAPs to map both at once
        # and 4 + 6 SWAPS to map one after the other.
        self.assertEqual(set(current_mapping.keys()), set(out.keys()))
        swaps = list(swapper.map({current_mapping[k]: v for k, v in out.items()}))
        self.assertEqual(10, len(swaps))

    def test_map_extend_free(self) -> None:
        """Test that the extension mapper will take advantage of a "free" SWAP gate."""
        q = QuantumRegister(4)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[2], q[3]])
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(0, 2)
        self.arch_graph.add_edge(0, 3)
        self.arch_graph.add_edge(3, 4)
        self.arch_graph.add_edge(3, 5)
        current_mapping = {q[0]: 1, q[2]: 2, q[1]: 4, q[3]: 5}
        swapper = ApproximateTokenSwapper(
            self.arch_graph.to_undirected(as_view=True))  # type: ApproximateTokenSwapper[int]
        mapper = ExtensionSizeMapper(self.arch_graph, swapper.map, lookahead=True)

        out = mapper.map(self.circuit, current_mapping)
        # The mapper takes advantage of a "free" SWAP along 0-3 and maps all qubits in one step
        self.assertEqual(set(current_mapping.keys()), set(out.keys()))

    def test_qiskit_map_small(self) -> None:
        """Test the mapping of a single CNOT onto a 2-qubit path"""
        q = QuantumRegister(2)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.arch_graph.add_edge(0, 1, weight=2)
        current_mapping = {q[i]: i for i in range(2)}

        par_path_permuter = sequential_permuter(path.permute_path_partial)
        mapper = QiskitSizeMapper(self.arch_graph, par_path_permuter)

        out = mapper.map(self.circuit, current_mapping)
        self.assertEqual(current_mapping, out)

    def test_qiskit_map_small_2(self) -> None:
        """Test the mapping of a parallel Hadamard and CNOT on 0, 1<->2 (weight=2)"""
        q = QuantumRegister(3)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(HGate(), [q[2]])  # Hadamard
        # G: 0, 1<->2
        self.arch_graph.add_node(0)
        self.arch_graph.add_edge(1, 2, weight=2)
        current_mapping = {q[i]: i for i in range(3)}
        mapper = QiskitSizeMapper(self.arch_graph, dummy_permuter)

        out = mapper.map(self.circuit, current_mapping)
        # The output is empty because there is a 1-qubit gate.
        self.assertEqual({}, out)

    def test_qiskit_map_two_layers(self) -> None:
        """Test mapping two sequential CNOTs"""
        q = QuantumRegister(3)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[1], q[2]])
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(1, 2)
        current_mapping = {q[i]: i for i in range(3)}
        mapper = QiskitSizeMapper(self.arch_graph,
                                  sequential_permuter(path.permute_path_partial))

        out = mapper.map(self.circuit, current_mapping)
        del current_mapping[q[2]]
        self.assertEqual(current_mapping, out)

    def test_qiskit_map_choice(self) -> None:
        """Given two gates map both."""
        q = QuantumRegister(4)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[2], q[3]])
        self.arch_graph = nx.path_graph(8).to_directed(as_view=True)
        current_mapping = {q[0]: 1, q[1]: 6, q[2]: 0, q[3]: 7}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected(as_view=True))
        mapper = QiskitSizeMapper(self.arch_graph, swapper.map, seed=0)

        out = mapper.map(self.circuit, current_mapping)
        # All qubits are mapped, since it takes 10 SWAPs to map both at once
        # and 4 + 6 SWAPS to map one after the other.
        self.assertEqual(set(current_mapping.keys()), set(out.keys()))
        swaps = list(swapper.map({current_mapping[k]: v for k, v in out.items()}))
        self.assertEqual(10, len(swaps))

    def test_qiskit_map_partial(self) -> None:
        """Checks if the qiskit mapper will return a partial mapping for single-qubit gates."""
        q = QuantumRegister(5)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[2], q[3]])
        self.circuit.apply_operation_back(HGate(), [q[4]])  # Hadamard
        self.arch_graph = nx.path_graph(8).to_directed(as_view=True)
        partial_mapping = {q[0]: 1, q[1]: 6, q[2]: 0, q[3]: 7}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected(as_view=True))
        mapper = QiskitSizeMapper(self.arch_graph, swapper.map)

        layer = Mapper.first_layer(self.circuit)
        if layer is None:
            self.fail("Layer unexpectedly was None.")

        binops = layer.twoQ_gates()
        result = mapper._qiskit_trial(binops, partial_mapping)
        if result is None:
            self.fail("The output was None.")
        out = result[1]
        self.assertEqual(partial_mapping.keys(), set(out.keys()))
        swaps = list(swapper.map({partial_mapping[k]: v for k, v in out.items()}))
        self.assertEqual(10, len(swaps))

    def test_qiskit_map_free(self) -> None:
        """Test that the extension mapper will take advantage of a "free" SWAP gate."""
        q = QuantumRegister(4)
        self.circuit.add_qreg(q)
        self.circuit.apply_operation_back(CnotGate(), [q[0], q[1]])
        self.circuit.apply_operation_back(CnotGate(), [q[2], q[3]])
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(0, 2)
        self.arch_graph.add_edge(0, 3)
        self.arch_graph.add_edge(3, 4)
        self.arch_graph.add_edge(3, 5)
        current_mapping = {q[0]: 1, q[2]: 2, q[1]: 4, q[3]: 5}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected(as_view=True))
        mapper = QiskitSizeMapper(self.arch_graph, swapper.map)

        out = mapper.map(self.circuit, current_mapping)
        # The mapper takes advantage of a "free" SWAP along 0-3 and maps all qubits in one step
        self.assertEqual(set(current_mapping.keys()), set(out.keys()))

    def test_qiskit_map_subsets(self) -> None:
        """Test that larger subsets of gates can be mapped simultaneously."""
        q = QuantumRegister(6)
        self.circuit.add_qreg(q)
        for i in range(0, 6, 2):
            self.circuit.apply_operation_back(CnotGate(), [q[i], q[i + 1]])
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(0, 2)
        self.arch_graph.add_edge(2, 6)
        self.arch_graph.add_edge(0, 3)
        self.arch_graph.add_edge(3, 4)
        self.arch_graph.add_edge(3, 5)
        self.arch_graph.add_edge(5, 7)
        current_mapping = {q[0]: 1, q[1]: 4,
                           q[2]: 2, q[3]: 5,
                           q[4]: 6, q[5]: 7}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected(as_view=True))
        mapper = QiskitSizeMapper(self.arch_graph, swapper.map)

        out = mapper.map(self.circuit, current_mapping)
        swaps = list(swapper.map({current_mapping[k]: v for k, v in out.items()}))
        self.assertEqual(10, len(swaps))
