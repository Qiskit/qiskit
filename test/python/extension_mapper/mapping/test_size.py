"""Test cases for the mapping.size package"""
import unittest
from unittest import TestCase

import random
import numpy as np
import networkx as nx

from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit

from qiskit.transpiler.passes.extension_mapper.src.mapping.size import SizeMapper
from qiskit.transpiler.passes.extension_mapper.src.permutation.general \
    import ApproximateTokenSwapper


def dummy_permuter(mapping):
    """A permuter that always returns the empty list"""
    # pylint: disable=unused-argument
    return []


class TestSizeMapper(TestCase):
    """The test cases."""

    def setUp(self):
        random.seed(0)
        np.random.seed(0)

        self.circuit = DAGCircuit()
        self.circuit.add_basis_element('cx', 2)
        self.circuit.add_basis_element('u2', 1, number_parameters=2)

        self.arch_graph = nx.DiGraph()

    def test_map_simple_empty(self):
        """Test the mapping of an empty circuit on 2 interacting qubits."""
        self.circuit.add_qreg(QuantumRegister(2, name="q"))
        self.arch_graph.add_edge(0, 1)
        current_mapping = {("q", i): i for i in range(2)}
        mapper = SizeMapper(self.arch_graph, dummy_permuter)

        out = mapper.simple(self.circuit, current_mapping)
        # No qubits should be mapped.
        self.assertEqual({}, out)

    def test_map_simple_small(self):
        """Test the mapping of a single CNOT onto a 2-qubit path"""
        self.circuit.add_qreg(QuantumRegister(2, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.arch_graph.add_edge(0, 1, weight=2)
        current_mapping = {("q", i): i for i in range(2)}

        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected())
        mapper = SizeMapper(self.arch_graph, swapper.map)

        out = mapper.simple(self.circuit, current_mapping)
        self.assertEqual({('q', 0): 0, ('q', 1): 1}, out)

    def test_map_simple_small_2(self):
        """Test the mapping of a parallel Hadamard and CNOT on 0, 1<->2 (weight=2)"""
        self.circuit.add_qreg(QuantumRegister(3, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('u2', [('q', 2)], params=['0', 'pi'])  # Hadamard
        # G: 0, 1<->2
        self.arch_graph.add_node(0)
        self.arch_graph.add_edge(1, 2, weight=2)
        current_mapping = {("q", i): i for i in range(3)}
        mapper = SizeMapper(self.arch_graph, dummy_permuter)

        out = mapper.simple(self.circuit, current_mapping)
        # The output is empty because there is a 1-qubit gate.
        self.assertEqual({}, out)
        # The cnots must be reshuffled to qubits 1 and 2, but we dont care which order.
        # self.assertEqual({('q', 0), ('q', 1)}, set(out.keys()))
        # self.assertEqual({1, 2}, set(out.values()))

    def test_map_simple_two_layers(self):
        """Test mapping two sequential CNOTs"""
        self.circuit.add_qreg(QuantumRegister(3, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('cx', [('q', 1), ('q', 2)])
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(1, 2)
        current_mapping = {("q", i): i for i in range(3)}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected())
        mapper = SizeMapper(self.arch_graph, swapper.map)

        out = mapper.simple(self.circuit, current_mapping)
        self.assertEqual({('q', 0): 0, ('q', 1): 1}, out)

    def test_map_simple_choice(self):
        """Given two gates map the cheapest one."""
        self.circuit.add_qreg(QuantumRegister(4, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('cx', [('q', 2), ('q', 3)])
        self.arch_graph = nx.path_graph(8)
        current_mapping = {("q", 0): 1, ("q", 1): 6, ("q", 2): 0, ("q", 3): 7}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected())
        mapper = SizeMapper(self.arch_graph, swapper.map)

        out = mapper.simple(self.circuit, current_mapping)
        # Only q0 and q1 should be mapped since they are closer.
        self.assertEqual({('q', 0), ('q', 1)}, set(out.keys()))

    def test_map_greedy_empty(self):
        """Test mapping an empty circuit."""
        self.circuit.add_qreg(QuantumRegister(2, name="q"))
        self.arch_graph.add_edge(0, 1)
        current_mapping = {('q', i): 1 - i for i in [0, 1]}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected())
        mapper = SizeMapper(self.arch_graph, swapper.map)

        out = mapper.greedy(self.circuit, current_mapping)
        self.assertEqual({}, out)

    def test_map_greedy_small(self):
        """Test mapping a single CNOT to the architecture (weight=2)"""
        self.circuit.add_qreg(QuantumRegister(2, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.arch_graph.add_edge(0, 1, weight=2)
        current_mapping0 = {('q', i): i for i in [0, 1]}
        current_mapping1 = {('q', i): 1 - i for i in [0, 1]}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected())
        mapper = SizeMapper(self.arch_graph, swapper.map)

        # Qubits should not be moved needlessly.
        out0 = mapper.greedy(self.circuit, current_mapping0)
        self.assertEqual({}, out0)
        out1 = mapper.greedy(self.circuit, current_mapping1)
        self.assertEqual({}, out1)

    def test_map_greedy_small_2(self):
        """Test mapping a parallel Hadamard and CNOT to 0, 1<->2 (weight=2)"""
        self.circuit.add_qreg(QuantumRegister(3, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('u2', [('q', 2)], params=['0', 'pi'])  # Hadamard
        # G: 0, 1<->2
        self.arch_graph.add_node(0)
        self.arch_graph.add_edge(1, 2, weight=2)
        current_mapping = {('q', i): i for i in range(3)}
        mapper = SizeMapper(self.arch_graph, dummy_permuter)

        out = mapper.greedy(self.circuit, current_mapping)
        # The cnots must be reshuffled to qubits 1 and 2, but we dont care which order.
        self.assertEqual({('q', 0), ('q', 1)}, set(out.keys()))
        self.assertEqual({1, 2}, set(out.values()))

    def test_map_greedy_cnots_modular(self):
        """Test mapping two sequential CNOTs on modular graph."""
        self.circuit.add_qreg(QuantumRegister(3, name='q'))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('cx', [('q', 1), ('q', 2)])
        # Modular graph, m = n = 2
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(0, 2)
        self.arch_graph.add_edge(2, 3)

        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected())
        current_mapping = {('q', 0): 1, ('q', 1): 0, ('q', 2): 2}
        mapper = SizeMapper(self.arch_graph, swapper.map)

        out = mapper.greedy(self.circuit, current_mapping)
        # The qubits should not be moved.
        self.assertEqual({}, out)

    @unittest.skip('Changes to nx.max_weight_matching')
    def test_map_greedy_two_layers_move(self):
        """Test mapping two sequential CNOTs that require a shuffling of qubits."""
        self.circuit.add_qreg(QuantumRegister(3, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('cx', [('q', 1), ('q', 2)])
        # Modular graph, m = n = 2
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(0, 2)
        self.arch_graph.add_edge(2, 3)
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected(as_view=True))

        apx_permuter = swapper.map
        current_mapping = {('q', 0): 1, ('q', 1): 2, ('q', 2): 3}
        mapper = SizeMapper(self.arch_graph, apx_permuter)

        out = mapper.greedy(self.circuit, current_mapping)
        # Because of matching limitations q1 should be moved to 0.
        output_mapping = {('q', 0): 1, ('q', 1): 0}
        self.assertEqual(output_mapping, out)

    def test_map_extend_empty(self):
        """Test the mapping of an empty circuit on 2 interacting qubits."""
        self.circuit.add_qreg(QuantumRegister(2, name="q"))
        self.arch_graph.add_edge(0, 1)
        current_mapping = {("q", i): i for i in range(2)}
        mapper = SizeMapper(self.arch_graph, dummy_permuter)

        out = mapper.simple_extend(self.circuit, current_mapping)
        # No qubits should be mapped.
        self.assertEqual({}, out)

    def test_map_extend_small(self):
        """Test the mapping of a single CNOT onto a 2-qubit path"""
        self.circuit.add_qreg(QuantumRegister(2, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.arch_graph.add_edge(0, 1, weight=2)
        current_mapping = {("q", i): i for i in range(2)}

        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected())
        mapper = SizeMapper(self.arch_graph, swapper.map)

        out = mapper.simple_extend(self.circuit, current_mapping)
        self.assertEqual({}, out)

    def test_map_extend_small_2(self):
        """Test the mapping of a parallel Hadamard and CNOT on 0, 1<->2 (weight=2)"""
        self.circuit.add_qreg(QuantumRegister(3, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('u2', [('q', 2)], params=['0', 'pi'])  # Hadamard
        # G: 0, 1<->2
        self.arch_graph.add_node(0)
        self.arch_graph.add_edge(1, 2, weight=2)
        current_mapping = {("q", i): i for i in range(3)}
        mapper = SizeMapper(self.arch_graph, dummy_permuter)

        out = mapper.simple_extend(self.circuit, current_mapping)
        # The output is empty because there is a 1-qubit gate.
        self.assertEqual({}, out)
        # The cnots must be reshuffled to qubits 1 and 2, but we dont care which order.
        # self.assertEqual({('q', 0), ('q', 1)}, set(out.keys()))
        # self.assertEqual({1, 2}, set(out.values()))

    def test_map_extend_two_layers(self):
        """Test mapping two sequential CNOTs"""
        self.circuit.add_qreg(QuantumRegister(3, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('cx', [('q', 1), ('q', 2)])
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(1, 2)
        current_mapping = {("q", i): i for i in range(3)}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected())
        mapper = SizeMapper(self.arch_graph, swapper.map)

        out = mapper.simple_extend(self.circuit, current_mapping)
        self.assertEqual({}, out)

    def test_map_extend_choice(self):
        """Given two gates map both."""
        self.circuit.add_qreg(QuantumRegister(4, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('cx', [('q', 2), ('q', 3)])
        self.circuit.add_basis_element("swap", 2)
        self.arch_graph = nx.path_graph(8).to_directed()
        current_mapping = {("q", 0): 1, ("q", 1): 6, ("q", 2): 0, ("q", 3): 7}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected(as_view=True))
        mapper = SizeMapper(self.arch_graph, swapper.map, allow_swaps=True)

        out = mapper.simple_extend(self.circuit, current_mapping)
        # All qubits are mapped, since it takes 10 SWAPs to map both at once
        # and 4 + 6 SWAPS to map one after the other.
        self.assertEqual(set(current_mapping.keys()), set(out.keys()))
        swaps = list(swapper.map({current_mapping[k]: v for k, v in out.items()}))
        self.assertEqual(10, len(swaps))

    def test_map_extend_free(self):
        """Test that the extension mapper will take advantage of a "free" SWAP gate."""
        self.circuit.add_qreg(QuantumRegister(4, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('cx', [('q', 2), ('q', 3)])
        self.circuit.add_basis_element("swap", 2)
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(0, 2)
        self.arch_graph.add_edge(0, 3)
        self.arch_graph.add_edge(3, 4)
        self.arch_graph.add_edge(3, 5)
        current_mapping = {('q', 0): 1, ('q', 2): 2, ('q', 1): 4, ('q', 3): 5}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected(as_view=True))
        mapper = SizeMapper(self.arch_graph, swapper.map, allow_swaps=True)

        out = mapper.simple_extend(self.circuit, current_mapping, lookahead=True)
        # The mapper takes advantage of a "free" SWAP along 0-3 and maps all qubits in one step
        self.assertEqual(set(current_mapping.keys()), set(out.keys()))

    @unittest.skip('Unpredictable behavior.')
    def test_map_extend_lookahead(self):
        """Test that larger subsets of gates can be mapped simultaneously."""
        self.circuit.add_qreg(QuantumRegister(6, name="q"))
        for i in range(0, 6, 2):
            self.circuit.apply_operation_back('cx', [('q', i), ('q', i + 1)])
        self.circuit.add_basis_element("swap", 2)
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(0, 2)
        self.arch_graph.add_edge(2, 6)
        self.arch_graph.add_edge(0, 3)
        self.arch_graph.add_edge(3, 4)
        self.arch_graph.add_edge(3, 5)
        self.arch_graph.add_edge(5, 7)
        current_mapping = {('q', 0): 1, ('q', 1): 4,
                           ('q', 2): 2, ('q', 3): 5,
                           ('q', 4): 6, ('q', 5): 7}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected(as_view=True))
        mapper = SizeMapper(self.arch_graph, swapper.map, allow_swaps=True)

        out = mapper.simple_extend(self.circuit, current_mapping, lookahead=True)
        print(out)
        self.assertEqual(6, len(out))
        swaps = list(swapper.map({current_mapping[k]: v for k, v in out.items()}))
        self.assertEqual(7, len(swaps))

    def test_qiskit_map_empty(self):
        """Test the mapping of an empty circuit on 2 interacting qubits."""
        self.circuit.add_qreg(QuantumRegister(2, name="q"))
        self.arch_graph.add_edge(0, 1)
        current_mapping = {("q", i): i for i in range(2)}
        mapper = SizeMapper(self.arch_graph, dummy_permuter)

        out = mapper.qiskit_mapper(self.circuit, current_mapping)
        # No qubits should be mapped.
        self.assertEqual({}, out)

    def test_qiskit_map_small(self):
        """Test the mapping of a single CNOT onto a 2-qubit path"""
        self.circuit.add_qreg(QuantumRegister(2, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.arch_graph.add_edge(0, 1, weight=2)
        current_mapping = {("q", i): i for i in range(2)}

        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected())
        mapper = SizeMapper(self.arch_graph, swapper.map)

        out = mapper.simple_extend(self.circuit, current_mapping)
        self.assertEqual({}, out)

    def test_qiskit_map_small_2(self):
        """Test the mapping of a parallel Hadamard and CNOT on 0, 1<->2 (weight=2)"""
        self.circuit.add_qreg(QuantumRegister(3, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('u2', [('q', 2)], params=['0', 'pi'])  # Hadamard
        # G: 0, 1<->2
        self.arch_graph.add_node(0)
        self.arch_graph.add_edge(1, 2, weight=2)
        current_mapping = {("q", i): i for i in range(3)}
        mapper = SizeMapper(self.arch_graph, dummy_permuter)

        out = mapper.qiskit_mapper(self.circuit, current_mapping)
        # The output is empty because there is a 1-qubit gate.
        self.assertEqual({}, out)
        # The cnots must be reshuffled to qubits 1 and 2, but we dont care which order.
        # self.assertEqual({('q', 0), ('q', 1)}, set(out.keys()))
        # self.assertEqual({1, 2}, set(out.values()))

    def test_qiskit_map_two_layers(self):
        """Test mapping two sequential CNOTs"""
        self.circuit.add_qreg(QuantumRegister(3, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('cx', [('q', 1), ('q', 2)])
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(1, 2)
        current_mapping = {("q", i): i for i in range(3)}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected())
        mapper = SizeMapper(self.arch_graph, swapper.map)

        out = mapper.qiskit_mapper(self.circuit, current_mapping)
        self.assertEqual({}, out)

    def test_qiskit_map_choice(self):
        """Given two gates map one."""
        self.circuit.add_qreg(QuantumRegister(4, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('cx', [('q', 2), ('q', 3)])
        self.circuit.add_basis_element("swap", 2)
        self.arch_graph = nx.path_graph(8).to_directed(as_view=True)
        current_mapping = {("q", 0): 1, ("q", 1): 6, ("q", 2): 0, ("q", 3): 7}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected(as_view=True))
        mapper = SizeMapper(self.arch_graph, swapper.map, allow_swaps=True)

        out = mapper.qiskit_mapper(self.circuit, current_mapping, seed=0)
        # The second gate is mapped after the first.
        self.assertEqual({('q', 0), ('q', 1)}, set(out.keys()))
        swaps = list(swapper.map({current_mapping[k]: v for k, v in out.items()}))
        self.assertEqual(4, len(swaps))

    def test_qiskit_map_free(self):
        """Test that the extension mapper will take advantage of a "free" SWAP gate."""
        self.circuit.add_qreg(QuantumRegister(4, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('cx', [('q', 2), ('q', 3)])
        self.circuit.add_basis_element("swap", 2)
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(0, 2)
        self.arch_graph.add_edge(0, 3)
        self.arch_graph.add_edge(3, 4)
        self.arch_graph.add_edge(3, 5)
        current_mapping = {('q', 0): 1, ('q', 2): 2, ('q', 1): 4, ('q', 3): 5}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected(as_view=True))
        mapper = SizeMapper(self.arch_graph, swapper.map, allow_swaps=True)

        out = mapper.qiskit_mapper(self.circuit, current_mapping)
        # The mapper takes advantage of a "free" SWAP along 0-3 and maps all qubits in one step
        self.assertEqual(set(current_mapping.keys()), set(out.keys()))

    def test_qiskit_map_subsets(self):
        """Test that larger subsets of gates can be mapped simultaneously."""
        self.circuit.add_qreg(QuantumRegister(6, name="q"))
        for i in range(0, 6, 2):
            self.circuit.apply_operation_back('cx', [('q', i), ('q', i + 1)])
        self.circuit.add_basis_element("swap", 2)
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(0, 2)
        self.arch_graph.add_edge(2, 6)
        self.arch_graph.add_edge(0, 3)
        self.arch_graph.add_edge(3, 4)
        self.arch_graph.add_edge(3, 5)
        self.arch_graph.add_edge(5, 7)
        current_mapping = {('q', 0): 1, ('q', 1): 4,
                           ('q', 2): 2, ('q', 3): 5,
                           ('q', 4): 6, ('q', 5): 7}
        swapper = ApproximateTokenSwapper(self.arch_graph.to_undirected(as_view=True))
        mapper = SizeMapper(self.arch_graph, swapper.map, allow_swaps=True)

        out = mapper.qiskit_mapper(self.circuit, current_mapping)
        swaps = list(swapper.map({current_mapping[k]: v for k, v in out.items()}))
        self.assertEqual(10, len(swaps))
