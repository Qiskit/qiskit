"""Test cases for the scoring package"""
from unittest import TestCase

import networkx as nx
from qiskit import QuantumRegister
from qiskit.dagcircuit import DAGCircuit

from qiskit.transpiler.passes.extension_mapper.src import scoring


class TestScoring(TestCase):
    """The test cases."""
    def setUp(self):
        self.circuit = DAGCircuit()
        self.circuit.add_basis_element('cx', 2)
        self.circuit.add_basis_element('u2', 1, number_parameters=2)
        self.gate_costs = {
            'id': 0, 'u1': 0, 'measure': 0, 'reset': 0, 'barrier': 0,
            'u2': 1, 'u3': 1, 'U': 1,
            'cx': 10, 'CX': 10
            }

        self.arch_graph = nx.Graph()

    def test_cost_small(self):
        """Test computing the score for a small circuit.

            Circuit consists of only a CNOT on an architecture with a single weighted(=2) edge
        """
        self.circuit.add_qreg(QuantumRegister(2, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        permutation = {('q', 0): 0, ('q', 1): 1}
        self.arch_graph.add_edge(0, 1, weight=2)

        out = scoring.cost(self.circuit, permutation, self.arch_graph, gate_costs=self.gate_costs)
        self.assertEqual(scoring.Cost(
            depth=1,
            synchronous_cost=20,
            asynchronous_cost=20,
            cumulative_cost=20
            ), out)

    def test_cost_small_2(self):
        """Test computing the score on a small circuit of a CNOT and Hadamard in parallel."""
        self.circuit.add_qreg(QuantumRegister(3, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('u2', [('q', 2)], params=['0', 'pi'])  # Hadamard
        placement = {('q', 0): 0, ('q', 1): 1, ('q', 2): 2}
        self.arch_graph.add_edge(0, 1, weight=2)

        out = scoring.cost(self.circuit, placement, self.arch_graph)
        self.assertEqual(scoring.Cost(
            depth=1,
            synchronous_cost=20,
            asynchronous_cost=20,
            cumulative_cost=21
            ), out)

    def test_async_cost_small(self):
        """Test the asynchronous cost versus synchronous cost of a circuit."""
        self.circuit.add_qreg(QuantumRegister(4, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        self.circuit.apply_operation_back('u2', [('q', 2)], params=['0', 'pi'])  # Hadamard
        self.circuit.apply_operation_back('cx', [('q', 2), ('q', 3)])
        placement = {('q', 0): 0, ('q', 1): 1, ('q', 2): 2, ('q', 3): 3}
        self.arch_graph.add_edge(0, 1)
        self.arch_graph.add_edge(2, 3)

        out = scoring.cost(self.circuit, placement, self.arch_graph)
        self.assertEqual(scoring.Cost(
            depth=2,
            synchronous_cost=20,
            asynchronous_cost=11,
            cumulative_cost=21
            ), out)

    def test_cost_noedge(self):
        """Test the case where an edge is used that does not exist in the architecture graph."""
        self.circuit.add_qreg(QuantumRegister(2, name="q"))
        self.circuit.apply_operation_back('cx', [('q', 0), ('q', 1)])
        permutation = {('q', 0): 0, ('q', 1): 1}
        self.arch_graph.add_nodes_from([0, 1])  # no edge!
        self.assertRaises(KeyError, scoring.cost, self.circuit, permutation, self.arch_graph,
                          gate_costs=self.gate_costs)
