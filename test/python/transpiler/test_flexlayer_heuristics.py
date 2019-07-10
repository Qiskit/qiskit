# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for FlexlayerHeuristics."""

import unittest

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import CouplingMap, Layout
from qiskit.transpiler.passes.mapping.algorithm.dependency_graph import DependencyGraph
from qiskit.transpiler.passes.mapping.algorithm.flexlayer_heuristics import FlexlayerHeuristics


class TestFlexlayerHeuristics(unittest.TestCase):
    """Tests for FlexlayerHeuristics."""

    def test_search_4qcx4h1(self):
        """Test for 4 cx gates and 1 h gate in a 4q circuit.
        """
        qr = QuantumRegister(4, 'q')
        cr = ClassicalRegister(4, 'c')
        circuit = QuantumCircuit(qr, cr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.barrier()
        circuit.measure(qr, cr)

        dep_graph = DependencyGraph(circuit, graph_type="basic")
        coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])  # {0: [1], 1: [2, 3]}
        initial_layout = Layout.generate_trivial_layout(qr)
        algo = FlexlayerHeuristics(circuit, dep_graph, coupling, initial_layout)
        actual_dag, actual_last_layout = algo.search()

        expected = QuantumCircuit(qr, cr)
        expected.cx(qr[0], qr[1])
        expected.swap(qr[1], qr[2])
        expected.cx(qr[1], qr[3])
        expected.cx(qr[2], qr[1])
        expected.h(qr[2])
        expected.swap(qr[0], qr[1])
        expected.cx(qr[2], qr[1])
        expected.barrier()
        expected.measure(qr[1], cr[0])
        expected.measure(qr[2], cr[1])
        expected.measure(qr[0], cr[2])
        expected.measure(qr[3], cr[3])
        expected_last_layout = Layout.from_qubit_list([qr[2], qr[0], qr[1], qr[3]])
        self.assertEqual(actual_dag, circuit_to_dag(expected))
        self.assertEqual(str(actual_last_layout), str(expected_last_layout))

    def test_search_multi_creg(self):
        """Test for multiple ClassicalRegisters.
        """
        qr = QuantumRegister(4, 'q')
        cr1 = ClassicalRegister(2, 'c')
        cr2 = ClassicalRegister(2, 'd')
        circuit = QuantumCircuit(qr, cr1, cr2)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.h(qr[1])
        circuit.cx(qr[1], qr[0])
        circuit.barrier(qr)
        circuit.measure(qr[0], cr1[0])
        circuit.measure(qr[1], cr1[1])
        circuit.measure(qr[2], cr2[0])
        circuit.measure(qr[3], cr2[1])

        dep_graph = DependencyGraph(circuit, graph_type="basic")
        coupling = CouplingMap([[0, 1], [1, 2], [1, 3]])  # {0: [1], 1: [2, 3]}
        initial_layout = Layout.generate_trivial_layout(qr)
        algo = FlexlayerHeuristics(circuit, dep_graph, coupling, initial_layout)
        actual_dag, actual_last_layout = algo.search()

        expected = QuantumCircuit(qr, cr1, cr2)
        expected.cx(qr[0], qr[1])
        expected.swap(qr[1], qr[2])
        expected.cx(qr[1], qr[3])
        expected.cx(qr[2], qr[1])
        expected.h(qr[2])
        expected.swap(qr[0], qr[1])
        expected.cx(qr[2], qr[1])
        expected.barrier()
        expected.measure(qr[0], cr2[0])
        expected.measure(qr[1], cr1[0])
        expected.measure(qr[2], cr1[1])
        expected.measure(qr[3], cr2[1])
        expected_last_layout = Layout.from_qubit_list([qr[2], qr[0], qr[1], qr[3]])
        self.assertEqual(actual_dag, circuit_to_dag(expected))
        self.assertEqual(str(actual_last_layout), str(expected_last_layout))


if __name__ == "__main__":
    unittest.main()
