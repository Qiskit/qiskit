# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the LookaheadSwap pass"""

import unittest
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.mapper import CouplingMap
from qiskit.converters import circuit_to_dag
from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PassManager, transpile_dag
from ..common import QiskitTestCase


class TestLookaheadSwap(QiskitTestCase):
    """Tests the LookaheadSwap pass."""

    def test_lookahead_swap_doesnt_modify_mapped_circuit(self):
        """Test that lookahead mapper is idempotent.

        It should not modify a circuit which is already compatible with the
        coupling map, and can be applied repeatedly without modifying the circuit.
        """

        qr = QuantumRegister(3, name='q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[1])
        original_dag = circuit_to_dag(circuit)

        # Create coupling map which contains all two-qubit gates in the circuit.
        coupling_map = CouplingMap(couplinglist=[(0, 1), (0, 2)])

        pass_manager = PassManager()
        pass_manager.append(LookaheadSwap(coupling_map))
        mapped_dag = transpile_dag(original_dag, pass_manager=pass_manager)

        self.assertEqual(original_dag, mapped_dag)

        second_pass_manager = PassManager()
        second_pass_manager.append(LookaheadSwap(coupling_map))
        remapped_dag = transpile_dag(mapped_dag, pass_manager=second_pass_manager)

        self.assertEqual(mapped_dag, remapped_dag)

    def test_lookahead_swap_should_add_a_single_swap(self):
        """Test that LookaheadSwap will insert a SWAP to match layout.

        For a single cx gate which is not available in the current layout, test
        that the mapper inserts a single swap to enable the gate.
        """

        qr = QuantumRegister(3)
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        dag_circuit = circuit_to_dag(circuit)

        coupling_map = CouplingMap(couplinglist=[(0, 1), (1, 2)])

        pass_manager = PassManager()
        pass_manager.append([LookaheadSwap(coupling_map)])
        mapped_dag = transpile_dag(dag_circuit, pass_manager=pass_manager)

        self.assertEqual(mapped_dag.count_ops().get('swap', 0),
                         dag_circuit.count_ops().get('swap', 0) + 1)

    def test_lookahead_swap_finds_minimal_swap_solution(self):
        """Of many valid SWAPs, test that LookaheadSwap finds the cheapest path.

        For a two CNOT circuit: cx q[0],q[2]; cx q[0],q[1]
        on the initial layout: qN -> qN
        (At least) two solutions exist:
        - SWAP q[0],[1], cx q[0],q[2], cx q[0],q[1]
        - SWAP q[1],[2], cx q[0],q[2], SWAP q[1],q[2], cx q[0],q[1]

        Verify that we find the first solution, as it requires fewer SWAPs.
        """

        qr = QuantumRegister(3)
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[1])

        dag_circuit = circuit_to_dag(circuit)

        coupling_map = CouplingMap(couplinglist=[(0, 1), (1, 2)])

        pass_manager = PassManager()
        pass_manager.append([LookaheadSwap(coupling_map)])
        mapped_dag = transpile_dag(dag_circuit, pass_manager=pass_manager)

        self.assertEqual(mapped_dag.count_ops().get('swap', 0),
                         dag_circuit.count_ops().get('swap', 0) + 1)


if __name__ == '__main__':
    unittest.main()
