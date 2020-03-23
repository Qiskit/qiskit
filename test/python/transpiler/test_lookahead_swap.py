# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the LookaheadSwap pass"""

import unittest
from qiskit.transpiler.passes import LookaheadSwap
from qiskit.transpiler import CouplingMap
from qiskit.converters import circuit_to_dag
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


class TestLookaheadSwap(QiskitTestCase):
    """Tests the LookaheadSwap pass."""

    def test_lookahead_swap_doesnt_modify_mapped_circuit(self):
        """Test that lookahead swap is idempotent.

        It should not modify a circuit which is already compatible with the
        coupling map, and can be applied repeatedly without modifying the circuit.
        """

        qr = QuantumRegister(3, name='q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[1])
        original_dag = circuit_to_dag(circuit)

        # Create coupling map which contains all two-qubit gates in the circuit.
        coupling_map = CouplingMap([[0, 1], [0, 2]])

        mapped_dag = LookaheadSwap(coupling_map).run(original_dag)

        self.assertEqual(original_dag, mapped_dag)

        remapped_dag = LookaheadSwap(coupling_map).run(mapped_dag)

        self.assertEqual(mapped_dag, remapped_dag)

    def test_lookahead_swap_should_add_a_single_swap(self):
        """Test that LookaheadSwap will insert a SWAP to match layout.

        For a single cx gate which is not available in the current layout, test
        that the mapper inserts a single swap to enable the gate.
        """

        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        dag_circuit = circuit_to_dag(circuit)

        coupling_map = CouplingMap([[0, 1], [1, 2]])

        mapped_dag = LookaheadSwap(coupling_map).run(dag_circuit)

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

        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.cx(qr[0], qr[1])

        dag_circuit = circuit_to_dag(circuit)

        coupling_map = CouplingMap([[0, 1], [1, 2]])

        mapped_dag = LookaheadSwap(coupling_map).run(dag_circuit)

        self.assertEqual(mapped_dag.count_ops().get('swap', 0),
                         dag_circuit.count_ops().get('swap', 0) + 1)

    def test_lookahead_swap_maps_measurements(self):
        """Verify measurement nodes are updated to map correct cregs to re-mapped qregs.

        Create a circuit with measures on q0 and q2, following a swap between q0 and q2.
        Since that swap is not in the coupling, one of the two will be required to move.
        Verify that the mapped measure corresponds to one of the two possible layouts following
        the swap.

        """

        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)

        circuit.cx(qr[0], qr[2])
        circuit.measure(qr[0], cr[0])
        circuit.measure(qr[2], cr[1])

        dag_circuit = circuit_to_dag(circuit)

        coupling_map = CouplingMap([[0, 1], [1, 2]])

        mapped_dag = LookaheadSwap(coupling_map).run(dag_circuit)

        mapped_measure_qargs = set(op.qargs[0] for op in mapped_dag.named_nodes('measure'))

        self.assertIn(mapped_measure_qargs, [set((qr[0], qr[1])), set((qr[1], qr[2]))])

    def test_lookahead_swap_maps_barriers(self):
        """Verify barrier nodes are updated to re-mapped qregs.

        Create a circuit with a barrier on q0 and q2, following a swap between q0 and q2.
        Since that swap is not in the coupling, one of the two will be required to move.
        Verify that the mapped barrier corresponds to one of the two possible layouts following
        the swap.

        """

        qr = QuantumRegister(3, 'q')
        cr = ClassicalRegister(2)
        circuit = QuantumCircuit(qr, cr)

        circuit.cx(qr[0], qr[2])
        circuit.barrier(qr[0], qr[2])

        dag_circuit = circuit_to_dag(circuit)

        coupling_map = CouplingMap([[0, 1], [1, 2]])

        mapped_dag = LookaheadSwap(coupling_map).run(dag_circuit)

        mapped_barrier_qargs = [set(op.qargs) for op in mapped_dag.named_nodes('barrier')][0]

        self.assertIn(mapped_barrier_qargs, [set((qr[0], qr[1])), set((qr[1], qr[2]))])

    def test_lookahead_swap_higher_depth_width_is_better(self):
        """Test that lookahead swap finds better circuit with increasing search space.

        Increasing the tree width and depth is expected to yield a better (or same) quality
        circuit, in the form of fewer SWAPs.
        """

        qr = QuantumRegister(8, name='q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[4], qr[5])
        circuit.cx(qr[5], qr[6])
        circuit.cx(qr[6], qr[7])
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[6], qr[4])
        circuit.cx(qr[7], qr[1])
        circuit.cx(qr[4], qr[2])
        circuit.cx(qr[3], qr[7])
        circuit.cx(qr[5], qr[3])
        circuit.cx(qr[6], qr[2])
        circuit.cx(qr[2], qr[7])
        circuit.cx(qr[0], qr[6])
        circuit.cx(qr[5], qr[7])
        original_dag = circuit_to_dag(circuit)

        # Create a ring of 8 connected qubits
        coupling_map = CouplingMap.from_grid(num_rows=2, num_columns=4)

        mapped_dag_1 = LookaheadSwap(coupling_map, search_depth=3, search_width=3).run(original_dag)
        mapped_dag_2 = LookaheadSwap(coupling_map, search_depth=5, search_width=5).run(original_dag)

        num_swaps_1 = mapped_dag_1.count_ops().get('swap', 0)
        num_swaps_2 = mapped_dag_2.count_ops().get('swap', 0)

        self.assertLessEqual(num_swaps_2, num_swaps_1)


if __name__ == '__main__':
    unittest.main()
