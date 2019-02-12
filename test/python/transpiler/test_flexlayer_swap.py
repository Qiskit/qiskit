# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the FlexlayerSwap pass"""

import unittest
# from qiskit.transpiler.passes import FlexlayerSwap
from new_swappers import FlexlayerSwap
from qiskit.transpiler import PassManager, transpile_dag
from qiskit.mapper import CouplingMap, Layout
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


class TestFlexlayerSwap(QiskitTestCase):
    """ Tests the FlexlayerSwap pass."""

    def test_trivial_case(self):
        """No need to have any swap, the CX are distance 1 to each other
         q0:--(+)-[U]-(+)-
               |       |
         q1:---.-------|--
                       |
         q2:-----------.--

         CouplingMap map: [1]--[0]--[2]
        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[2])

        dag = circuit_to_dag(circuit)
        pass_ = FlexlayerSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_trivial_in_same_layer(self):
        """ No need to have any swap, two CXs distance 1 to each other, in the same layer
         q0:--(+)--
               |
         q1:---.---

         q2:--(+)--
               |
         q3:---.---

         CouplingMap map: [0]--[1]--[2]--[3]
        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[0], qr[1])

        dag = circuit_to_dag(circuit)
        pass_ = FlexlayerSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(dag, after)

    def test_a_single_swap(self):
        """ Adding a swap
         q0:-------

         q1:--(+)--
               |
         q2:---.---

         CouplingMap map: [1]--[0]--[2]

         q0:--X---.---
              |   |
         q1:--X---|---
                  |
         q2:-----(+)--

        """
        coupling = CouplingMap([[0, 1], [0, 2]])

        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[2])
        expected.cx(qr[1], qr[0])

        pass_ = FlexlayerSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_keep_layout(self):
        """After a swap, the following gates also change the wires.
         qr0:---.---[H]--
                |
         qr1:---|--------
                |
         qr2:--(+)-------

         CouplingMap map: [0]--[1]--[2]

         qr0:--X-----------
               |
         qr1:--X---.--[H]--
                   |
         qr2:-----(+)------
        """
        coupling = CouplingMap([[1, 0], [1, 2]])

        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[2])
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[1], qr[0])
        expected.cx(qr[1], qr[2])
        expected.h(qr[1])

        pass_ = FlexlayerSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_far_swap(self):
        """ A far swap that affects coming CXs.
         qr0:--(+)---.--
                |    |
         qr1:---|----|--
                |    |
         qr2:---|----|--
                |    |
         qr3:---.---(+)-

         CouplingMap map: [0]--[1]--[2]--[3]

         qr0:--X--------------
               |
         qr1:--X--X-----------
                  |
         qr2:-----X--(+)---.--
                      |    |
         qr3:---------.---(+)-

        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.swap(qr[0], qr[1])
        expected.swap(qr[1], qr[2])
        expected.cx(qr[2], qr[3])
        expected.cx(qr[3], qr[2])

        pass_ = FlexlayerSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_far_swap_with_gate_the_front(self):
        """ A far swap with a gate in the front.
         qr0:------(+)--
                    |
         qr1:-------|---
                    |
         qr2:-------|---
                    |
         qr3:--[H]--.---

         CouplingMap map: [0]--[1]--[2]--[3]

         q0:------X----------
                  |
         q1:------X--X-------
                     |
         q2:---------X--(+)--
                         |
         q3:-[H]---------.---

        """
        coupling = CouplingMap([[0, 1], [1, 2], [2, 3]])

        qr = QuantumRegister(4, 'qr')  # virtual qubit
        circuit = QuantumCircuit(qr)
        circuit.h(qr[3])
        circuit.cx(qr[3], qr[0])
        dag = circuit_to_dag(circuit)

        expected = QuantumCircuit(qr)
        expected.h(qr[3])
        expected.swap(qr[0], qr[1])
        expected.swap(qr[1], qr[2])
        expected.cx(qr[3], qr[2])

        pass_ = FlexlayerSwap(coupling)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_initial_layout(self):
        """ Using an initial_layout
         0:q1:--(+)--
                 |
         1:q0:---|---
                 |
         2:q2:---.---

         CouplingMap map: [0]--[1]--[2]

         0:q1:--X-------
                |
         1:q0:--X---.---
                    |
         2:q2:-----(+)--

        """
        coupling = CouplingMap([[0, 1], [1, 2]])

        qr = QuantumRegister(3, 'q')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[1], qr[2])
        dag = circuit_to_dag(circuit)
        layout = Layout([(qr, 1), (qr, 0), (qr, 2)])

        expected = QuantumCircuit(qr)
        expected.swap(qr[1], qr[0])
        expected.cx(qr[0], qr[2])

        pass_ = FlexlayerSwap(coupling, initial_layout=layout)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)

    def test_initial_layout_in_different_qregs(self):
        """ Using an initial_layout, and with several qregs
         0:q1_0:--(+)--
                   |
         1:q0_0:---|---
                   |
         2:q2_0:---.---

         CouplingMap map: [0]--[1]--[2]

         0:q1_0:--X-------
                  |
         1:q0_0:--X---.---
                      |
         2:q2_0:-----(+)--
        """
        coupling = CouplingMap([[0, 1], [1, 2]])

        qr0 = QuantumRegister(1, 'q0')
        qr1 = QuantumRegister(1, 'q1')
        qr2 = QuantumRegister(1, 'q2')
        circuit = QuantumCircuit(qr0, qr1, qr2)
        circuit.cx(qr1[0], qr2[0])
        dag = circuit_to_dag(circuit)
        layout = Layout([(qr1, 0), (qr0, 0), (qr2, 0)])

        expected = QuantumCircuit(qr0, qr1, qr2)
        expected.swap(qr1[0], qr0[0])
        expected.cx(qr0[0], qr2[0])

        pass_ = FlexlayerSwap(coupling, initial_layout=layout)
        after = pass_.run(dag)

        self.assertEqual(circuit_to_dag(expected), after)


    # def test_flexlayer_swap_doesnt_modify_mapped_circuit(self):
    #     """Test that lookahead mapper is idempotent.
    #
    #     It should not modify a circuit which is already compatible with the
    #     coupling map, and can be applied repeatedly without modifying the circuit.
    #     """
    #
    #     qr = QuantumRegister(3, name='q')
    #     circuit = QuantumCircuit(qr)
    #     circuit.cx(qr[0], qr[2])
    #     circuit.cx(qr[0], qr[1])
    #     original_dag = circuit_to_dag(circuit)
    #
    #     # Create coupling map which contains all two-qubit gates in the circuit.
    #     coupling_map = CouplingMap([[0, 1], [0, 2]])
    #
    #     pass_manager = PassManager()
    #     pass_manager.append(FlexlayerSwap(coupling_map))
    #     mapped_dag = transpile_dag(original_dag, pass_manager=pass_manager)
    #
    #     self.assertEqual(original_dag, mapped_dag)
    #
    #     second_pass_manager = PassManager()
    #     second_pass_manager.append(FlexlayerSwap(coupling_map))
    #     remapped_dag = transpile_dag(mapped_dag, pass_manager=second_pass_manager)
    #
    #     self.assertEqual(mapped_dag, remapped_dag)
    #
    # def test_flexlayer_swap_should_add_a_single_swap(self):
    #     """Test that LookaheadSwap will insert a SWAP to match layout.
    #
    #     For a single cx gate which is not available in the current layout, test
    #     that the mapper inserts a single swap to enable the gate.
    #     """
    #
    #     qr = QuantumRegister(3)
    #     circuit = QuantumCircuit(qr)
    #     circuit.cx(qr[0], qr[2])
    #     dag_circuit = circuit_to_dag(circuit)
    #
    #     coupling_map = CouplingMap([[0, 1], [1, 2]])
    #
    #     pass_manager = PassManager()
    #     pass_manager.append([FlexlayerSwap(coupling_map)])
    #     mapped_dag = transpile_dag(dag_circuit, pass_manager=pass_manager)
    #
    #     self.assertEqual(mapped_dag.count_ops().get('swap', 0),
    #                      dag_circuit.count_ops().get('swap', 0) + 1)
    #
    # def test_flexlayer_swap_finds_minimal_swap_solution(self):
    #     """Of many valid SWAPs, test that LookaheadSwap finds the cheapest path.
    #
    #     For a two CNOT circuit: cx q[0],q[2]; cx q[0],q[1]
    #     on the initial layout: qN -> qN
    #     (At least) two solutions exist:
    #     - SWAP q[0],[1], cx q[0],q[2], cx q[0],q[1]
    #     - SWAP q[1],[2], cx q[0],q[2], SWAP q[1],q[2], cx q[0],q[1]
    #
    #     Verify that we find the first solution, as it requires fewer SWAPs.
    #     """
    #
    #     qr = QuantumRegister(3)
    #     circuit = QuantumCircuit(qr)
    #     circuit.cx(qr[0], qr[2])
    #     circuit.cx(qr[0], qr[1])
    #
    #     dag_circuit = circuit_to_dag(circuit)
    #
    #     coupling_map = CouplingMap([[0, 1], [1, 2]])
    #
    #     pass_manager = PassManager()
    #     pass_manager.append([FlexlayerSwap(coupling_map)])
    #     mapped_dag = transpile_dag(dag_circuit, pass_manager=pass_manager)
    #
    #     self.assertEqual(mapped_dag.count_ops().get('swap', 0),
    #                      dag_circuit.count_ops().get('swap', 0) + 1)
    #
    # def test_flexlayer_swap_maps_measurements(self):
    #     """Verify measurement nodes are updated to map correct cregs to re-mapped qregs.
    #
    #     Create a circuit with measures on q0 and q2, following a swap between q0 and q2.
    #     Since that swap is not in the coupling, one of the two will be required to move.
    #     Verify that the mapped measure corresponds to one of the two possible layouts following
    #     the swap.
    #
    #     """
    #
    #     qr = QuantumRegister(3)
    #     cr = ClassicalRegister(2)
    #     circuit = QuantumCircuit(qr, cr)
    #
    #     circuit.cx(qr[0], qr[2])
    #     circuit.measure(qr[0], cr[0])
    #     circuit.measure(qr[2], cr[1])
    #
    #     dag_circuit = circuit_to_dag(circuit)
    #
    #     coupling_map = CouplingMap([[0, 1], [1, 2]])
    #
    #     pass_manager = PassManager()
    #     pass_manager.append([FlexlayerSwap(coupling_map)])
    #     mapped_dag = transpile_dag(dag_circuit, pass_manager=pass_manager)
    #
    #     mapped_measure_qargs = set(mapped_dag.multi_graph.nodes(data=True)[op]['qargs'][0]
    #                                for op in mapped_dag.get_named_nodes('measure'))
    #
    #     self.assertIn(mapped_measure_qargs,
    #                   [set(((QuantumRegister(3, 'q'), 0), (QuantumRegister(3, 'q'), 1))),
    #                    set(((QuantumRegister(3, 'q'), 1), (QuantumRegister(3, 'q'), 2)))])
    #
    # def test_flexlayer_swap_maps_barriers(self):
    #     """Verify barrier nodes are updated to re-mapped qregs.
    #
    #     Create a circuit with a barrier on q0 and q2, following a swap between q0 and q2.
    #     Since that swap is not in the coupling, one of the two will be required to move.
    #     Verify that the mapped barrier corresponds to one of the two possible layouts following
    #     the swap.
    #
    #     """
    #
    #     qr = QuantumRegister(3)
    #     cr = ClassicalRegister(2)
    #     circuit = QuantumCircuit(qr, cr)
    #
    #     circuit.cx(qr[0], qr[2])
    #     circuit.barrier(qr[0], qr[2])
    #
    #     dag_circuit = circuit_to_dag(circuit)
    #
    #     coupling_map = CouplingMap([[0, 1], [1, 2]])
    #
    #     pass_manager = PassManager()
    #     pass_manager.append([FlexlayerSwap(coupling_map)])
    #     mapped_dag = transpile_dag(dag_circuit, pass_manager=pass_manager)
    #
    #     mapped_barrier_qargs = [set(mapped_dag.multi_graph.nodes(data=True)[op]['qargs'])
    #                             for op in mapped_dag.get_named_nodes('barrier')][0]
    #
    #     self.assertIn(mapped_barrier_qargs,
    #                   [set(((QuantumRegister(3, 'q'), 0), (QuantumRegister(3, 'q'), 1))),
    #                    set(((QuantumRegister(3, 'q'), 1), (QuantumRegister(3, 'q'), 2)))])


if __name__ == '__main__':
    unittest.main()
