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

"""Test for the DAGCircuit object"""

import unittest

from ddt import ddt, data

import networkx as nx

from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit import QuantumRegister
from qiskit.circuit import ClassicalRegister, Clbit
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import Measure
from qiskit.circuit import Reset
from qiskit.circuit import Gate, Instruction
from qiskit.extensions.standard.iden import IdGate
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.x import CnotGate
from qiskit.extensions.standard.z import CzGate
from qiskit.extensions.standard.x import XGate
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.barrier import Barrier
from qiskit.dagcircuit.exceptions import DAGCircuitError
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.test import QiskitTestCase


def raise_if_dagcircuit_invalid(dag):
    """Validates the internal consistency of a DAGCircuit._multi_graph.
    Intended for use in testing.

    Raises:
       DAGCircuitError: if DAGCircuit._multi_graph is inconsistent.
    """

    multi_graph = dag._multi_graph

    if not nx.is_directed_acyclic_graph(multi_graph):
        raise DAGCircuitError('multi_graph is not a DAG.')

    # Every node should be of type in, out, or op.
    # All input/output nodes should be present in input_map/output_map.
    for node in multi_graph.nodes():
        if node.type == 'in':
            assert node is dag.input_map[node.wire]
        elif node.type == 'out':
            assert node is dag.output_map[node.wire]
        elif node.type == 'op':
            continue
        else:
            raise DAGCircuitError('Found node of unexpected type: {}'.format(
                node.type))

    # Shape of node.op should match shape of node.
    for node in dag.op_nodes():
        assert len(node.qargs) == node.op.num_qubits
        assert len(node.cargs) == node.op.num_clbits

    # Every edge should be labled with a known wire.
    edges_outside_wires = [edge_data['wire']
                           for source, dest, edge_data
                           in multi_graph.edges(data=True)
                           if edge_data['wire'] not in dag.wires]
    if edges_outside_wires:
        raise DAGCircuitError('multi_graph contains one or more edges ({}) '
                              'not found in DAGCircuit.wires ({}).'.format(edges_outside_wires,
                                                                           dag.wires))

    # Every wire should have exactly one input node and one output node.
    for wire in dag.wires:
        in_node = dag.input_map[wire]
        out_node = dag.output_map[wire]

        assert in_node.wire == wire
        assert out_node.wire == wire
        assert in_node.type == 'in'
        assert out_node.type == 'out'

    # Every wire should be propagated by exactly one edge between nodes.
    for wire in dag.wires:
        cur_node = dag.input_map[wire]
        out_node = dag.output_map[wire]

        while cur_node != out_node:
            out_edges = multi_graph.out_edges(cur_node, data=True)
            edges_to_follow = [(src, dest, data) for (src, dest, data) in out_edges
                               if data['wire'] == wire]

            assert len(edges_to_follow) == 1
            cur_node = edges_to_follow[0][1]

    # Wires can only terminate at input/output nodes.
    for op_node in dag.op_nodes():
        assert multi_graph.in_degree(op_node) == multi_graph.out_degree(op_node)

    # Node input/output edges should match node qarg/carg/condition.
    for node in dag.op_nodes():
        in_edges = multi_graph.in_edges(node, data=True)
        out_edges = multi_graph.out_edges(node, data=True)

        in_wires = {data['wire'] for src, dest, data in in_edges}
        out_wires = {data['wire'] for src, dest, data in out_edges}

        node_cond_bits = set(node.condition[0][:] if node.condition is not None else [])
        node_qubits = set(node.qargs)
        node_clbits = set(node.cargs)

        all_bits = node_qubits | node_clbits | node_cond_bits

        assert in_wires == all_bits, 'In-edge wires {} != node bits {}'.format(
            in_wires, all_bits)
        assert out_wires == all_bits, 'Out-edge wires {} != node bits {}'.format(
            out_wires, all_bits)


class TestDagCompose(QiskitTestCase):
    """Test composition of two dags"""

    def setUp(self):
        qreg1 = QuantumRegister(3, 'lqr_1')
        qreg2 = QuantumRegister(2, 'lqr_2')
        creg = ClassicalRegister(2, 'lcr')

        self.circuit_left = QuantumCircuit(qreg1, qreg2, creg)
        self.circuit_left.h(qreg1[0])
        self.circuit_left.x(qreg1[1])
        self.circuit_left.u1(0.1, qreg1[2])
        self.circuit_left.cx(qreg2[0], qreg2[1])

        self.left_qubit0 = qreg1[0]
        self.left_qubit1 = qreg1[1]
        self.left_qubit2 = qreg1[2]
        self.left_qubit3 = qreg2[0]
        self.left_qubit4 = qreg2[1]
        self.left_clbit0 = creg[0]
        self.left_clbit1 = creg[1]
        self.condition = (creg, 3)

    def test_compose_inorder(self):
        """Composing two dags of the same width, default order.

                       ┌───┐
        lqr_1_0: |0>───┤ H ├───     rqr_0: |0>──■───────
                       ├───┤                    │  ┌───┐
        lqr_1_1: |0>───┤ X ├───     rqr_1: |0>──┼──┤ X ├
                    ┌──┴───┴──┐                 │  ├───┤
        lqr_1_2: |0>┤ U1(0.1) ├  +  rqr_2: |0>──┼──┤ Y ├  =
                    └─────────┘               ┌─┴─┐└───┘
        lqr_2_0: |0>─────■─────     rqr_3: |0>┤ X ├─────
                       ┌─┴─┐                  └───┘┌───┐
        lqr_2_1: |0>───┤ X ├───     rqr_4: |0>─────┤ Z ├
                       └───┘                       └───┘
        lcr_0: 0 ═══════════

        lcr_1: 0 ═══════════


                        ┌───┐
         lqr_1_0: |0>───┤ H ├─────■───────
                        ├───┤     │  ┌───┐
         lqr_1_1: |0>───┤ X ├─────┼──┤ X ├
                     ┌──┴───┴──┐  │  ├───┤
         lqr_1_2: |0>┤ U1(0.1) ├──┼──┤ Y ├
                     └─────────┘┌─┴─┐└───┘
         lqr_2_0: |0>─────■─────┤ X ├─────
                        ┌─┴─┐   └───┘┌───┐
         lqr_2_1: |0>───┤ X ├────────┤ Z ├
                        └───┘        └───┘
         lcr_0: 0 ════════════════════════

         lcr_1: 0 ════════════════════════

        """
        qreg = QuantumRegister(5, 'rqr')
        right_qubit0 = qreg[0]
        right_qubit1 = qreg[1]
        right_qubit2 = qreg[2]
        right_qubit3 = qreg[3]
        right_qubit4 = qreg[4]

        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[3])
        circuit_right.x(qreg[1])
        circuit_right.y(qreg[2])
        circuit_right.z(qreg[4])

        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # default wiring: i <- i
        dag_left.compose(dag_right)
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit0, self.left_qubit3)
        circuit_expected.x(self.left_qubit1)
        circuit_expected.y(self.left_qubit2)
        circuit_expected.z(self.left_qubit4)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_inorder_smaller(self):
        """Composing with a smaller RHS dag, default order.

                       ┌───┐                       ┌─────┐
        lqr_1_0: |0>───┤ H ├───     rqr_0: |0>──■──┤ Tdg ├
                       ├───┤                  ┌─┴─┐└─────┘
        lqr_1_1: |0>───┤ X ├───     rqr_1: |0>┤ X ├───────
                    ┌──┴───┴──┐               └───┘
        lqr_1_2: |0>┤ U1(0.1) ├  +                          =
                    └─────────┘
        lqr_2_0: |0>─────■─────
                       ┌─┴─┐
        lqr_2_1: |0>───┤ X ├───
                       └───┘
        lcr_0: 0 ══════════════

        lcr_1: 0 ══════════════

                        ┌───┐        ┌─────┐
         lqr_1_0: |0>───┤ H ├─────■──┤ Tdg ├
                        ├───┤   ┌─┴─┐└─────┘
         lqr_1_1: |0>───┤ X ├───┤ X ├───────
                     ┌──┴───┴──┐└───┘
         lqr_1_2: |0>┤ U1(0.1) ├────────────
                     └─────────┘
         lqr_2_0: |0>─────■─────────────────
                        ┌─┴─┐
         lqr_2_1: |0>───┤ X ├───────────────
                        └───┘
         lcr_0: 0 ══════════════════════════

         lcr_1: 0 ══════════════════════════

        """
        qreg = QuantumRegister(2, 'rqr')
        right_qubit0 = qreg[0]
        right_qubit1 = qreg[1]

        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[1])
        circuit_right.tdg(qreg[0])

        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # default wiring: i <- i
        dag_left.compose(dag_right)
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit0, self.left_qubit1)
        circuit_expected.tdg(self.left_qubit0)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_permuted(self):
        """Composing two dags of the same width, permuted wires.
                        ┌───┐
         lqr_1_0: |0>───┤ H ├───      rqr_0: |0>──■───────
                        ├───┤                     │  ┌───┐
         lqr_1_1: |0>───┤ X ├───      rqr_1: |0>──┼──┤ X ├
                     ┌──┴───┴──┐                  │  ├───┤
         lqr_1_2: |0>┤ U1(0.1) ├      rqr_2: |0>──┼──┤ Y ├
                     └─────────┘                ┌─┴─┐└───┘
         lqr_2_0: |0>─────■─────  +   rqr_3: |0>┤ X ├─────   =
                        ┌─┴─┐                   └───┘┌───┐
         lqr_2_1: |0>───┤ X ├───      rqr_4: |0>─────┤ Z ├
                        └───┘                        └───┘
         lcr_0: 0 ══════════════

         lcr_1: 0 ══════════════

                        ┌───┐   ┌───┐
         lqr_1_0: |0>───┤ H ├───┤ Z ├
                        ├───┤   ├───┤
         lqr_1_1: |0>───┤ X ├───┤ X ├
                     ┌──┴───┴──┐├───┤
         lqr_1_2: |0>┤ U1(0.1) ├┤ Y ├
                     └─────────┘└───┘
         lqr_2_0: |0>─────■───────■──
                        ┌─┴─┐   ┌─┴─┐
         lqr_2_1: |0>───┤ X ├───┤ X ├
                        └───┘   └───┘
         lcr_0: 0 ═══════════════════

         lcr_1: 0 ═══════════════════
        """
        qreg = QuantumRegister(5, 'rqr')
        right_qubit0 = qreg[0]
        right_qubit1 = qreg[1]
        right_qubit2 = qreg[2]
        right_qubit3 = qreg[3]
        right_qubit4 = qreg[4]

        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[3])
        circuit_right.x(qreg[1])
        circuit_right.y(qreg[2])
        circuit_right.z(qreg[4])

        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # permuted wiring
        dag_left.compose(dag_right, edge_map={right_qubit0: self.left_qubit3,
                                              right_qubit1: self.left_qubit1,
                                              right_qubit2: self.left_qubit2,
                                              right_qubit3: self.left_qubit4,
                                              right_qubit4: self.left_qubit0})
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.z(self.left_qubit0)
        circuit_expected.x(self.left_qubit1)
        circuit_expected.y(self.left_qubit2)
        circuit_expected.cx(self.left_qubit3, self.left_qubit4)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_permuted_smaller(self):
        """Composing with a smaller RHS dag, and permuted wires.

                       ┌───┐                       ┌─────┐
        lqr_1_0: |0>───┤ H ├───     rqr_0: |0>──■──┤ Tdg ├
                       ├───┤                  ┌─┴─┐└─────┘
        lqr_1_1: |0>───┤ X ├───     rqr_1: |0>┤ X ├───────
                    ┌──┴───┴──┐               └───┘
        lqr_1_2: |0>┤ U1(0.1) ├  +                          =
                    └─────────┘
        lqr_2_0: |0>─────■─────
                       ┌─┴─┐
        lqr_2_1: |0>───┤ X ├───
                       └───┘
        lcr_0: 0 ══════════════

        lcr_1: 0 ══════════════

                        ┌───┐
         lqr_1_0: |0>───┤ H ├───────────────
                        ├───┤
         lqr_1_1: |0>───┤ X ├───────────────
                     ┌──┴───┴──┐┌───┐
         lqr_1_2: |0>┤ U1(0.1) ├┤ X ├───────
                     └─────────┘└─┬─┘┌─────┐
         lqr_2_0: |0>─────■───────■──┤ Tdg ├
                        ┌─┴─┐        └─────┘
         lqr_2_1: |0>───┤ X ├───────────────
                        └───┘
         lcr_0: 0 ══════════════════════════

         lcr_1: 0 ══════════════════════════
        """
        qreg = QuantumRegister(2, 'rqr')
        right_qubit0 = qreg[0]
        right_qubit1 = qreg[1]

        circuit_right = QuantumCircuit(qreg)
        circuit_right.cx(qreg[0], qreg[1])
        circuit_right.tdg(qreg[0])

        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # permuted wiring of subset
        dag_left.compose(dag_right, edge_map={right_qubit0: self.left_qubit3,
                                              right_qubit1: self.left_qubit2})
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit3, self.left_qubit2)
        circuit_expected.tdg(self.left_qubit3)

        self.assertEqual(circuit_composed, circuit_expected)

    def test_compose_classical(self):
        """Composing on classical bits.

                       ┌───┐                       ┌─────┐┌─┐
        lqr_1_0: |0>───┤ H ├───     rqr_0: |0>──■──┤ Tdg ├┤M├
                       ├───┤                  ┌─┴─┐└─┬─┬─┘└╥┘
        lqr_1_1: |0>───┤ X ├───     rqr_1: |0>┤ X ├──┤M├───╫─
                    ┌──┴───┴──┐               └───┘  └╥┘   ║
        lqr_1_2: |0>┤ U1(0.1) ├  +   rcr_0: 0 ════════╬════╩═  =
                    └─────────┘                       ║
        lqr_2_0: |0>─────■─────      rcr_1: 0 ════════╩══════
                       ┌─┴─┐
        lqr_2_1: |0>───┤ X ├───
                       └───┘
        lcr_0: 0 ══════════════

        lcr_1: 0 ══════════════

                       ┌─────┐┌─┐
        rqr_0: |0>──■──┤ Tdg ├┤M├
                  ┌─┴─┐└─┬─┬─┘└╥┘
        rqr_1: |0>┤ X ├──┤M├───╫─
                  └───┘  └╥┘   ║
         rcr_0: 0 ════════╬════╩═
                          ║
         rcr_1: 0 ════════╩══════

                       ┌───┐
        lqr_1_0: |0>───┤ H ├──────────────────
                       ├───┤        ┌─────┐┌─┐
        lqr_1_1: |0>───┤ X ├─────■──┤ Tdg ├┤M├
                    ┌──┴───┴──┐  │  └─────┘└╥┘
        lqr_1_2: |0>┤ U1(0.1) ├──┼──────────╫─
                    └─────────┘  │          ║
        lqr_2_0: |0>─────■───────┼──────────╫─
                       ┌─┴─┐   ┌─┴─┐  ┌─┐   ║
        lqr_2_1: |0>───┤ X ├───┤ X ├──┤M├───╫─
                       └───┘   └───┘  └╥┘   ║
           lcr_0: 0 ═══════════════════╩════╬═
                                            ║
           lcr_1: 0 ════════════════════════╩═
        """
        qreg = QuantumRegister(2, 'rqr')
        creg = ClassicalRegister(2, 'rcr')
        right_qubit0 = qreg[0]
        right_qubit1 = qreg[1]
        right_clbit0 = creg[0]
        right_clbit1 = creg[1]

        circuit_right = QuantumCircuit(qreg, creg)
        circuit_right.cx(qreg[0], qreg[1])
        circuit_right.tdg(qreg[0])
        circuit_right.measure(qreg, creg)

        dag_left = circuit_to_dag(self.circuit_left)
        dag_right = circuit_to_dag(circuit_right)

        # permuted subset of qubits and clbits
        dag_left.compose(dag_right, edge_map={right_qubit0: self.left_qubit1,
                                              right_qubit1: self.left_qubit4,
                                              right_clbit0: self.left_clbit1,
                                              right_clbit1: self.left_clbit0})
        circuit_composed = dag_to_circuit(dag_left)

        circuit_expected = self.circuit_left.copy()
        circuit_expected.cx(self.left_qubit1, self.left_qubit4)
        circuit_expected.tdg(self.left_qubit1)
        circuit_expected.measure(self.left_qubit4, self.left_clbit0)
        circuit_expected.measure(self.left_qubit1, self.left_clbit1)

        self.assertEqual(circuit_composed, circuit_expected)


if __name__ == '__main__':
    unittest.main()
