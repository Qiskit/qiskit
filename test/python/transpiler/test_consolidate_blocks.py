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

"""
Tests for the ConsolidateBlocks transpiler pass.
"""

import unittest
import numpy as np

from qiskit.circuit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import U2Gate
from qiskit.extensions import UnitaryGate
from qiskit.converters import circuit_to_dag
from qiskit.transpiler.passes import ConsolidateBlocks
from qiskit.quantum_info.operators import Operator
from qiskit.quantum_info.operators.measures import process_fidelity
from qiskit.test import QiskitTestCase
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks


class TestConsolidateBlocks(QiskitTestCase):
    """
    Tests to verify that consolidating blocks of gates into unitaries
    works correctly.
    """

    def test_consolidate_small_block(self):
        """test a small block of gates can be turned into a unitary on same wires"""
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        qc.p(0.5, qr[0])
        qc.u(1.5708, 0.2, 0.6, qr[1])
        qc.cx(qr[0], qr[1])
        dag = circuit_to_dag(qc)

        pass_ = ConsolidateBlocks(force_consolidate=True)
        pass_.property_set["block_list"] = [list(dag.topological_op_nodes())]
        new_dag = pass_.run(dag)

        unitary = Operator(qc)
        self.assertEqual(len(new_dag.op_nodes()), 1)
        fidelity = process_fidelity(Operator(new_dag.op_nodes()[0].op), unitary)
        self.assertAlmostEqual(fidelity, 1.0, places=7)

    def test_wire_order(self):
        """order of qubits and the corresponding unitary is correct"""
        qr = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qr)
        qc.cx(qr[1], qr[0])
        dag = circuit_to_dag(qc)

        pass_ = ConsolidateBlocks(force_consolidate=True)
        pass_.property_set["block_list"] = [dag.op_nodes()]
        new_dag = pass_.run(dag)

        new_node = new_dag.op_nodes()[0]
        self.assertEqual(new_node.qargs, [qr[0], qr[1]])
        unitary = Operator(qc)
        fidelity = process_fidelity(Operator(new_node.op), unitary)
        self.assertAlmostEqual(fidelity, 1.0, places=7)

    def test_topological_order_preserved(self):
        """the original topological order of nodes is preserved
                                                   ______
        q0:--[p]-------.----      q0:-------------|      |--
                       |                 ______   |  U2  |
        q1:--[u2]--(+)-(+)--   =  q1:---|      |--|______|--
                    |                   |  U1  |
        q2:---------.-------      q2:---|______|------------
        """
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.p(0.5, qr[0])
        qc.u(1.5708, 0.2, 0.6, qr[1])
        qc.cx(qr[2], qr[1])
        qc.cx(qr[0], qr[1])
        dag = circuit_to_dag(qc)

        pass_ = ConsolidateBlocks(force_consolidate=True)
        topo_ops = list(dag.topological_op_nodes())
        block_1 = [topo_ops[1], topo_ops[2]]
        block_2 = [topo_ops[0], topo_ops[3]]
        pass_.property_set["block_list"] = [block_1, block_2]
        new_dag = pass_.run(dag)

        new_topo_ops = [i for i in new_dag.topological_op_nodes() if i.type == "op"]
        self.assertEqual(len(new_topo_ops), 2)
        self.assertEqual(new_topo_ops[0].qargs, [qr[1], qr[2]])
        self.assertEqual(new_topo_ops[1].qargs, [qr[0], qr[1]])

    def test_3q_blocks(self):
        """blocks of more than 2 qubits work."""
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.p(0.5, qr[0])
        qc.u(1.5708, 0.2, 0.6, qr[1])
        qc.cx(qr[2], qr[1])
        qc.cx(qr[0], qr[1])
        dag = circuit_to_dag(qc)

        pass_ = ConsolidateBlocks(force_consolidate=True)
        pass_.property_set["block_list"] = [list(dag.topological_op_nodes())]
        new_dag = pass_.run(dag)

        unitary = Operator(qc)
        self.assertEqual(len(new_dag.op_nodes()), 1)
        fidelity = process_fidelity(Operator(new_dag.op_nodes()[0].op), unitary)
        self.assertAlmostEqual(fidelity, 1.0, places=7)

    def test_block_spanning_two_regs(self):
        """blocks spanning wires on different quantum registers work."""
        qr0 = QuantumRegister(1, "qr0")
        qr1 = QuantumRegister(1, "qr1")
        qc = QuantumCircuit(qr0, qr1)
        qc.p(0.5, qr0[0])
        qc.u(1.5708, 0.2, 0.6, qr1[0])
        qc.cx(qr0[0], qr1[0])
        dag = circuit_to_dag(qc)

        pass_ = ConsolidateBlocks(force_consolidate=True)
        pass_.property_set["block_list"] = [list(dag.topological_op_nodes())]
        new_dag = pass_.run(dag)

        unitary = Operator(qc)
        self.assertEqual(len(new_dag.op_nodes()), 1)
        fidelity = process_fidelity(Operator(new_dag.op_nodes()[0].op), unitary)
        self.assertAlmostEqual(fidelity, 1.0, places=7)

    def test_block_spanning_two_regs_different_index(self):
        """blocks spanning wires on different quantum registers work when the wires
        could have conflicting indices. This was raised in #2806 when a CX was applied
        across multiple registers and their indices collided, raising an error."""
        qr0 = QuantumRegister(1, "qr0")
        qr1 = QuantumRegister(2, "qr1")
        qc = QuantumCircuit(qr0, qr1)
        qc.cx(qr0[0], qr1[1])
        dag = circuit_to_dag(qc)

        pass_ = ConsolidateBlocks(force_consolidate=True)
        pass_.property_set["block_list"] = [list(dag.topological_op_nodes())]
        new_dag = pass_.run(dag)

        original_unitary = UnitaryGate(Operator(qc))

        from qiskit.converters import dag_to_circuit

        new_unitary = UnitaryGate(Operator(dag_to_circuit(new_dag)))

        self.assertEqual(original_unitary, new_unitary)

    def test_node_added_before_block(self):
        """Test that a node before a block remains before the block

        This issue was raised in #2737 where the measure was moved
        to be after the 2nd ID gate, as the block was added when the
        first node in the block was seen.

        blocks = [['id', 'cx', 'id']]

                ┌────┐┌───┐
        q_0: |0>┤ Id ├┤ X ├──────
                └┬─┬─┘└─┬─┘┌────┐
        q_1: |0>─┤M├────■──┤ Id ├
                 └╥┘       └────┘
        c_0:  0 ══╩══════════════
        """
        qc = QuantumCircuit(2, 1)
        qc.i(0)
        qc.measure(1, 0)
        qc.cx(1, 0)
        qc.i(1)

        # can't just add all the nodes to one block as in other tests
        # as we are trying to test the block gets added in the correct place
        # so use a pass to collect the blocks instead
        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(ConsolidateBlocks())
        qc1 = pass_manager.run(qc)

        self.assertEqual(qc, qc1)

    def test_consolidate_blocks_big(self):
        """Test ConsolidateBlocks with U2(<big numbers>)
        https://github.com/Qiskit/qiskit-terra/issues/3637#issuecomment-612954865

             ┌────────────────┐     ┌───┐
        q_0: ┤ U2(-804.15,pi) ├──■──┤ X ├
             ├────────────────┤┌─┴─┐└─┬─┘
        q_1: ┤ U2(-6433.2,pi) ├┤ X ├──■──
             └────────────────┘└───┘
        """
        circuit = QuantumCircuit(2)
        circuit.append(U2Gate(-804.15, np.pi), [0])
        circuit.append(U2Gate(-6433.2, np.pi), [1])
        circuit.cx(0, 1)
        circuit.cx(1, 0)

        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(ConsolidateBlocks())
        result = pass_manager.run(circuit)

        self.assertEqual(circuit, result)

    def test_node_added_after_block(self):
        """Test that a node after the block remains after the block

        This example was raised in #2764, and checks that the final CX
        stays after the main block, even though one of the nodes in the
        block was declared after it. This occurred when the block was
        added when the last node in the block was seen.

        blocks = [['cx', 'id', 'id']]

        q_0: |0>─────────────■──
                     ┌────┐┌─┴─┐
        q_1: |0>──■──┤ Id ├┤ X ├
                ┌─┴─┐├────┤└───┘
        q_2: |0>┤ X ├┤ Id ├─────
                └───┘└────┘
        """
        qc = QuantumCircuit(3)
        qc.cx(1, 2)
        qc.i(1)
        qc.cx(0, 1)
        qc.i(2)

        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(ConsolidateBlocks())
        qc1 = pass_manager.run(qc)

        self.assertEqual(qc, qc1)

    def test_node_middle_of_blocks(self):
        """Test that a node surrounded by blocks stays in the same place

        This is a larger test to ensure multiple blocks can all be collected
        and added back in the correct order.

        blocks = [['cx', 'id'], ['cx', 'id'], ['id', 'cx'], ['id', 'cx']]

        q_0: |0>──■───────────────────■──
                ┌─┴─┐┌────┐   ┌────┐┌─┴─┐
        q_1: |0>┤ X ├┤ Id ├─X─┤ Id ├┤ X ├
                ├───┤├────┤ │ ├────┤├───┤
        q_2: |0>┤ X ├┤ Id ├─X─┤ Id ├┤ X ├
                └─┬─┘└────┘   └────┘└─┬─┘
        q_3: |0>──■───────────────────■──

        """
        qc = QuantumCircuit(4)
        qc.cx(0, 1)
        qc.cx(3, 2)
        qc.i(1)
        qc.i(2)

        qc.swap(1, 2)

        qc.i(1)
        qc.i(2)
        qc.cx(0, 1)
        qc.cx(3, 2)

        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(ConsolidateBlocks())
        qc1 = pass_manager.run(qc)

        self.assertEqual(qc, qc1)

    def test_classical_conditions_maintained(self):
        """Test that consolidate blocks doesn't drop the classical conditions
        This issue was raised in #2752
        """
        qc = QuantumCircuit(1, 1)
        qc.h(0).c_if(qc.cregs[0], 1)
        qc.measure(0, 0)

        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(ConsolidateBlocks())
        qc1 = pass_manager.run(qc)

        self.assertEqual(qc, qc1)

    def test_no_kak_in_basis(self):
        """Test that pass just returns the input dag without a KAK gate."""
        qc = QuantumCircuit(1)
        qc.h(0)
        dag = circuit_to_dag(qc)
        consolidate_blocks_pass = ConsolidateBlocks(basis_gates=["u3"])
        res = consolidate_blocks_pass.run(dag)
        self.assertEqual(res, dag)

    def test_single_gate_block_outside_basis(self):
        """Test that a single gate block outside the configured basis gets converted."""
        qc = QuantumCircuit(2)
        qc.swap(0, 1)
        consolidate_block_pass = ConsolidateBlocks(basis_gates=["id", "cx", "rz", "sx", "x"])
        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())
        pass_manager.append(consolidate_block_pass)
        expected = QuantumCircuit(2)
        expected.unitary(np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]), [0, 1])
        self.assertEqual(expected, pass_manager.run(qc))


if __name__ == "__main__":
    unittest.main()
