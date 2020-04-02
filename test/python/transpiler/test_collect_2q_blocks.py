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

"""
Tests for the Collect2qBlocks transpiler pass.
"""

import unittest

from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.converters import circuit_to_dag
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Collect2qBlocks
from qiskit.test import QiskitTestCase


class TestCollect2qBlocks(QiskitTestCase):
    """
    Tests to verify that blocks of 2q interactions are found correctly.
    """

    def test_blocks_in_topological_order(self):
        """the pass returns blocks in correct topological order
                                                     ______
         q0:--[u1]-------.----      q0:-------------|      |--
                         |                 ______   |  U2  |
         q1:--[u2]--(+)-(+)---   =  q1:---|      |--|______|--
                     |                    |  U1  |
         q2:---------.--------      q2:---|______|------------
        """
        qr = QuantumRegister(3, "qr")
        qc = QuantumCircuit(qr)
        qc.u1(0.5, qr[0])
        qc.u2(0.2, 0.6, qr[1])
        qc.cx(qr[2], qr[1])
        qc.cx(qr[0], qr[1])
        dag = circuit_to_dag(qc)

        topo_ops = list(dag.topological_op_nodes())
        block_1 = [topo_ops[1], topo_ops[2]]
        block_2 = [topo_ops[0], topo_ops[3]]

        pass_ = Collect2qBlocks()
        pass_.run(dag)
        self.assertTrue(pass_.property_set['block_list'], [block_1, block_2])

    def test_block_interrupted_by_gate(self):
        """Test that blocks interrupted by a gate that can't be added
        to the block can be collected correctly

        This was raised in #2775 where a measure in the middle of a block
        stopped the block collection from working properly. This was because
        the pass didn't expect to have measures in the middle of the circuit.

        blocks : [['cx', 'id', 'id', 'id'], ['id', 'cx']]

                ┌───┐┌───┐┌─┐     ┌───┐┌───┐
        q_0: |0>┤ X ├┤ I ├┤M├─────┤ I ├┤ X ├
                └─┬─┘├───┤└╥┘┌───┐└───┘└─┬─┘
        q_1: |0>──■──┤ I ├─╫─┤ I ├───────■──
                     └───┘ ║ └───┘
         c_0: 0 ═══════════╩════════════════

        """
        qc = QuantumCircuit(2, 1)
        qc.cx(1, 0)
        qc.i(0)
        qc.i(1)
        qc.measure(0, 0)
        qc.i(0)
        qc.i(1)
        qc.cx(1, 0)

        dag = circuit_to_dag(qc)
        pass_ = Collect2qBlocks()
        pass_.run(dag)

        # list from Collect2QBlocks of nodes that it should have put into blocks
        good_names = ['cx', 'u1', 'u2', 'u3', 'id']
        dag_nodes = [node for node in dag.topological_op_nodes() if node.name in good_names]

        # we have to convert them to sets as the ordering can be different
        # but equivalent between python 3.5 and 3.7
        # there is no implied topology in a block, so this isn't an issue
        dag_nodes = [set(dag_nodes[:4]), set(dag_nodes[4:])]
        pass_nodes = [set(bl) for bl in pass_.property_set['block_list']]

        self.assertEqual(dag_nodes, pass_nodes)

    def test_block_with_classical_register(self):
        """Test that only blocks that share quantum wires are added to the block.
        It was the case that gates which shared a classical wire could be added to
        the same block, despite not sharing the same qubits. This was fixed in #2956.

                                    ┌─────────────────────┐
        q_0: |0>────────────────────┤ U2(0.25*pi,0.25*pi) ├
                     ┌─────────────┐└──────────┬──────────┘
        q_1: |0>──■──┤ U1(0.25*pi) ├───────────┼───────────
                ┌─┴─┐└──────┬──────┘           │
        q_2: |0>┤ X ├───────┼──────────────────┼───────────
                └───┘    ┌──┴──┐            ┌──┴──┐
        c0_0: 0 ═════════╡ = 0 ╞════════════╡ = 0 ╞════════
                         └─────┘            └─────┘

        Previously the blocks collected were : [['cx', 'u1', 'u2']]
        This is now corrected to : [['cx', 'u1']]
        """

        qasmstr = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c0[1];

        cx q[1],q[2];
        if(c0==0) u1(0.25*pi) q[1];
        if(c0==0) u2(0.25*pi, 0.25*pi) q[0];
        """
        qc = QuantumCircuit.from_qasm_str(qasmstr)

        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())

        pass_manager.run(qc)

        self.assertEqual([['cx']],
                         [[n.name for n in block]
                          for block in pass_manager.property_set['block_list']])

    def test_do_not_merge_conditioned_gates(self):
        """Validate that classically conditioned gates are never considered for
        inclusion in a block. Note that there are cases where gates conditioned
        on the same (register, value) pair could be correctly merged, but this is
        not yet implemented.

                 ┌─────────┐┌─────────┐┌─────────┐      ┌───┐
        qr_0: |0>┤ U1(0.1) ├┤ U1(0.2) ├┤ U1(0.3) ├──■───┤ X ├────■───
                 └─────────┘└────┬────┘└────┬────┘┌─┴─┐ └─┬─┘  ┌─┴─┐
        qr_1: |0>────────────────┼──────────┼─────┤ X ├───■────┤ X ├─
                                 │          │     └───┘   │    └─┬─┘
        qr_2: |0>────────────────┼──────────┼─────────────┼──────┼───
                              ┌──┴──┐    ┌──┴──┐       ┌──┴──┐┌──┴──┐
         cr_0: 0 ═════════════╡     ╞════╡     ╞═══════╡     ╞╡     ╞
                              │ = 0 │    │ = 0 │       │ = 0 ││ = 1 │
         cr_1: 0 ═════════════╡     ╞════╡     ╞═══════╡     ╞╡     ╞
                              └─────┘    └─────┘       └─────┘└─────┘

        Previously the blocks collected were : [['u1', 'u1', 'u1', 'cx', 'cx', 'cx']]
        This is now corrected to : [['cx']]
        """
        # ref: https://github.com/Qiskit/qiskit-terra/issues/3215

        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(2, 'cr')

        qc = QuantumCircuit(qr, cr)
        qc.u1(0.1, 0)
        qc.u1(0.2, 0).c_if(cr, 0)
        qc.u1(0.3, 0).c_if(cr, 0)
        qc.cx(0, 1)
        qc.cx(1, 0).c_if(cr, 0)
        qc.cx(0, 1).c_if(cr, 1)

        pass_manager = PassManager()
        pass_manager.append(Collect2qBlocks())

        pass_manager.run(qc)
        self.assertEqual([['cx']],
                         [[n.name for n in block]
                          for block in pass_manager.property_set['block_list']])


if __name__ == '__main__':
    unittest.main()
