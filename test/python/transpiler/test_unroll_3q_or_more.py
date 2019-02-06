# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the Unroll3qOrMore pass"""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.transpiler.passes import Unroll3qOrMore
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestUnroll3qOrMore(QiskitTestCase):
    """Tests the Unroll3qOrMore pass, for unrolling all
    gates until reaching only 1q or 2q gates."""

    def test_ccx(self):
        """Test decompose CCX.
        """
        qr1 = QuantumRegister(2, 'qr1')
        qr2 = QuantumRegister(1, 'qr2')
        circuit = QuantumCircuit(qr1, qr2)
        circuit.ccx(qr1[0], qr1[1], qr2[0])
        dag = circuit_to_dag(circuit)
        pass_ = Unroll3qOrMore()
        after_dag = pass_.run(dag)
        op_nodes = after_dag.op_nodes(data=True)
        self.assertEqual(len(op_nodes), 15)
        for node in op_nodes:
            op = node[1]["op"]
            self.assertIn(op.name, ['h', 't', 'tdg', 'cx'])

    def test_cswap(self):
        """Test decompose CSwap (recursively).
        """
        qr1 = QuantumRegister(2, 'qr1')
        qr2 = QuantumRegister(1, 'qr2')
        circuit = QuantumCircuit(qr1, qr2)
        circuit.cswap(qr1[0], qr1[1], qr2[0])
        dag = circuit_to_dag(circuit)
        pass_ = Unroll3qOrMore()
        after_dag = pass_.run(dag)
        op_nodes = after_dag.op_nodes(data=True)
        self.assertEqual(len(op_nodes), 17)
        for node in op_nodes:
            op = node[1]["op"]
            self.assertIn(op.name, ['h', 't', 'tdg', 'cx'])

    def test_decompose_conditional(self):
        """Test decompose a 3-qubit gate with a conditional.
        """
        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.ccx(qr[0], qr[1], qr[2]).c_if(cr, 0)
        dag = circuit_to_dag(circuit)
        pass_ = Unroll3qOrMore()
        after_dag = pass_.run(dag)
        op_nodes = after_dag.op_nodes(data=True)
        self.assertEqual(len(op_nodes), 15)
        for node in op_nodes:
            op = node[1]["op"]
            self.assertIn(op.name, ['h', 't', 'tdg', 'cx'])
            self.assertEqual(node[1]['condition'], (cr, 0))
