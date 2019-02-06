# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the decompose pass"""

from sympy import pi

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.transpiler.passes import Decompose
from qiskit.converters import circuit_to_dag
from qiskit.extensions.standard import HGate
from qiskit.extensions.standard import ToffoliGate
from qiskit.test import QiskitTestCase


class TestDecompose(QiskitTestCase):
    """Tests the decompose pass."""

    def test_basic(self):
        """Test decompose a single H into u2.
        """
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)
        pass_ = Decompose(HGate)
        after_dag = pass_.run(dag)
        op_nodes = after_dag.op_nodes(data=True)
        self.assertEqual(len(op_nodes), 1)
        self.assertEqual(op_nodes[0][1]["op"].name, 'u2')

    def test_decompose_only_h(self):
        """Test to decompose a single H, without the rest
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        dag = circuit_to_dag(circuit)
        pass_ = Decompose(HGate)
        after_dag = pass_.run(dag)
        op_nodes = after_dag.op_nodes(data=True)
        self.assertEqual(len(op_nodes), 2)
        for node in op_nodes:
            op = node[1]["op"]
            self.assertIn(op.name, ['cx', 'u2'])

    def test_decompose_toffoli(self):
        """Test decompose CCX.
        """
        qr1 = QuantumRegister(2, 'qr1')
        qr2 = QuantumRegister(1, 'qr2')
        circuit = QuantumCircuit(qr1, qr2)
        circuit.ccx(qr1[0], qr1[1], qr2[0])
        dag = circuit_to_dag(circuit)
        pass_ = Decompose(ToffoliGate)
        after_dag = pass_.run(dag)
        op_nodes = after_dag.op_nodes(data=True)
        self.assertEqual(len(op_nodes), 15)
        for node in op_nodes:
            op = node[1]["op"]
            self.assertIn(op.name, ['h', 't', 'tdg', 'cx'])

    def test_decompose_conditional(self):
        """Test decompose a 1-qubit gates with a conditional.
        """
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr).c_if(cr, 1)
        circuit.x(qr).c_if(cr, 1)
        dag = circuit_to_dag(circuit)
        pass_ = Decompose(HGate)
        after_dag = pass_.run(dag)

        ref_circuit = QuantumCircuit(qr, cr)
        ref_circuit.u2(0, pi, qr[0]).c_if(cr, 1)
        ref_circuit.x(qr).c_if(cr, 1)
        ref_dag = circuit_to_dag(ref_circuit)

        self.assertEqual(after_dag, ref_dag)
