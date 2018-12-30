# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Test the Unroller pass"""

from sympy import pi

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.transpiler.passes import Unroller
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase


class TestUnroller(QiskitTestCase):
    """Tests the Unroller pass."""

    def test_basic_unroll(self):
        """Test decompose a single H into u2.
        """
        qr = QuantumRegister(1, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        dag = circuit_to_dag(circuit)
        pass_ = Unroller(['u2'])
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.get_op_nodes(data=True)
        self.assertEqual(len(op_nodes), 1)
        self.assertEqual(op_nodes[0][1]["op"].name, 'u2')

    def test_unroll_no_basis(self):
        """Test no-basis unrolls all the way to U, CX.
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.cx(qr[0], qr[1])
        dag = circuit_to_dag(circuit)
        pass_ = Unroller()
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.get_op_nodes(data=True)
        self.assertEqual(len(op_nodes), 2)
        for node in op_nodes:
            op = node[1]["op"]
            self.assertIn(op.name, ['U', 'CX'])

    def test_unroll_toffoli(self):
        """Test unroll toffoli on multi regs to h, t, tdg, cx.
        """
        qr1 = QuantumRegister(2, 'qr1')
        qr2 = QuantumRegister(1, 'qr2')
        circuit = QuantumCircuit(qr1, qr2)
        circuit.ccx(qr1[0], qr1[1], qr2[0])
        dag = circuit_to_dag(circuit)
        pass_ = Unroller(['h', 't', 'tdg', 'cx'])
        unrolled_dag = pass_.run(dag)
        op_nodes = unrolled_dag.get_op_nodes(data=True)
        self.assertEqual(len(op_nodes), 15)
        for node in op_nodes:
            op = node[1]["op"]
            self.assertIn(op.name, ['h', 't', 'tdg', 'cx'])

    def test_unroll_1q_chain_conditional(self):
        """Test unroll chain of 1-qubit gates interrupted by conditional.
        """
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        circuit.tdg(qr)
        circuit.z(qr)
        circuit.t(qr)
        circuit.ry(0.5, qr)
        circuit.rz(0.3, qr)
        circuit.rx(0.1, qr)
        circuit.measure(qr, cr)
        circuit.x(qr).c_if(cr, 1)
        circuit.y(qr).c_if(cr, 1)
        circuit.z(qr).c_if(cr, 1)
        dag = circuit_to_dag(circuit)
        pass_ = Unroller(['u1', 'u2', 'u3'])
        unrolled_dag = pass_.run(dag)

        ref_circuit = QuantumCircuit(qr, cr)
        ref_circuit.u2(0, pi, qr[0])
        ref_circuit.u1(-pi/4, qr[0])
        ref_circuit.u1(pi, qr[0])
        ref_circuit.u1(pi/4, qr[0])
        ref_circuit.u3(0.5, 0, 0, qr[0])
        ref_circuit.u1(0.3, qr[0])
        ref_circuit.u3(0.1, -pi/2, pi/2, qr[0])
        ref_circuit.measure(qr[0], cr[0])
        ref_circuit.u3(pi, 0, pi, qr[0]).c_if(cr, 1)
        ref_circuit.u3(pi, pi/2, pi/2, qr[0]).c_if(cr, 1)
        ref_circuit.u1(pi, qr[0]).c_if(cr, 1)
        ref_dag = circuit_to_dag(ref_circuit)
        self.assertEqual(unrolled_dag, ref_dag)
