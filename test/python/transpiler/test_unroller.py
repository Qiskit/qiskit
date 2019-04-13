# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=unused-import

"""Test the Unroller pass"""

from sympy import pi

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.simulator import snapshot
from qiskit.transpiler.passes import Unroller
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.exceptions import QiskitError


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
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 1)
        self.assertEqual(op_nodes[0].name, 'u2')

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
        op_nodes = unrolled_dag.op_nodes()
        self.assertEqual(len(op_nodes), 15)
        for node in op_nodes:
            self.assertIn(node.name, ['h', 't', 'tdg', 'cx'])

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

    def test_unroll_no_basis(self):
        """Test when a given gate has no decompositions.
        """
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.h(qr)
        dag = circuit_to_dag(circuit)
        pass_ = Unroller(basis=[])

        with self.assertRaises(QiskitError):
            pass_.run(dag)

    def test_unroll_all_instructions(self):
        """Test unrolling a circuit containing all standard instructions.
        """
        qr = QuantumRegister(3, 'qr')
        cr = ClassicalRegister(3, 'cr')
        circuit = QuantumCircuit(qr, cr)
        circuit.ccx(qr[0], qr[1], qr[2])
        circuit.ch(qr[0], qr[2])
        circuit.crz(0.5, qr[1], qr[2])
        circuit.cswap(qr[1], qr[0], qr[2])
        circuit.cu1(0.1, qr[0], qr[2])
        circuit.cu3(0.2, 0.1, 0.0, qr[1], qr[2])
        circuit.cx(qr[1], qr[0])
        circuit.cy(qr[1], qr[2])
        circuit.cz(qr[2], qr[0])
        circuit.h(qr[1])
        circuit.iden(qr[0])
        circuit.rx(0.1, qr[0])
        circuit.ry(0.2, qr[1])
        circuit.rz(0.3, qr[2])
        circuit.rzz(0.6, qr[1], qr[0])
        circuit.s(qr[0])
        circuit.sdg(qr[1])
        circuit.swap(qr[1], qr[2])
        circuit.t(qr[2])
        circuit.tdg(qr[0])
        circuit.u0(1, qr[0])
        circuit.u1(0.1, qr[1])
        circuit.u2(0.2, -0.1, qr[0])
        circuit.u3(0.3, 0.0, -0.1, qr[2])
        circuit.x(qr[2])
        circuit.y(qr[1])
        circuit.z(qr[0])
        circuit.snapshot('0')
        circuit.measure(qr, cr)
        dag = circuit_to_dag(circuit)
        pass_ = Unroller(basis=['u3', 'cx'])
        unrolled_dag = pass_.run(dag)

        ref_circuit = QuantumCircuit(qr, cr)
        ref_circuit.u3(pi/2, 0, pi, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(0, 0, -pi/4, qr[2])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.u3(0, 0, pi/4, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(0, 0, pi/4, qr[1])
        ref_circuit.u3(0, 0, -pi/4, qr[2])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.cx(qr[0], qr[1])
        ref_circuit.u3(0, 0, pi/4, qr[0])
        ref_circuit.u3(0, 0, -pi/4, qr[1])
        ref_circuit.cx(qr[0], qr[1])
        ref_circuit.u3(0, 0, pi/4, qr[2])
        ref_circuit.u3(pi/2, 0, pi, qr[2])
        ref_circuit.u3(pi/2, 0, pi, qr[2])
        ref_circuit.u3(0, 0, -pi/2, qr[2])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.u3(pi/2, 0, pi, qr[2])
        ref_circuit.u3(0, 0, pi/4, qr[2])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.u3(0, 0, pi/2, qr[0])
        ref_circuit.u3(0, 0, pi/4, qr[2])
        ref_circuit.u3(pi/2, 0, pi, qr[2])
        ref_circuit.u3(0, 0, pi/2, qr[2])
        ref_circuit.u3(pi, 0, pi, qr[2])
        ref_circuit.u3(0, 0, 0.25, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(0, 0, -0.25, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[2], qr[0])
        ref_circuit.u3(pi/2, 0, pi, qr[2])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.u3(0, 0, -pi/4, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(0, 0, pi/4, qr[2])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.u3(0, 0, pi/4, qr[0])
        ref_circuit.u3(0, 0, -pi/4, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.u3(0, 0, -pi/4, qr[0])
        ref_circuit.u3(0, 0, pi/4, qr[1])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.u3(0, 0, pi/4, qr[2])
        ref_circuit.u3(pi/2, 0, pi, qr[2])
        ref_circuit.cx(qr[2], qr[0])
        ref_circuit.u3(0, 0, 0.05, qr[0])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.u3(0, 0, -0.05, qr[2])
        ref_circuit.cx(qr[0], qr[2])
        ref_circuit.u3(0, 0, 0.05, qr[2])
        ref_circuit.u3(0, 0, -0.05, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(-0.1, 0, -0.05, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.u3(pi/2, 0, pi, qr[0])
        ref_circuit.u3(0.1, 0.1, 0, qr[2])
        ref_circuit.u3(0, 0, -pi/2, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(pi/2, 0, pi, qr[1])
        ref_circuit.u3(0.2, 0, 0, qr[1])
        ref_circuit.u3(0, 0, pi/2, qr[2])
        ref_circuit.cx(qr[2], qr[0])
        ref_circuit.u3(pi/2, 0, pi, qr[0])
        ref_circuit.u3(0, 0, 0, qr[0])
        ref_circuit.u3(0.1, -pi/2, pi/2, qr[0])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.u3(0, 0, 0.6, qr[0])
        ref_circuit.cx(qr[1], qr[0])
        ref_circuit.u3(0, 0, pi/2, qr[0])
        ref_circuit.u3(0, 0, -pi/4, qr[0])
        ref_circuit.u3(0, 0, 0, qr[0])
        ref_circuit.u3(pi/2, 0.2, -0.1, qr[0])
        ref_circuit.u3(0, 0, pi, qr[0])
        ref_circuit.u3(0, 0, -pi/2, qr[1])
        ref_circuit.u3(0, 0, 0.3, qr[2])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.cx(qr[2], qr[1])
        ref_circuit.cx(qr[1], qr[2])
        ref_circuit.u3(0, 0, 0.1, qr[1])
        ref_circuit.u3(pi, pi/2, pi/2, qr[1])
        ref_circuit.u3(0, 0, pi/4, qr[2])
        ref_circuit.u3(0.3, 0.0, -0.1, qr[2])
        ref_circuit.u3(pi, 0, pi, qr[2])
        ref_circuit.snapshot('0')
        ref_circuit.measure(qr, cr)
        ref_dag = circuit_to_dag(ref_circuit)
        self.assertEqual(unrolled_dag, ref_dag)
