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

"""Test the decompose pass"""

from numpy import pi

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.transpiler.passes import Decompose
from qiskit.converters import circuit_to_dag
from qiskit.circuit.library import HGate
from qiskit.circuit.library import CCXGate
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
        op_nodes = after_dag.op_nodes()
        self.assertEqual(len(op_nodes), 1)
        self.assertEqual(op_nodes[0].name, 'u2')

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
        op_nodes = after_dag.op_nodes()
        self.assertEqual(len(op_nodes), 2)
        for node in op_nodes:
            self.assertIn(node.name, ['cx', 'u2'])

    def test_decompose_toffoli(self):
        """Test decompose CCX.
        """
        qr1 = QuantumRegister(2, 'qr1')
        qr2 = QuantumRegister(1, 'qr2')
        circuit = QuantumCircuit(qr1, qr2)
        circuit.ccx(qr1[0], qr1[1], qr2[0])
        dag = circuit_to_dag(circuit)
        pass_ = Decompose(CCXGate)
        after_dag = pass_.run(dag)
        op_nodes = after_dag.op_nodes()
        self.assertEqual(len(op_nodes), 15)
        for node in op_nodes:
            self.assertIn(node.name, ['h', 't', 'tdg', 'cx'])

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

    def test_decompose_oversized_instruction(self):
        """Test decompose on a single-op gate that doesn't use all qubits."""
        # ref: https://github.com/Qiskit/qiskit-terra/issues/3440
        qc1 = QuantumCircuit(2)
        qc1.x(0)
        gate = qc1.to_gate()

        qc2 = QuantumCircuit(2)
        qc2.append(gate, [0, 1])

        output = qc2.decompose()

        self.assertEqual(qc1, output)
