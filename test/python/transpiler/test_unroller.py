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

# pylint: disable=unused-import

"""Test the Unroller pass"""

from numpy import pi

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.extensions.simulator import snapshot
from qiskit.transpiler.passes import Unroller
from qiskit.converters import circuit_to_dag
from qiskit.test import QiskitTestCase
from qiskit.exceptions import QiskitError
from qiskit.circuit import Parameter


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

    def test_simple_unroll_parameterized_without_expressions(self):
        """Verify unrolling parameterized gates without expressions."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        theta = Parameter('theta')

        qc.rz(theta, qr[0])
        dag = circuit_to_dag(qc)

        unrolled_dag = Unroller(['u1', 'u3', 'cx']).run(dag)

        expected = QuantumCircuit(qr)
        expected.u1(theta, qr[0])

        self.assertEqual(circuit_to_dag(expected), unrolled_dag)

    def test_simple_unroll_parameterized_with_expressions(self):
        """Verify unrolling parameterized gates with expressions."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)

        theta = Parameter('theta')
        phi = Parameter('phi')
        sum_ = theta + phi

        qc.rz(sum_, qr[0])
        dag = circuit_to_dag(qc)

        unrolled_dag = Unroller(['u1', 'u3', 'cx']).run(dag)

        expected = QuantumCircuit(qr)
        expected.u1(sum_, qr[0])

        self.assertEqual(circuit_to_dag(expected), unrolled_dag)

    def test_definition_unroll_parameterized(self):
        """Verify that unrolling complex gates with parameters does not raise."""
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)

        theta = Parameter('theta')

        qc.cu1(theta, qr[1], qr[0])
        qc.cu1(theta * theta, qr[0], qr[1])
        dag = circuit_to_dag(qc)

        out_dag = Unroller(['u1', 'cx']).run(dag)

        self.assertEqual(out_dag.count_ops(), {'u1': 6, 'cx': 4})

    def test_unrolling_parameterized_composite_gates(self):
        """Verify unrolling circuits with parameterized composite gates."""
        qr1 = QuantumRegister(2)
        subqc = QuantumCircuit(qr1)

        theta = Parameter('theta')

        subqc.rz(theta, qr1[0])
        subqc.cx(qr1[0], qr1[1])
        subqc.rz(theta, qr1[1])

        # Expanding across register with shared parameter
        qr2 = QuantumRegister(4)
        qc = QuantumCircuit(qr2)

        qc.append(subqc.to_instruction(), [qr2[0], qr2[1]])
        qc.append(subqc.to_instruction(), [qr2[2], qr2[3]])

        dag = circuit_to_dag(qc)
        out_dag = Unroller(['u1', 'u3', 'cx']).run(dag)

        expected = QuantumCircuit(qr2)
        expected.u1(theta, qr2[0])
        expected.cx(qr2[0], qr2[1])
        expected.u1(theta, qr2[1])

        expected.u1(theta, qr2[2])
        expected.cx(qr2[2], qr2[3])
        expected.u1(theta, qr2[3])
        self.assertEqual(circuit_to_dag(expected), out_dag)

        # Expanding across register with shared parameter
        qc = QuantumCircuit(qr2)

        phi = Parameter('phi')
        gamma = Parameter('gamma')

        qc.append(subqc.to_instruction({theta: phi}), [qr2[0], qr2[1]])
        qc.append(subqc.to_instruction({theta: gamma}), [qr2[2], qr2[3]])

        dag = circuit_to_dag(qc)
        out_dag = Unroller(['u1', 'u3', 'cx']).run(dag)

        expected = QuantumCircuit(qr2)
        expected.u1(phi, qr2[0])
        expected.cx(qr2[0], qr2[1])
        expected.u1(phi, qr2[1])

        expected.u1(gamma, qr2[2])
        expected.cx(qr2[2], qr2[3])
        expected.u1(gamma, qr2[3])

        self.assertEqual(circuit_to_dag(expected), out_dag)


class TestUnrollAllInstructions(QiskitTestCase):
    """Test unrolling a circuit containing all standard instructions."""

    def setUp(self):
        qr = self.qr = QuantumRegister(3, 'qr')
        cr = self.cr = ClassicalRegister(3, 'cr')
        self.circuit = QuantumCircuit(qr, cr)
        self.ref_circuit = QuantumCircuit(qr, cr)
        self.pass_ = Unroller(basis=['u3', 'cx', 'id'])

    def compare_dags(self):
        """compare dags in class tests"""
        dag = circuit_to_dag(self.circuit)
        unrolled_dag = self.pass_.run(dag)
        ref_dag = circuit_to_dag(self.ref_circuit)
        self.assertEqual(unrolled_dag, ref_dag)

    def test_unroll_crx(self):
        """test unroll crx"""
        self.circuit.crx(0.5, 1, 2)
        self.ref_circuit.u3(0, 0, pi/2, 2)
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.u3(-0.25, 0, 0, 2)
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.u3(0.25, -pi/2, 0, 2)
        self.compare_dags()

    def test_unroll_cry(self):
        """test unroll cry"""
        self.circuit.cry(0.5, 1, 2)
        self.ref_circuit.u3(0.25, 0, 0, 2)
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.u3(-0.25, 0, 0, 2)
        self.ref_circuit.cx(1, 2)
        self.compare_dags()

    def test_unroll_ccx(self):
        """test unroll ccx"""
        self.circuit.ccx(0, 1, 2)
        self.ref_circuit.u3(pi/2, 0, pi, 2)
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.u3(0, 0, -pi/4, 2)
        self.ref_circuit.cx(0, 2)
        self.ref_circuit.u3(0, 0, pi/4, 2)
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.u3(0, 0, pi/4, 1)
        self.ref_circuit.u3(0, 0, -pi/4, 2)
        self.ref_circuit.cx(0, 2)
        self.ref_circuit.cx(0, 1)
        self.ref_circuit.u3(0, 0, pi/4, 0)
        self.ref_circuit.u3(0, 0, -pi/4, 1)
        self.ref_circuit.cx(0, 1)
        self.ref_circuit.u3(0, 0, pi/4, 2)
        self.ref_circuit.u3(pi/2, 0, pi, 2)
        self.compare_dags()

    def test_unroll_ch(self):
        """test unroll ch"""
        self.circuit.ch(0, 2)
        self.ref_circuit.u3(0, 0, pi/2, 2)
        self.ref_circuit.u3(pi/2, 0, pi, 2)
        self.ref_circuit.u3(0, 0, pi/4, 2)
        self.ref_circuit.cx(0, 2)
        self.ref_circuit.u3(0, 0, -pi/4, 2)
        self.ref_circuit.u3(pi/2, 0, pi, 2)
        self.ref_circuit.u3(0, 0, -pi/2, 2)
        self.compare_dags()

    def test_unroll_crz(self):
        """test unroll crz"""
        self.circuit.crz(0.5, 1, 2)
        self.ref_circuit.u3(0, 0, 0.25, 2)
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.u3(0, 0, -0.25, 2)
        self.ref_circuit.cx(1, 2)

    def test_unroll_cswap(self):
        """test unroll cswap"""
        self.circuit.cswap(1, 0, 2)
        self.ref_circuit.cx(2, 0)
        self.ref_circuit.u3(pi/2, 0, pi, 2)
        self.ref_circuit.cx(0, 2)
        self.ref_circuit.u3(0, 0, -pi/4, 2)
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.u3(0, 0, pi/4, 2)
        self.ref_circuit.cx(0, 2)
        self.ref_circuit.u3(0, 0, pi/4, 0)
        self.ref_circuit.u3(0, 0, -pi/4, 2)
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.cx(1, 0)
        self.ref_circuit.u3(0, 0, -pi/4, 0)
        self.ref_circuit.u3(0, 0, pi/4, 1)
        self.ref_circuit.cx(1, 0)
        self.ref_circuit.u3(0, 0, pi/4, 2)
        self.ref_circuit.u3(pi/2, 0, pi, 2)
        self.ref_circuit.cx(2, 0)
        self.compare_dags()

    def test_unroll_cu1(self):
        """test unroll cu1"""
        self.circuit.cu1(0.1, 0, 2)
        self.ref_circuit.u3(0, 0, 0.05, 0)
        self.ref_circuit.cx(0, 2)
        self.ref_circuit.u3(0, 0, -0.05, 2)
        self.ref_circuit.cx(0, 2)
        self.ref_circuit.u3(0, 0, 0.05, 2)
        self.compare_dags()

    def test_unroll_cu3(self):
        """test unroll cu3"""
        self.circuit.cu3(0.2, 0.1, 0.0, 1, 2)
        self.ref_circuit.u3(0, 0, 0.05, 1)
        self.ref_circuit.u3(0, 0, -0.05, 2)
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.u3(-0.1, 0, -0.05, 2)
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.u3(0.1, 0.1, 0, 2)
        self.compare_dags()

    def test_unroll_cx(self):
        """test unroll cx"""
        self.circuit.cx(1, 0)
        self.ref_circuit.cx(1, 0)
        self.compare_dags()

    def test_unroll_cy(self):
        """test unroll cy"""
        self.circuit.cy(1, 2)
        self.ref_circuit.u3(0, 0, -pi/2, 2)
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.u3(0, 0, pi/2, 2)
        self.compare_dags()

    def test_unroll_cz(self):
        """test unroll cz"""
        self.circuit.cz(2, 0)
        self.ref_circuit.u3(pi/2, 0, pi, 0)
        self.ref_circuit.cx(2, 0)
        self.ref_circuit.u3(pi/2, 0, pi, 0)
        self.compare_dags()

    def test_unroll_h(self):
        """test unroll h"""
        self.circuit.h(1)
        self.ref_circuit.u3(pi/2, 0, pi, 1)
        self.compare_dags()

    def test_unroll_i(self):
        """test unroll i"""
        self.circuit.i(0)
        self.ref_circuit.i(0)
        self.compare_dags()

    def test_unroll_rx(self):
        """test unroll rx"""
        self.circuit.rx(0.1, 0)
        self.ref_circuit.u3(0.1, -pi/2, pi/2, 0)
        self.compare_dags()

    def test_unroll_ry(self):
        """test unroll ry"""
        self.circuit.ry(0.2, 1)
        self.ref_circuit.u3(0.2, 0, 0, 1)
        self.compare_dags()

    def test_unroll_rz(self):
        """test unroll rz"""
        self.circuit.rz(0.3, 2)
        self.ref_circuit.u3(0, 0, 0.3, 2)
        self.compare_dags()

    def test_unroll_rzz(self):
        """test unroll rzz"""
        self.circuit.rzz(0.6, 1, 0)
        self.ref_circuit.cx(1, 0)
        self.ref_circuit.u3(0, 0, 0.6, 0)
        self.ref_circuit.cx(1, 0)
        self.compare_dags()

    def test_unroll_s(self):
        """test unroll s"""
        self.circuit.s(0)
        self.ref_circuit.u3(0, 0, pi/2, 0)
        self.compare_dags()

    def test_unroll_sdg(self):
        """test unroll sdg"""
        self.circuit.sdg(1)
        self.ref_circuit.u3(0, 0, -pi/2, 1)
        self.compare_dags()

    def test_unroll_swap(self):
        """test unroll swap"""
        self.circuit.swap(1, 2)
        self.ref_circuit.cx(1, 2)
        self.ref_circuit.cx(2, 1)
        self.ref_circuit.cx(1, 2)
        self.compare_dags()

    def test_unroll_t(self):
        """test unroll t"""
        self.circuit.t(2)
        self.ref_circuit.u3(0, 0, pi/4, 2)
        self.compare_dags()

    def test_unroll_tdg(self):
        """test unroll tdg"""
        self.circuit.tdg(0)
        self.ref_circuit.u3(0, 0, -pi/4, 0)
        self.compare_dags()

    def test_unroll_u1(self):
        """test unroll u1"""
        self.circuit.u1(0.1, 1)
        self.ref_circuit.u3(0, 0, 0.1, 1)
        self.compare_dags()

    def test_unroll_u2(self):
        """test unroll u2"""
        self.circuit.u2(0.2, -0.1, 0)
        self.ref_circuit.u3(pi/2, 0.2, -0.1, 0)
        self.compare_dags()

    def test_unroll_u3(self):
        """test unroll u3"""
        self.circuit.u3(0.3, 0.0, -0.1, 2)
        self.ref_circuit.u3(0.3, 0.0, -0.1, 2)
        self.compare_dags()

    def test_unroll_x(self):
        """test unroll x"""
        self.circuit.x(2)
        self.ref_circuit.u3(pi, 0, pi, 2)
        self.compare_dags()

    def test_unroll_y(self):
        """test unroll y"""
        self.circuit.y(1)
        self.ref_circuit.u3(pi, pi/2, pi/2, 1)
        self.compare_dags()

    def test_unroll_z(self):
        """test unroll z"""
        self.circuit.z(0)
        self.ref_circuit.u3(0, 0, pi, 0)
        self.compare_dags()

    def test_unroll_snapshot(self):
        """test unroll snapshot"""
        self.circuit.snapshot('0')
        self.ref_circuit.snapshot('0')
        self.compare_dags()

    def test_unroll_measure(self):
        """test unroll measure"""
        self.circuit.measure(self.qr, self.cr)
        self.ref_circuit.measure(self.qr, self.cr)
        self.compare_dags()
