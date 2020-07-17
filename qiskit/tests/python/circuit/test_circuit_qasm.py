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

"""Test Qiskit's QuantumCircuit class."""

from math import pi

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.quantum_info import random_unitary


class TestCircuitQasm(QiskitTestCase):
    """QuantumCircuit Qasm tests."""

    def test_circuit_qasm(self):
        """Test circuit qasm() method.
        """
        qr1 = QuantumRegister(1, 'qr1')
        qr2 = QuantumRegister(2, 'qr2')
        cr = ClassicalRegister(3, 'cr')
        qc = QuantumCircuit(qr1, qr2, cr)
        qc.u1(0.3, qr1[0])
        qc.u2(0.2, 0.1, qr2[0])
        qc.u3(0.3, 0.2, 0.1, qr2[1])
        qc.s(qr2[1])
        qc.sdg(qr2[1])
        qc.cx(qr1[0], qr2[1])
        qc.barrier(qr2)
        qc.cx(qr2[1], qr1[0])
        qc.h(qr2[1])
        qc.x(qr2[1]).c_if(cr, 0)
        qc.y(qr1[0]).c_if(cr, 1)
        qc.z(qr1[0]).c_if(cr, 2)
        qc.barrier(qr1, qr2)
        qc.measure(qr1[0], cr[0])
        qc.measure(qr2[0], cr[1])
        qc.measure(qr2[1], cr[2])
        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg qr1[1];
qreg qr2[2];
creg cr[3];
u1(0.3) qr1[0];
u2(0.2,0.1) qr2[0];
u3(0.3,0.2,0.1) qr2[1];
s qr2[1];
sdg qr2[1];
cx qr1[0],qr2[1];
barrier qr2[0],qr2[1];
cx qr2[1],qr1[0];
h qr2[1];
if(cr==0) x qr2[1];
if(cr==1) y qr1[0];
if(cr==2) z qr1[0];
barrier qr1[0],qr2[0],qr2[1];
measure qr1[0] -> cr[0];
measure qr2[0] -> cr[1];
measure qr2[1] -> cr[2];\n"""
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_circuit_qasm_with_composite_circuit(self):
        """Test circuit qasm() method when a composite circuit instruction
        is included within circuit.
        """

        composite_circ_qreg = QuantumRegister(2)
        composite_circ = QuantumCircuit(composite_circ_qreg, name="composite_circ")
        composite_circ.h(0)
        composite_circ.x(1)
        composite_circ.cx(0, 1)
        composite_circ_instr = composite_circ.to_instruction()

        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.append(composite_circ_instr, [0, 1])
        qc.measure([0, 1], [0, 1])

        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
gate composite_circ q0,q1 {h q0; x q1; cx q0,q1; }
qreg qr[2];
creg cr[2];
h qr[0];
cx qr[0],qr[1];
barrier qr[0],qr[1];
composite_circ qr[0],qr[1];
measure qr[0] -> cr[0];
measure qr[1] -> cr[1];\n"""
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_circuit_qasm_with_multiple_same_composite_circuits(self):
        """Test circuit qasm() method when a composite circuit is added
        to the circuit multiple times
        """

        composite_circ_qreg = QuantumRegister(2)
        composite_circ = QuantumCircuit(composite_circ_qreg, name="composite_circ")
        composite_circ.h(0)
        composite_circ.x(1)
        composite_circ.cx(0, 1)
        composite_circ_instr = composite_circ.to_instruction()

        qr = QuantumRegister(2, 'qr')
        cr = ClassicalRegister(2, 'cr')
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.append(composite_circ_instr, [0, 1])
        qc.append(composite_circ_instr, [0, 1])
        qc.measure([0, 1], [0, 1])

        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
gate composite_circ q0,q1 {h q0; x q1; cx q0,q1; }
qreg qr[2];
creg cr[2];
h qr[0];
cx qr[0],qr[1];
barrier qr[0],qr[1];
composite_circ qr[0],qr[1];
composite_circ qr[0],qr[1];
measure qr[0] -> cr[0];
measure qr[1] -> cr[1];\n"""
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_circuit_qasm_with_multiple_composite_circuits_with_same_name(self):
        """Test circuit qasm() method when multiple composite circuit instructions
        with the same circuit name are added to the circuit
        """

        my_gate = QuantumCircuit(1, name='my_gate')
        my_gate.h(0)
        my_gate_inst1 = my_gate.to_instruction()

        my_gate = QuantumCircuit(1, name='my_gate')
        my_gate.x(0)
        my_gate_inst2 = my_gate.to_instruction()

        my_gate = QuantumCircuit(1, name='my_gate')
        my_gate.x(0)
        my_gate_inst3 = my_gate.to_instruction()

        qr = QuantumRegister(1, name='qr')
        circuit = QuantumCircuit(qr, name='circuit')
        circuit.append(my_gate_inst1, [qr[0]])
        circuit.append(my_gate_inst2, [qr[0]])
        my_gate_inst2_id = id(circuit.data[-1][0])
        circuit.append(my_gate_inst3, [qr[0]])
        my_gate_inst3_id = id(circuit.data[-1][0])

        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
gate my_gate_{0} q0 {{x q0; }}
gate my_gate_{1} q0 {{x q0; }}
gate my_gate q0 {{h q0; }}
qreg qr[1];
my_gate qr[0];
my_gate_{1} qr[0];
my_gate_{0} qr[0];\n""".format(my_gate_inst3_id, my_gate_inst2_id)
        self.assertEqual(circuit.qasm(), expected_qasm)

    def test_circuit_qasm_pi(self):
        """Test circuit qasm() method with pi params.
        """
        circuit = QuantumCircuit(2)
        circuit.append(random_unitary(4, seed=1234), [0, 1])
        circuit = circuit.decompose()
        circuit.u3(2*pi, 3*pi, -5*pi, 0)
        qasm_str = circuit.qasm()
        circuit2 = QuantumCircuit.from_qasm_str(qasm_str)
        self.assertEqual(circuit, circuit2)
