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
from qiskit.circuit import Parameter
from qiskit.qasm.exceptions import QasmError


class TestCircuitQasm(QiskitTestCase):
    """QuantumCircuit Qasm tests."""

    def test_circuit_qasm(self):
        """Test circuit qasm() method."""
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        cr = ClassicalRegister(3, "cr")
        qc = QuantumCircuit(qr1, qr2, cr)
        qc.p(0.3, qr1[0])
        qc.u(0.3, 0.2, 0.1, qr2[1])
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
p(0.3) qr1[0];
u(0.3,0.2,0.1) qr2[1];
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

        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.append(composite_circ_instr, [0, 1])
        qc.measure([0, 1], [0, 1])

        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
gate composite_circ q0,q1 { h q0; x q1; cx q0,q1; }
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

        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.append(composite_circ_instr, [0, 1])
        qc.append(composite_circ_instr, [0, 1])
        qc.measure([0, 1], [0, 1])

        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
gate composite_circ q0,q1 { h q0; x q1; cx q0,q1; }
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

        my_gate = QuantumCircuit(1, name="my_gate")
        my_gate.h(0)
        my_gate_inst1 = my_gate.to_instruction()

        my_gate = QuantumCircuit(1, name="my_gate")
        my_gate.x(0)
        my_gate_inst2 = my_gate.to_instruction()

        my_gate = QuantumCircuit(1, name="my_gate")
        my_gate.x(0)
        my_gate_inst3 = my_gate.to_instruction()

        qr = QuantumRegister(1, name="qr")
        circuit = QuantumCircuit(qr, name="circuit")
        circuit.append(my_gate_inst1, [qr[0]])
        circuit.append(my_gate_inst2, [qr[0]])
        my_gate_inst2_id = id(circuit.data[-1][0])
        circuit.append(my_gate_inst3, [qr[0]])
        my_gate_inst3_id = id(circuit.data[-1][0])

        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
gate my_gate q0 {{ h q0; }}
gate my_gate_{1} q0 {{ x q0; }}
gate my_gate_{0} q0 {{ x q0; }}
qreg qr[1];
my_gate qr[0];
my_gate_{1} qr[0];
my_gate_{0} qr[0];\n""".format(
            my_gate_inst3_id, my_gate_inst2_id
        )
        self.assertEqual(circuit.qasm(), expected_qasm)

    def test_circuit_qasm_with_composite_circuit_with_children_composite_circuit(self):
        """Test circuit qasm() method when composite circuits with children
        composite circuits in the definitions are added to the circuit"""

        child_circ = QuantumCircuit(2, name="child_circ")
        child_circ.h(0)
        child_circ.cx(0, 1)

        parent_circ = QuantumCircuit(3, name="parent_circ")
        parent_circ.append(child_circ, range(2))
        parent_circ.h(2)

        grandparent_circ = QuantumCircuit(4, name="grandparent_circ")
        grandparent_circ.append(parent_circ, range(3))
        grandparent_circ.x(3)

        qc = QuantumCircuit(4)
        qc.append(grandparent_circ, range(4))

        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
gate child_circ q0,q1 { h q0; cx q0,q1; }
gate parent_circ q0,q1,q2 { child_circ q0,q1; h q2; }
gate grandparent_circ q0,q1,q2,q3 { parent_circ q0,q1,q2; x q3; }
qreg q[4];
grandparent_circ q[0],q[1],q[2],q[3];\n"""

        self.assertEqual(qc.qasm(), expected_qasm)

    def test_circuit_qasm_pi(self):
        """Test circuit qasm() method with pi params."""
        circuit = QuantumCircuit(2)
        circuit.cz(0, 1)
        circuit.u(2 * pi, 3 * pi, -5 * pi, 0)
        qasm_str = circuit.qasm()
        circuit2 = QuantumCircuit.from_qasm_str(qasm_str)
        self.assertEqual(circuit, circuit2)

    def test_circuit_qasm_with_composite_circuit_with_one_param(self):
        """Test circuit qasm() method when a composite circuit instruction
        has one param
        """
        original_str = """OPENQASM 2.0;
include "qelib1.inc";
gate nG0(param0) q0 { h q0; }
qreg q[3];
creg c[3];
nG0(pi) q[0];\n"""
        qc = QuantumCircuit.from_qasm_str(original_str)

        self.assertEqual(original_str, qc.qasm())

    def test_circuit_qasm_with_composite_circuit_with_many_params_and_qubits(self):
        """Test circuit qasm() method when a composite circuit instruction
        has many params and qubits
        """
        original_str = """OPENQASM 2.0;
include "qelib1.inc";
gate nG0(param0,param1) q0,q1 { h q0; h q1; }
qreg q[3];
qreg r[3];
creg c[3];
creg d[3];
nG0(pi,pi/2) q[0],r[0];\n"""
        qc = QuantumCircuit.from_qasm_str(original_str)

        self.assertEqual(original_str, qc.qasm())

    def test_unbound_circuit_raises(self):
        """Test circuits with unbound parameters raises."""
        qc = QuantumCircuit(1)
        theta = Parameter("θ")
        qc.rz(theta, 0)
        with self.assertRaises(QasmError):
            qc.qasm()

    def test_gate_qasm_with_ctrl_state(self):
        """Test gate qasm() with controlled gate that has ctrl_state setting."""
        from qiskit.quantum_info import Operator

        qc = QuantumCircuit(2)
        qc.ch(0, 1, ctrl_state=0)
        qasm_str = qc.qasm()
        self.assertEqual(Operator(qc), Operator(QuantumCircuit.from_qasm_str(qasm_str)))

    def test_circuit_qasm_with_mcx_gate(self):
        """Test circuit qasm() method with MCXGate
        See https://github.com/Qiskit/qiskit-terra/issues/4943
        """
        qc = QuantumCircuit(4)
        qc.mcx([0, 1, 2], 3)

        # qasm output doesn't support parameterized gate yet.
        # param0 for "gate mcuq(param0) is not used inside the definition
        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
gate mcx q0,q1,q2,q3 { h q3; p(pi/8) q0; p(pi/8) q1; p(pi/8) q2; p(pi/8) q3; cx q0,q1; p(-pi/8) q1; cx q0,q1; cx q1,q2; p(-pi/8) q2; cx q0,q2; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; h q3; }
qreg q[4];
mcx q[0],q[1],q[2],q[3];\n"""
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_circuit_qasm_with_mcx_gate_variants(self):
        """Test circuit qasm() method with MCXGrayCode, MCXRecursive, MCXVChain"""
        import qiskit.circuit.library as cl

        n = 5
        qc = QuantumCircuit(2 * n - 1)
        qc.append(cl.MCXGrayCode(n), range(n + 1))
        qc.append(cl.MCXRecursive(n), range(n + 2))
        qc.append(cl.MCXVChain(n), range(2 * n - 1))

        # qasm output doesn't support parameterized gate yet.
        # param0 for "gate mcuq(param0) is not used inside the definition
        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
gate mcx q0,q1,q2,q3 { h q3; p(pi/8) q0; p(pi/8) q1; p(pi/8) q2; p(pi/8) q3; cx q0,q1; p(-pi/8) q1; cx q0,q1; cx q1,q2; p(-pi/8) q2; cx q0,q2; p(pi/8) q2; cx q1,q2; p(-pi/8) q2; cx q0,q2; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q1,q3; p(pi/8) q3; cx q2,q3; p(-pi/8) q3; cx q0,q3; h q3; }
gate mcu1(param0) q0,q1,q2,q3,q4,q5 { cu1(pi/16) q4,q5; cx q4,q3; cu1(-pi/16) q3,q5; cx q4,q3; cu1(pi/16) q3,q5; cx q3,q2; cu1(-pi/16) q2,q5; cx q4,q2; cu1(pi/16) q2,q5; cx q3,q2; cu1(-pi/16) q2,q5; cx q4,q2; cu1(pi/16) q2,q5; cx q2,q1; cu1(-pi/16) q1,q5; cx q4,q1; cu1(pi/16) q1,q5; cx q3,q1; cu1(-pi/16) q1,q5; cx q4,q1; cu1(pi/16) q1,q5; cx q2,q1; cu1(-pi/16) q1,q5; cx q4,q1; cu1(pi/16) q1,q5; cx q3,q1; cu1(-pi/16) q1,q5; cx q4,q1; cu1(pi/16) q1,q5; cx q1,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q3,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q2,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q3,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q1,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q3,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q2,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; cx q3,q0; cu1(-pi/16) q0,q5; cx q4,q0; cu1(pi/16) q0,q5; }
gate mcx_gray q0,q1,q2,q3,q4,q5 { h q5; mcu1(pi) q0,q1,q2,q3,q4,q5; h q5; }
gate mcx_recursive q0,q1,q2,q3,q4,q5,q6 { mcx q0,q1,q2,q6; mcx q3,q4,q6,q5; mcx q0,q1,q2,q6; mcx q3,q4,q6,q5; }
gate mcx_vchain q0,q1,q2,q3,q4,q5,q6,q7,q8 { rccx q0,q1,q6; rccx q2,q6,q7; rccx q3,q7,q8; ccx q4,q8,q5; rccx q3,q7,q8; rccx q2,q6,q7; rccx q0,q1,q6; }
qreg q[9];
mcx_gray q[0],q[1],q[2],q[3],q[4],q[5];
mcx_recursive q[0],q[1],q[2],q[3],q[4],q[5],q[6];
mcx_vchain q[0],q[1],q[2],q[3],q[4],q[5],q[6],q[7],q[8];\n"""

        self.assertEqual(qc.qasm(), expected_qasm)
