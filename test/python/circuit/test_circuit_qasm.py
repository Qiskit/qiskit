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
import re

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.circuit import Parameter, Qubit, Clbit
from qiskit.qasm.exceptions import QasmError

# Regex pattern to match valid OpenQASM identifiers
VALID_QASM2_IDENTIFIER = re.compile("[a-z][a-zA-Z_0-9]*")


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
        my_gate_inst2_id = id(circuit.data[-1].operation)
        circuit.append(my_gate_inst3, [qr[0]])
        my_gate_inst3_id = id(circuit.data[-1].operation)

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

    def test_circuit_qasm_with_registerless_bits(self):
        """Test that registerless bits do not have naming collisions in their registers."""
        initial_registers = [QuantumRegister(2), ClassicalRegister(2)]
        qc = QuantumCircuit(*initial_registers, [Qubit(), Clbit()])
        # Match a 'qreg identifier[3];'-like QASM register declaration.
        register_regex = re.compile(r"\s*[cq]reg\s+(\w+)\s*\[\d+\]\s*", re.M)
        qasm_register_names = set()
        for statement in qc.qasm().split(";"):
            match = register_regex.match(statement)
            if match:
                qasm_register_names.add(match.group(1))
        self.assertEqual(len(qasm_register_names), 4)

        # Check that no additional registers were added to the circuit.
        self.assertEqual(len(qc.qregs), 1)
        self.assertEqual(len(qc.cregs), 1)

        # Check that the registerless-register names are recalculated after adding more registers,
        # to avoid naming clashes in this case.
        generated_names = qasm_register_names - {register.name for register in initial_registers}
        for generated_name in generated_names:
            qc.add_register(QuantumRegister(1, name=generated_name))
        qasm_register_names = set()
        for statement in qc.qasm().split(";"):
            match = register_regex.match(statement)
            if match:
                qasm_register_names.add(match.group(1))
        self.assertEqual(len(qasm_register_names), 6)

    def test_circuit_qasm_with_repeated_instruction_names(self):
        """Test that qasm() doesn't change the name of the instructions that live in circuit.data,
        but a copy of them when there are repeated names."""
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.x(1)
        # Create some random custom gate and name it "custom"
        custom = QuantumCircuit(1)
        custom.h(0)
        custom.y(0)
        gate = custom.to_gate()
        gate.name = "custom"
        # Another random custom gate named "custom" as well
        custom2 = QuantumCircuit(2)
        custom2.x(0)
        custom2.z(1)
        gate2 = custom2.to_gate()
        gate2.name = "custom"
        # Append custom gates with same name to original circuit
        qc.append(gate, [0])
        qc.append(gate2, [1, 0])
        # Expected qasm string will append the id to the second gate with repeated name
        expected_qasm = f"""OPENQASM 2.0;
include "qelib1.inc";
gate custom q0 {{ h q0; y q0; }}
gate custom_{id(gate2)} q0,q1 {{ x q0; z q1; }}
qreg q[2];
h q[0];
x q[1];
custom q[0];
custom_{id(gate2)} q[1],q[0];\n"""
        # Check qasm() produced the correct string
        self.assertEqual(expected_qasm, qc.qasm())
        # Check instruction names were not changed by qasm()
        names = ["h", "x", "custom", "custom"]
        for idx, instruction in enumerate(qc._data):
            self.assertEqual(instruction.operation.name, names[idx])

    def test_circuit_qasm_with_invalid_identifiers(self):
        """Test that qasm() detects and corrects invalid OpenQASM gate identifiers,
        while not changing the instructions on the original circuit"""
        qc = QuantumCircuit(2)

        # Create some gate and give it an invalid name
        custom = QuantumCircuit(1)
        custom.x(0)
        custom.u(0, 0, pi, 0)
        gate = custom.to_gate()
        gate.name = "A[$]"

        # Another gate also with invalid name
        custom2 = QuantumCircuit(2)
        custom2.x(0)
        custom2.append(gate, [1])
        gate2 = custom2.to_gate()
        gate2.name = "invalid[name]"

        # Unitary gate, for which qasm string is produced by internal method
        qc.unitary([[0, 1], [1, 0]], 0, label="[valid?]")

        # Append gates
        qc.append(gate, [0])
        qc.append(gate2, [1, 0])

        # Expected qasm with valid identifiers
        expected_qasm = "\n".join(
            [
                "OPENQASM 2.0;",
                'include "qelib1.inc";',
                "gate gate__valid__ p0 {",
                "	u3(pi,-pi/2,pi/2) p0;",
                "}",
                "gate gate_A___ q0 { x q0; u(0,0,pi) q0; }",
                "gate invalid_name_ q0,q1 { x q0; gate_A___ q1; }",
                "qreg q[2];",
                "gate__valid__ q[0];",
                "gate_A___ q[0];",
                "invalid_name_ q[1],q[0];",
                "",
            ]
        )

        # Check qasm() produces the correct string
        self.assertEqual(expected_qasm, qc.qasm())

        # Check instruction names were not changed by qasm()
        names = ["unitary", "A[$]", "invalid[name]"]
        for idx, instruction in enumerate(qc._data):
            self.assertEqual(instruction.operation.name, names[idx])

    def test_circuit_qasm_with_duplicate_invalid_identifiers(self):
        """Test that qasm() corrects invalid identifiers and the de-duplication
        code runs correctly, without altering original instructions"""
        base = QuantumCircuit(1)

        # First gate with invalid name, escapes to "invalid__"
        clash1 = QuantumCircuit(1, name="invalid??")
        clash1.x(0)
        base.append(clash1, [0])

        # Second gate with invalid name that also escapes to "invalid__"
        clash2 = QuantumCircuit(1, name="invalid[]")
        clash2.z(0)
        base.append(clash2, [0])

        # Check qasm is correctly produced
        names = set()
        for match in re.findall(r"gate (\S+)", base.qasm()):
            self.assertTrue(VALID_QASM2_IDENTIFIER.fullmatch(match))
            names.add(match)
        self.assertEqual(len(names), 2)

        # Check instruction names were not changed by qasm()
        names = ["invalid??", "invalid[]"]
        for idx, instruction in enumerate(base._data):
            self.assertEqual(instruction.operation.name, names[idx])

    def test_circuit_qasm_with_double_precision_rotation_angle(self):
        """Test that qasm() emits high precision rotation angles per default."""
        from qiskit.circuit.tools.pi_check import MAX_FRAC

        qc = QuantumCircuit(1)
        qc.p(0.123456789, 0)
        qc.p(pi * pi, 0)
        qc.p(MAX_FRAC * pi + 1, 0)

        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
p(0.123456789) q[0];
p(9.869604401089358) q[0];
p(51.26548245743669) q[0];\n"""
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_circuit_qasm_with_rotation_angles_close_to_pi(self):
        """Test that qasm() properly rounds values closer than 1e-12 to pi."""

        qc = QuantumCircuit(1)
        qc.p(pi + 1e-11, 0)
        qc.p(pi + 1e-12, 0)
        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
p(3.141592653599793) q[0];
p(pi) q[0];\n"""
        self.assertEqual(qc.qasm(), expected_qasm)

    def test_circuit_raises_on_single_bit_condition(self):
        """OpenQASM 2 can't represent single-bit conditions, so test that a suitable error is
        printed if this is attempted."""
        qc = QuantumCircuit(1, 1)
        qc.x(0).c_if(0, True)

        with self.assertRaisesRegex(QasmError, "OpenQASM 2 can only condition on registers"):
            qc.qasm()
