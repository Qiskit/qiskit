# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test QASM3 exporter."""

# We can't really help how long the lines output by the exporter are in some cases.
# pylint: disable=line-too-long

from io import StringIO
import re
import unittest

from ddt import ddt, data

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit import Parameter, Qubit, Clbit, Instruction
from qiskit.test import QiskitTestCase
from qiskit.qasm3 import Exporter, dumps, dump, QASM3ExporterError
from qiskit.qasm3.exporter import QASM3Builder
from qiskit.qasm3.printer import BasicPrinter
from qiskit.qasm import pi


# Tests marked with this decorator should be restored after gate definition with parameters is fixed
# properly, and the dummy tests after them should be deleted.  See gh-7335.
requires_fixed_parameterisation = unittest.expectedFailure


class TestQASM3Functions(QiskitTestCase):
    """QASM3 module - high level functions"""

    def setUp(self):
        self.circuit = QuantumCircuit(2)
        self.circuit.u(2 * pi, 3 * pi, -5 * pi, 0)
        self.expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "qubit[2] _all_qubits;",
                "let q = _all_qubits[0:1];",
                "U(2*pi, 3*pi, -5*pi) q[0];",
                "",
            ]
        )
        super().setUp()

    def test_dumps(self):
        """Test dumps."""
        result = dumps(self.circuit)
        self.assertEqual(result, self.expected_qasm)

    def test_dump(self):
        """Test dump into an IO stream."""
        io = StringIO()
        dump(self.circuit, io)
        result = io.getvalue()
        self.assertEqual(result, self.expected_qasm)


@ddt
class TestCircuitQASM3(QiskitTestCase):
    """QASM3 exporter."""

    maxDiff = 1_000_000

    @classmethod
    def setUpClass(cls):
        # These regexes are not perfect by any means, but sufficient for simple tests on controlled
        # input circuits.
        cls.register_regex = re.compile(r"^\s*let\s+(?P<name>\w+\b)", re.U | re.M)
        scalar_type_names = {
            "angle",
            "duration",
            "float",
            "int",
            "stretch",
            "uint",
        }
        cls.scalar_parameter_regex = re.compile(
            r"^\s*((input|output|const)\s+)?"  # Modifier
            rf"({'|'.join(scalar_type_names)})\s*(\[[^\]]+\])?\s+"  # Type name and designator
            r"(?P<name>\w+\b)",  # Parameter name
            re.U | re.M,
        )
        super().setUpClass()

    def test_regs_conds_qasm(self):
        """Test with registers and conditionals."""
        qr1 = QuantumRegister(1, "qr1")
        qr2 = QuantumRegister(2, "qr2")
        cr = ClassicalRegister(3, "cr")
        qc = QuantumCircuit(qr1, qr2, cr)
        qc.measure(qr1[0], cr[0])
        qc.measure(qr2[0], cr[1])
        qc.measure(qr2[1], cr[2])
        qc.x(qr2[1]).c_if(cr, 0)
        qc.y(qr1[0]).c_if(cr, 1)
        qc.z(qr1[0]).c_if(cr, 2)
        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "bit[3] cr;",
                "qubit[3] _all_qubits;",
                "let qr1 = _all_qubits[0:0];",
                "let qr2 = _all_qubits[1:2];",
                "cr[0] = measure qr1[0];",
                "cr[1] = measure qr2[0];",
                "cr[2] = measure qr2[1];",
                "if (cr == 0) {",
                "  x qr2[1];",
                "}",
                "if (cr == 1) {",
                "  y qr1[0];",
                "}",
                "if (cr == 2) {",
                "  z qr1[0];",
                "}",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(qc), expected_qasm)

    def test_registers_as_aliases(self):
        """Test that different types of alias creation and concatenation work."""
        qubits = [Qubit() for _ in [None] * 10]
        first_four = QuantumRegister(name="first_four", bits=qubits[:4])
        last_five = QuantumRegister(name="last_five", bits=qubits[5:])
        alternate = QuantumRegister(name="alternate", bits=qubits[::2])
        sporadic = QuantumRegister(name="sporadic", bits=[qubits[4], qubits[2], qubits[9]])
        qc = QuantumCircuit(qubits, first_four, last_five, alternate, sporadic)
        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "qubit[10] _all_qubits;",
                "let first_four = _all_qubits[0:3];",
                "let last_five = _all_qubits[5:9];",
                # The exporter does not attempt to output steps.
                "let alternate = _all_qubits[0:0] ++ _all_qubits[2:2] ++ _all_qubits[4:4] ++ _all_qubits[6:6] ++ _all_qubits[8:8];",
                "let sporadic = _all_qubits[4:4] ++ _all_qubits[2:2] ++ _all_qubits[9:9];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(qc), expected_qasm)

    def test_composite_circuit(self):
        """Test with a composite circuit instruction and barriers"""
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

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "def composite_circ(qubit _gate_q_0, qubit _gate_q_1) {",
                "  h _gate_q_0;",
                "  x _gate_q_1;",
                "  cx _gate_q_0, _gate_q_1;",
                "}",
                "bit[2] cr;",
                "qubit[2] _all_qubits;",
                "let qr = _all_qubits[0:1];",
                "h qr[0];",
                "cx qr[0], qr[1];",
                "barrier qr[0], qr[1];",
                "composite_circ qr[0], qr[1];",
                "cr[0] = measure qr[0];",
                "cr[1] = measure qr[1];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(qc), expected_qasm)

    def test_custom_gate(self):
        """Test custom gates (via to_gate)."""
        composite_circ_qreg = QuantumRegister(2)
        composite_circ = QuantumCircuit(composite_circ_qreg, name="composite_circ")
        composite_circ.h(0)
        composite_circ.x(1)
        composite_circ.cx(0, 1)
        composite_circ_instr = composite_circ.to_gate()

        qr = QuantumRegister(2, "qr")
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(qr, cr)
        qc.h(0)
        qc.cx(0, 1)
        qc.barrier()
        qc.append(composite_circ_instr, [0, 1])
        qc.measure([0, 1], [0, 1])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "gate composite_circ _gate_q_0, _gate_q_1 {",
                "  h _gate_q_0;",
                "  x _gate_q_1;",
                "  cx _gate_q_0, _gate_q_1;",
                "}",
                "bit[2] cr;",
                "qubit[2] _all_qubits;",
                "let qr = _all_qubits[0:1];",
                "h qr[0];",
                "cx qr[0], qr[1];",
                "barrier qr[0], qr[1];",
                "composite_circ qr[0], qr[1];",
                "cr[0] = measure qr[0];",
                "cr[1] = measure qr[1];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(qc), expected_qasm)

    def test_same_composite_circuits(self):
        """Test when a composite circuit is added to the circuit multiple times."""
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

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "def composite_circ(qubit _gate_q_0, qubit _gate_q_1) {",
                "  h _gate_q_0;",
                "  x _gate_q_1;",
                "  cx _gate_q_0, _gate_q_1;",
                "}",
                "bit[2] cr;",
                "qubit[2] _all_qubits;",
                "let qr = _all_qubits[0:1];",
                "h qr[0];",
                "cx qr[0], qr[1];",
                "barrier qr[0], qr[1];",
                "composite_circ qr[0], qr[1];",
                "composite_circ qr[0], qr[1];",
                "cr[0] = measure qr[0];",
                "cr[1] = measure qr[1];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(qc), expected_qasm)

    def test_composite_circuits_with_same_name(self):
        """Test when multiple composite circuit instructions same name and different
        implementation."""
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
        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "def my_gate(qubit _gate_q_0) {",
                "  h _gate_q_0;",
                "}",
                f"def my_gate_{my_gate_inst2_id}(qubit _gate_q_0) {{",
                "  x _gate_q_0;",
                "}",
                f"def my_gate_{my_gate_inst3_id}(qubit _gate_q_0) {{",
                "  x _gate_q_0;",
                "}",
                "qubit[1] _all_qubits;",
                "let qr = _all_qubits[0:0];",
                "my_gate qr[0];",
                f"my_gate_{my_gate_inst2_id} qr[0];",
                f"my_gate_{my_gate_inst3_id} qr[0];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(circuit), expected_qasm)

    def test_pi_disable_constants_false(self):
        """Test pi constant (disable_constants=False)"""
        circuit = QuantumCircuit(2)
        circuit.u(2 * pi, 3 * pi, -5 * pi, 0)
        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "qubit[2] _all_qubits;",
                "let q = _all_qubits[0:1];",
                "U(2*pi, 3*pi, -5*pi) q[0];",
                "",
            ]
        )
        self.assertEqual(Exporter(disable_constants=False).dumps(circuit), expected_qasm)

    def test_pi_disable_constants_true(self):
        """Test pi constant (disable_constants=True)"""
        circuit = QuantumCircuit(2)
        circuit.u(2 * pi, 3 * pi, -5 * pi, 0)
        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "qubit[2] _all_qubits;",
                "let q = _all_qubits[0:1];",
                "U(6.283185307179586, 9.42477796076938, -15.707963267948966) q[0];",
                "",
            ]
        )
        self.assertEqual(Exporter(disable_constants=True).dumps(circuit), expected_qasm)

    def test_custom_gate_with_unbound_parameter(self):
        """Test custom gate with unbound parameter."""
        parameter_a = Parameter("a")

        custom = QuantumCircuit(1, name="custom")
        custom.rx(parameter_a, 0)
        circuit = QuantumCircuit(1)
        circuit.append(custom.to_gate(), [0])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "gate custom(a) _gate_q_0 {",
                "  rx(a) _gate_q_0;",
                "}",
                "input float[64] a;",
                "qubit[1] _all_qubits;",
                "let q = _all_qubits[0:0];",
                "custom(a) q[0];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(circuit), expected_qasm)

    def test_custom_gate_with_bound_parameter(self):
        """Test custom gate with bound parameter."""
        parameter_a = Parameter("a")

        custom = QuantumCircuit(1)
        custom.rx(parameter_a, 0)
        custom_gate = custom.bind_parameters({parameter_a: 0.5}).to_gate()
        custom_gate.name = "custom"

        circuit = QuantumCircuit(1)
        circuit.append(custom_gate, [0])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "gate custom _gate_q_0 {",
                "  rx(0.5) _gate_q_0;",
                "}",
                "qubit[1] _all_qubits;",
                "let q = _all_qubits[0:0];",
                "custom q[0];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(circuit), expected_qasm)

    @requires_fixed_parameterisation
    def test_custom_gate_with_params_bound_main_call(self):
        """Custom gate with unbound parameters that are bound in the main circuit"""
        parameter0 = Parameter("p0")
        parameter1 = Parameter("p1")

        custom = QuantumCircuit(2, name="custom")
        custom.rz(parameter0, 0)
        custom.rz(parameter1 / 2, 1)

        qr_all_qubits = QuantumRegister(3, "q")
        qr_r = QuantumRegister(3, "r")
        circuit = QuantumCircuit(qr_all_qubits, qr_r)
        circuit.append(custom.to_gate(), [qr_all_qubits[0], qr_r[0]])

        circuit.assign_parameters({parameter0: pi, parameter1: pi / 2}, inplace=True)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "gate custom(p0, p1) _gate_q_0, _gate_q_1 {",
                "  rz(pi) _gate_q_0;",
                "  rz(pi/4) _gate_q_1;",
                "}",
                "qubit[6] _all_qubits;",
                "let q = _all_qubits[0:2];",
                "let r = _all_qubits[3:5];",
                "custom(pi, pi/2) q[0], r[0];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(circuit), expected_qasm)

    def test_reused_custom_parameter(self):
        """Test reused custom gate with parameter."""
        parameter_a = Parameter("a")

        custom = QuantumCircuit(1)
        custom.rx(parameter_a, 0)

        circuit = QuantumCircuit(1)
        circuit.append(custom.bind_parameters({parameter_a: 0.5}).to_gate(), [0])
        circuit.append(custom.bind_parameters({parameter_a: 1}).to_gate(), [0])

        circuit_name_0 = circuit.data[0][0].definition.name
        circuit_name_1 = circuit.data[1][0].definition.name

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                f"gate {circuit_name_0} _gate_q_0 {{",
                "  rx(0.5) _gate_q_0;",
                "}",
                f"gate {circuit_name_1} _gate_q_0 {{",
                "  rx(1) _gate_q_0;",
                "}",
                "qubit[1] _all_qubits;",
                "let q = _all_qubits[0:0];",
                f"{circuit_name_0} q[0];",
                f"{circuit_name_1} q[0];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(circuit), expected_qasm)

    def test_unbound_circuit(self):
        """Test with unbound parameters (turning them into inputs)."""
        qc = QuantumCircuit(1)
        theta = Parameter("θ")
        qc.rz(theta, 0)
        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "input float[64] θ;",
                "qubit[1] _all_qubits;",
                "let q = _all_qubits[0:0];",
                "rz(θ) q[0];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(qc), expected_qasm)

    def test_unknown_parameterized_gate_called_multiple_times(self):
        """Test that a parameterised gate is called correctly if the first instance of it is
        generic."""
        x, y = Parameter("x"), Parameter("y")
        qc = QuantumCircuit(2)
        qc.rzx(x, 0, 1)
        qc.rzx(y, 0, 1)
        qc.rzx(0.5, 0, 1)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                "gate rzx(x) _gate_q_0, _gate_q_1 {",
                "  h _gate_q_1;",
                "  cx _gate_q_0, _gate_q_1;",
                "  rz(x) _gate_q_1;",
                "  cx _gate_q_0, _gate_q_1;",
                "  h _gate_q_1;",
                "}",
                "input float[64] x;",
                "input float[64] y;",
                "qubit[2] _all_qubits;",
                "let q = _all_qubits[0:1];",
                "rzx(x) q[0], q[1];",
                "rzx(y) q[0], q[1];",
                "rzx(0.5) q[0], q[1];",
                "",
            ]
        )

        # Set the includes and basis gates to ensure that this gate is unknown.
        exporter = Exporter(includes=[], basis_gates=("rz", "h", "cx"))
        self.assertEqual(exporter.dumps(qc), expected_qasm)

    def test_gate_qasm_with_ctrl_state(self):
        """Test with open controlled gate that has ctrl_state"""
        qc = QuantumCircuit(2)
        qc.ch(0, 1, ctrl_state=0)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "gate ch_o0 _gate_q_0, _gate_q_1 {",
                "  x _gate_q_0;",
                "  ch _gate_q_0, _gate_q_1;",
                "  x _gate_q_0;",
                "}",
                "qubit[2] _all_qubits;",
                "let q = _all_qubits[0:1];",
                "ch_o0 q[0], q[1];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(qc), expected_qasm)

    def test_custom_gate_collision_with_stdlib(self):
        """Test a custom gate with name collision with the standard library."""
        custom = QuantumCircuit(2, name="cx")
        custom.cx(0, 1)
        custom_gate = custom.to_gate()

        qc = QuantumCircuit(2)
        qc.append(custom_gate, [0, 1])
        custom_gate_id = id(qc.data[-1][0])
        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                f"gate cx_{custom_gate_id} _gate_q_0, _gate_q_1 {{",
                "  cx _gate_q_0, _gate_q_1;",
                "}",
                "qubit[2] _all_qubits;",
                "let q = _all_qubits[0:1];",
                f"cx_{custom_gate_id} q[0], q[1];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(qc), expected_qasm)

    @requires_fixed_parameterisation
    def test_no_include(self):
        """Test explicit gate declaration (no include)"""
        q = QuantumRegister(2, "q")
        circuit = QuantumCircuit(q)
        circuit.rz(pi / 2, 0)
        circuit.sx(0)
        circuit.cx(0, 1)
        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                "gate cx c, t {",
                "  ctrl @ U(pi, 0, pi) c, t;",
                "}",
                "gate u3(_gate_p_0, _gate_p_1, _gate_p_2) _gate_q_0 {",
                "  U(0, 0, pi/2) _gate_q_0;",
                "}",
                "gate u1(_gate_p_0) _gate_q_0 {",
                "  u3(0, 0, pi/2) _gate_q_0;",
                "}",
                "gate rz(_gate_p_0) _gate_q_0 {",
                "  u1(pi/2) _gate_q_0;",
                "}",
                "gate sdg _gate_q_0 {",
                "  u1(-pi/2) _gate_q_0;",
                "}",
                "gate u2(_gate_p_0, _gate_p_1) _gate_q_0 {",
                "  u3(pi/2, 0, pi) _gate_q_0;",
                "}",
                "gate h _gate_q_0 {",
                "  u2(0, pi) _gate_q_0;",
                "}",
                "gate sx _gate_q_0 {",
                "  sdg _gate_q_0;",
                "  h _gate_q_0;",
                "  sdg _gate_q_0;",
                "}",
                "qubit[2] _all_qubits;",
                "let q = _all_qubits[0:1];",
                "rz(pi/2) q[0];",
                "sx q[0];",
                "cx q[0], q[1];",
                "",
            ]
        )
        self.assertEqual(Exporter(includes=[]).dumps(circuit), expected_qasm)

    @requires_fixed_parameterisation
    def test_teleportation(self):
        """Teleportation with physical qubits"""
        qc = QuantumCircuit(3, 2)
        qc.h(1)
        qc.cx(1, 2)
        qc.barrier()
        qc.cx(0, 1)
        qc.h(0)
        qc.barrier()
        qc.measure([0, 1], [0, 1])
        qc.barrier()
        qc.x(2).c_if(qc.clbits[1], 1)
        qc.z(2).c_if(qc.clbits[0], 1)

        transpiled = transpile(qc, initial_layout=[0, 1, 2])
        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                "gate cx c, t {",
                "  ctrl @ U(pi, 0, pi) c, t;",
                "}",
                "gate u3(_gate_p_0, _gate_p_1, _gate_p_2) _gate_q_0 {",
                "  U(pi/2, 0, pi) _gate_q_0;",
                "}",
                "gate u2(_gate_p_0, _gate_p_1) _gate_q_0 {",
                "  u3(pi/2, 0, pi) _gate_q_0;",
                "}",
                "gate h _gate_q_0 {",
                "  u2(0, pi) _gate_q_0;",
                "}",
                "gate x _gate_q_0 {",
                "  u3(pi, 0, pi) _gate_q_0;",
                "}",
                "gate u1(_gate_p_0) _gate_q_0 {",
                "  u3(0, 0, pi) _gate_q_0;",
                "}",
                "gate z _gate_q_0 {",
                "  u1(pi) _gate_q_0;",
                "}",
                "bit[2] c;",
                "h $1;",
                "cx $1, $2;",
                "barrier $0, $1, $2;",
                "cx $0, $1;",
                "h $0;",
                "barrier $0, $1, $2;",
                "c[0] = measure $0;",
                "c[1] = measure $1;",
                "barrier $0, $1, $2;",
                "if (c[1] == 1) {",
                "  x $2;",
                "}",
                "if (c[0] == 1) {",
                "  z $2;",
                "}",
                "",
            ]
        )
        self.assertEqual(Exporter(includes=[]).dumps(transpiled), expected_qasm)

    @requires_fixed_parameterisation
    def test_basis_gates(self):
        """Teleportation with physical qubits"""
        qc = QuantumCircuit(3, 2)
        qc.h(1)
        qc.cx(1, 2)
        qc.barrier()
        qc.cx(0, 1)
        qc.h(0)
        qc.barrier()
        qc.measure([0, 1], [0, 1])
        qc.barrier()
        qc.x(2).c_if(qc.clbits[1], 1)
        qc.z(2).c_if(qc.clbits[0], 1)

        transpiled = transpile(qc, initial_layout=[0, 1, 2])
        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                "gate u3(_gate_p_0, _gate_p_1, _gate_p_2) _gate_q_0 {",
                "  U(pi/2, 0, pi) _gate_q_0;",
                "}",
                "gate u2(_gate_p_0, _gate_p_1) _gate_q_0 {",
                "  u3(pi/2, 0, pi) _gate_q_0;",
                "}",
                "gate h _gate_q_0 {",
                "  u2(0, pi) _gate_q_0;",
                "}",
                "gate x _gate_q_0 {",
                "  u3(pi, 0, pi) _gate_q_0;",
                "}",
                "bit[2] c;",
                "h $1;",
                "cx $1, $2;",
                "barrier $0, $1, $2;",
                "cx $0, $1;",
                "h $0;",
                "barrier $0, $1, $2;",
                "c[0] = measure $0;",
                "c[1] = measure $1;",
                "barrier $0, $1, $2;",
                "if (c[1] == 1) {",
                "  x $2;",
                "}",
                "if (c[0] == 1) {",
                "  z $2;",
                "}",
                "",
            ]
        )
        self.assertEqual(
            Exporter(includes=[], basis_gates=["cx", "z", "U"]).dumps(transpiled),
            expected_qasm,
        )

    def test_reset_statement(self):
        """Test that a reset statement gets output into valid QASM 3.  This includes tests of reset
        operations on single qubits and in nested scopes."""
        inner = QuantumCircuit(1, name="inner_gate")
        inner.reset(0)
        qreg = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qreg)
        qc.reset(0)
        qc.append(inner, [1], [])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                "def inner_gate(qubit _gate_q_0) {",
                "  reset _gate_q_0;",
                "}",
                "qubit[2] _all_qubits;",
                "let qr = _all_qubits[0:1];",
                "reset qr[0];",
                "inner_gate qr[1];",
                "",
            ]
        )
        self.assertEqual(Exporter(includes=[]).dumps(qc), expected_qasm)

    def test_loose_qubits(self):
        """Test that qubits that are not in any register can be used without issue."""
        bits = [Qubit(), Qubit()]
        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(bits, qr, cr)
        qc.h(0)
        qc.h(1)
        qc.h(2)
        qc.h(3)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit[4] _all_qubits;",
                "let qr = _all_qubits[2:3];",
                "h _all_qubits[0];",
                "h _all_qubits[1];",
                "h qr[0];",
                "h qr[1];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_loose_clbits(self):
        """Test that clbits that are not in any register can be used without issue."""
        qreg = QuantumRegister(1, name="qr")
        bits = [Clbit() for _ in [None] * 7]
        cr1 = ClassicalRegister(name="cr1", bits=bits[1:3])
        cr2 = ClassicalRegister(name="cr2", bits=bits[4:6])
        qc = QuantumCircuit(bits, qreg, cr1, cr2)
        qc.measure(0, 0)
        qc.measure(0, 1)
        qc.measure(0, 2)
        qc.measure(0, 3)
        qc.measure(0, 4)
        qc.measure(0, 5)
        qc.measure(0, 6)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "bit[3] _loose_clbits;",
                "bit[2] cr1;",
                "bit[2] cr2;",
                "qubit[1] _all_qubits;",
                "let qr = _all_qubits[0:0];",
                "_loose_clbits[0] = measure qr[0];",
                "cr1[0] = measure qr[0];",
                "cr1[1] = measure qr[0];",
                "_loose_clbits[1] = measure qr[0];",
                "cr2[0] = measure qr[0];",
                "cr2[1] = measure qr[0];",
                "_loose_clbits[2] = measure qr[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_alias_classical_registers(self):
        """Test that clbits that are not in any register can be used without issue."""
        qreg = QuantumRegister(1, name="qr")
        bits = [Clbit() for _ in [None] * 7]
        cr1 = ClassicalRegister(name="cr1", bits=bits[1:3])
        cr2 = ClassicalRegister(name="cr2", bits=bits[4:6])
        # cr3 overlaps cr2, but this should be allowed in this alias form.
        cr3 = ClassicalRegister(name="cr3", bits=bits[5:])
        qc = QuantumCircuit(bits, qreg, cr1, cr2, cr3)
        qc.measure(0, 0)
        qc.measure(0, 1)
        qc.measure(0, 2)
        qc.measure(0, 3)
        qc.measure(0, 4)
        qc.measure(0, 5)
        qc.measure(0, 6)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "bit[7] _all_clbits;",
                "let cr1 = _all_clbits[1:2];",
                "let cr2 = _all_clbits[4:5];",
                "let cr3 = _all_clbits[5:6];",
                "qubit[1] _all_qubits;",
                "let qr = _all_qubits[0:0];",
                "_all_clbits[0] = measure qr[0];",
                "cr1[0] = measure qr[0];",
                "cr1[1] = measure qr[0];",
                "_all_clbits[3] = measure qr[0];",
                "cr2[0] = measure qr[0];",
                "cr2[1] = measure qr[0];",
                "cr3[1] = measure qr[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc, alias_classical_registers=True), expected_qasm)

    def test_simple_for_loop(self):
        """Test that a simple for loop outputs the expected result."""
        parameter = Parameter("x")
        loop_body = QuantumCircuit(1)
        loop_body.rx(parameter, 0)
        loop_body.break_loop()
        loop_body.continue_loop()

        qc = QuantumCircuit(2)
        qc.for_loop([0, 3, 4], parameter, loop_body, [1], [])
        qc.x(0)

        qr_name = qc.qregs[0].name

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "qubit[2] _all_qubits;",
                f"let {qr_name} = _all_qubits[0:1];",
                f"for {parameter.name} in {{0, 3, 4}} {{",
                f"  rx({parameter.name}) {qr_name}[1];",
                "  break;",
                "  continue;",
                "}",
                f"x {qr_name}[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_nested_for_loop(self):
        """Test that a for loop nested inside another outputs the expected result."""
        inner_parameter = Parameter("x")
        outer_parameter = Parameter("y")

        inner_body = QuantumCircuit(2)
        inner_body.rz(inner_parameter, 0)
        inner_body.rz(outer_parameter, 1)
        inner_body.break_loop()

        outer_body = QuantumCircuit(2)
        outer_body.h(0)
        outer_body.rz(outer_parameter, 1)
        # Note we reverse the order of the bits here to test that this is traced.
        outer_body.for_loop(range(1, 5, 2), inner_parameter, inner_body, [1, 0], [])
        outer_body.continue_loop()

        qc = QuantumCircuit(2)
        qc.for_loop(range(4), outer_parameter, outer_body, [0, 1], [])
        qc.x(0)

        qr_name = qc.qregs[0].name

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "qubit[2] _all_qubits;",
                f"let {qr_name} = _all_qubits[0:1];",
                f"for {outer_parameter.name} in [0:3] {{",
                f"  h {qr_name}[0];",
                f"  rz({outer_parameter.name}) {qr_name}[1];",
                f"  for {inner_parameter.name} in [1:2:4] {{",
                # Note the reversed bit order.
                f"    rz({inner_parameter.name}) {qr_name}[1];",
                f"    rz({outer_parameter.name}) {qr_name}[0];",
                "    break;",
                "  }",
                "  continue;",
                "}",
                f"x {qr_name}[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_regular_parameter_in_nested_for_loop(self):
        """Test that a for loop nested inside another outputs the expected result, including
        defining parameters that are used in nested loop scopes."""
        inner_parameter = Parameter("x")
        outer_parameter = Parameter("y")
        regular_parameter = Parameter("t")

        inner_body = QuantumCircuit(2)
        inner_body.h(0)
        inner_body.rx(regular_parameter, 1)
        inner_body.break_loop()

        outer_body = QuantumCircuit(2)
        outer_body.h(0)
        outer_body.h(1)
        # Note we reverse the order of the bits here to test that this is traced.
        outer_body.for_loop(range(1, 5, 2), inner_parameter, inner_body, [1, 0], [])
        outer_body.continue_loop()

        qc = QuantumCircuit(2)
        qc.for_loop(range(4), outer_parameter, outer_body, [0, 1], [])
        qc.x(0)

        qr_name = qc.qregs[0].name

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                # This next line will be missing until gh-7280 is fixed.
                f"input float[64] {regular_parameter.name};",
                "qubit[2] _all_qubits;",
                f"let {qr_name} = _all_qubits[0:1];",
                f"for {outer_parameter.name} in [0:3] {{",
                f"  h {qr_name}[0];",
                f"  h {qr_name}[1];",
                f"  for {inner_parameter.name} in [1:2:4] {{",
                # Note the reversed bit order.
                f"    h {qr_name}[1];",
                f"    rx({regular_parameter.name}) {qr_name}[0];",
                "    break;",
                "  }",
                "  continue;",
                "}",
                f"x {qr_name}[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_for_loop_with_no_parameter(self):
        """Test that a for loop with the parameter set to ``None`` outputs the expected result."""
        loop_body = QuantumCircuit(1)
        loop_body.h(0)

        qc = QuantumCircuit(2)
        qc.for_loop([0, 3, 4], None, loop_body, [1], [])
        qr_name = qc.qregs[0].name

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "qubit[2] _all_qubits;",
                f"let {qr_name} = _all_qubits[0:1];",
                "for _ in {0, 3, 4} {",
                f"  h {qr_name}[1];",
                "}",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_simple_while_loop(self):
        """Test that a simple while loop works correctly."""
        loop_body = QuantumCircuit(1)
        loop_body.h(0)
        loop_body.break_loop()
        loop_body.continue_loop()

        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr)
        qc.while_loop((cr, 0), loop_body, [1], [])
        qc.x(0)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit[2] _all_qubits;",
                "let qr = _all_qubits[0:1];",
                "while (cr == 0) {",
                "  h qr[1];",
                "  break;",
                "  continue;",
                "}",
                "x qr[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_nested_while_loop(self):
        """Test that a while loop nested inside another outputs the expected result."""
        inner_body = QuantumCircuit(2, 2)
        inner_body.measure(0, 0)
        inner_body.measure(1, 1)
        inner_body.break_loop()

        outer_body = QuantumCircuit(2, 2)
        outer_body.measure(0, 0)
        outer_body.measure(1, 1)
        # We reverse the order of the bits here to test this works, and test a single-bit condition.
        outer_body.while_loop((outer_body.clbits[0], 0), inner_body, [1, 0], [1, 0])
        outer_body.continue_loop()

        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr)
        qc.while_loop((cr, 0), outer_body, [0, 1], [0, 1])
        qc.x(0)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit[2] _all_qubits;",
                "let qr = _all_qubits[0:1];",
                "while (cr == 0) {",
                "  cr[0] = measure qr[0];",
                "  cr[1] = measure qr[1];",
                # Note the reversed bits in the body.
                "  while (cr[0] == 0) {",
                "    cr[1] = measure qr[1];",
                "    cr[0] = measure qr[0];",
                "    break;",
                "  }",
                "  continue;",
                "}",
                "x qr[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_simple_if_statement(self):
        """Test that a simple if statement with no else works correctly."""
        true_body = QuantumCircuit(1)
        true_body.h(0)

        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr)
        qc.if_test((cr, 0), true_body, [1], [])
        qc.x(0)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit[2] _all_qubits;",
                "let qr = _all_qubits[0:1];",
                "if (cr == 0) {",
                "  h qr[1];",
                "}",
                "x qr[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_simple_if_else_statement(self):
        """Test that a simple if statement with an else branch works correctly."""
        true_body = QuantumCircuit(1)
        true_body.h(0)
        false_body = QuantumCircuit(1)
        false_body.z(0)

        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr)
        qc.if_else((cr, 0), true_body, false_body, [1], [])
        qc.x(0)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit[2] _all_qubits;",
                "let qr = _all_qubits[0:1];",
                "if (cr == 0) {",
                "  h qr[1];",
                "} else {",
                "  z qr[1];",
                "}",
                "x qr[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_nested_if_else_statement(self):
        """Test that a nested if/else statement works correctly."""
        inner_true_body = QuantumCircuit(2, 2)
        inner_true_body.measure(0, 0)
        inner_false_body = QuantumCircuit(2, 2)
        inner_false_body.measure(1, 1)

        outer_true_body = QuantumCircuit(2, 2)
        outer_true_body.if_else(
            (outer_true_body.clbits[0], 0), inner_true_body, inner_false_body, [0, 1], [0, 1]
        )
        outer_false_body = QuantumCircuit(2, 2)
        # Note the flipped bits here.
        outer_false_body.if_else(
            (outer_false_body.clbits[0], 1), inner_true_body, inner_false_body, [1, 0], [1, 0]
        )

        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr)
        qc.if_else((cr, 0), outer_true_body, outer_false_body, [0, 1], [0, 1])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit[2] _all_qubits;",
                "let qr = _all_qubits[0:1];",
                "if (cr == 0) {",
                "  if (cr[0] == 0) {",
                "    cr[0] = measure qr[0];",
                "  } else {",
                "    cr[1] = measure qr[1];",
                "  }",
                "} else {",
                "  if (cr[0] == 1) {",
                "    cr[1] = measure qr[1];",
                "  } else {",
                "    cr[0] = measure qr[0];",
                "  }",
                "}",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_chain_else_if(self):
        """Test the basic 'else/if' chaining logic for flattening the else scope if its content is a
        single if/else statement."""
        inner_true_body = QuantumCircuit(2, 2)
        inner_true_body.measure(0, 0)
        inner_false_body = QuantumCircuit(2, 2)
        inner_false_body.measure(1, 1)

        outer_true_body = QuantumCircuit(2, 2)
        outer_true_body.if_else(
            (outer_true_body.clbits[0], 0), inner_true_body, inner_false_body, [0, 1], [0, 1]
        )
        outer_false_body = QuantumCircuit(2, 2)
        # Note the flipped bits here.
        outer_false_body.if_else(
            (outer_false_body.clbits[0], 1), inner_true_body, inner_false_body, [1, 0], [1, 0]
        )

        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr)
        qc.if_else((cr, 0), outer_true_body, outer_false_body, [0, 1], [0, 1])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit[2] _all_qubits;",
                "let qr = _all_qubits[0:1];",
                "if (cr == 0) {",
                "  if (cr[0] == 0) {",
                "    cr[0] = measure qr[0];",
                "  } else {",
                "    cr[1] = measure qr[1];",
                "  }",
                "} else if (cr[0] == 1) {",
                "  cr[1] = measure qr[1];",
                "} else {",
                "  cr[0] = measure qr[0];",
                "}",
                "",
            ]
        )
        # This is not the default behaviour, and it's pretty buried how you'd access it.
        builder = QASM3Builder(
            qc,
            includeslist=("stdgates.inc",),
            basis_gates=("U",),
            disable_constants=False,
            alias_classical_registers=False,
        )
        stream = StringIO()
        BasicPrinter(stream, indent="  ", chain_else_if=True).visit(builder.build_program())
        self.assertEqual(stream.getvalue(), expected_qasm)

    def test_chain_else_if_does_not_chain_if_extra_instructions(self):
        """Test the basic 'else/if' chaining logic for flattening the else scope if its content is a
        single if/else statement does not cause a flattening if the 'else' block is not a single
        if/else."""
        inner_true_body = QuantumCircuit(2, 2)
        inner_true_body.measure(0, 0)
        inner_false_body = QuantumCircuit(2, 2)
        inner_false_body.measure(1, 1)

        outer_true_body = QuantumCircuit(2, 2)
        outer_true_body.if_else(
            (outer_true_body.clbits[0], 0), inner_true_body, inner_false_body, [0, 1], [0, 1]
        )
        outer_false_body = QuantumCircuit(2, 2)
        # Note the flipped bits here.
        outer_false_body.if_else(
            (outer_false_body.clbits[0], 1), inner_true_body, inner_false_body, [1, 0], [1, 0]
        )
        outer_false_body.h(0)

        qr = QuantumRegister(2, name="qr")
        cr = ClassicalRegister(2, name="cr")
        qc = QuantumCircuit(qr, cr)
        qc.if_else((cr, 0), outer_true_body, outer_false_body, [0, 1], [0, 1])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "bit[2] cr;",
                "qubit[2] _all_qubits;",
                "let qr = _all_qubits[0:1];",
                "if (cr == 0) {",
                "  if (cr[0] == 0) {",
                "    cr[0] = measure qr[0];",
                "  } else {",
                "    cr[1] = measure qr[1];",
                "  }",
                "} else {",
                "  if (cr[0] == 1) {",
                "    cr[1] = measure qr[1];",
                "  } else {",
                "    cr[0] = measure qr[0];",
                "  }",
                "  h qr[0];",
                "}",
                "",
            ]
        )
        # This is not the default behaviour, and it's pretty buried how you'd access it.
        builder = QASM3Builder(
            qc,
            includeslist=("stdgates.inc",),
            basis_gates=("U",),
            disable_constants=False,
            alias_classical_registers=False,
        )
        stream = StringIO()
        BasicPrinter(stream, indent="  ", chain_else_if=True).visit(builder.build_program())
        self.assertEqual(stream.getvalue(), expected_qasm)

    def test_custom_gate_used_in_loop_scope(self):
        """Test that a custom gate only used within a loop scope still gets a definition at the top
        level."""
        parameter_a = Parameter("a")
        parameter_b = Parameter("b")

        custom = QuantumCircuit(1)
        custom.rx(parameter_a, 0)
        custom_gate = custom.bind_parameters({parameter_a: 0.5}).to_gate()
        custom_gate.name = "custom"

        loop_body = QuantumCircuit(1)
        loop_body.append(custom_gate, [0])

        qc = QuantumCircuit(1)
        qc.for_loop(range(2), parameter_b, loop_body, [0], [])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "gate custom _gate_q_0 {",
                "  rx(0.5) _gate_q_0;",
                "}",
                "qubit[1] _all_qubits;",
                "let q = _all_qubits[0:0];",
                "for b in [0:1] {",
                "  custom q[0];",
                "}",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_parameter_expression_after_naming_escape(self):
        """Test that :class:`.Parameter` instances are correctly renamed when they are used with
        :class:`.ParameterExpression` blocks, even if they have names that needed to be escaped."""
        param = Parameter("measure")  # an invalid name
        qc = QuantumCircuit(1)
        qc.u(2 * param, 0, 0, 0)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "input float[64] measure__generated0;",
                "qubit[1] _all_qubits;",
                "let q = _all_qubits[0:0];",
                "U(2*measure__generated0, 0, 0) q[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_parameters_and_registers_cannot_have_naming_clashes(self):
        """Test that parameters and registers are considered part of the same symbol table for the
        purposes of avoiding clashes."""
        qreg = QuantumRegister(1, "clash")
        param = Parameter("clash")
        qc = QuantumCircuit(qreg)
        qc.u(param, 0, 0, 0)

        out_qasm = dumps(qc)
        register_name = self.register_regex.search(out_qasm)
        parameter_name = self.scalar_parameter_regex.search(out_qasm)
        self.assertTrue(register_name)
        self.assertTrue(parameter_name)
        self.assertIn("clash", register_name["name"])
        self.assertIn("clash", parameter_name["name"])
        self.assertNotEqual(register_name["name"], parameter_name["name"])

    # Not necessarily all the reserved keywords, just a sensibly-sized subset.
    @data("bit", "const", "def", "defcal", "float", "gate", "include", "int", "let", "measure")
    def test_reserved_keywords_as_names_are_escaped(self, keyword):
        """Test that reserved keywords used to name registers and parameters are escaped into
        another form when output, and the escaping cannot introduce new conflicts."""
        with self.subTest("register"):
            qreg = QuantumRegister(1, keyword)
            qc = QuantumCircuit(qreg)
            out_qasm = dumps(qc)
            register_name = self.register_regex.search(out_qasm)
            self.assertTrue(register_name)
            self.assertNotEqual(keyword, register_name["name"])
        with self.subTest("parameter"):
            qc = QuantumCircuit(1)
            param = Parameter(keyword)
            qc.u(param, 0, 0, 0)
            out_qasm = dumps(qc)
            parameter_name = self.scalar_parameter_regex.search(out_qasm)
            self.assertTrue(parameter_name)
            self.assertNotEqual(keyword, parameter_name["name"])


class TestCircuitQASM3ExporterTemporaryCasesWithBadParameterisation(QiskitTestCase):
    """Test functionality that is not what we _want_, but is what we need to do while the definition
    of custom gates with parameterisation does not work correctly.

    These tests are modified versions of those marked with the `requires_fixed_parameterisation`
    decorator, and this whole class can be deleted once those are fixed.  See gh-7335.
    """

    maxDiff = 1_000_000

    def test_basis_gates(self):
        """Teleportation with physical qubits"""
        qc = QuantumCircuit(3, 2)
        first_h = qc.h(1)[0]
        qc.cx(1, 2)
        qc.barrier()
        qc.cx(0, 1)
        qc.h(0)
        qc.barrier()
        qc.measure([0, 1], [0, 1])
        qc.barrier()
        first_x = qc.x(2).c_if(qc.clbits[1], 1)[0]
        qc.z(2).c_if(qc.clbits[0], 1)

        u2 = first_h.definition.data[0][0]
        u3_1 = u2.definition.data[0][0]
        u3_2 = first_x.definition.data[0][0]

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                f"gate u3_{id(u3_1)}(_gate_p_0, _gate_p_1, _gate_p_2) _gate_q_0 {{",
                "  U(pi/2, 0, pi) _gate_q_0;",
                "}",
                f"gate u2_{id(u2)}(_gate_p_0, _gate_p_1) _gate_q_0 {{",
                f"  u3_{id(u3_1)}(pi/2, 0, pi) _gate_q_0;",
                "}",
                "gate h _gate_q_0 {",
                f"  u2_{id(u2)}(0, pi) _gate_q_0;",
                "}",
                f"gate u3_{id(u3_2)}(_gate_p_0, _gate_p_1, _gate_p_2) _gate_q_0 {{",
                "  U(pi, 0, pi) _gate_q_0;",
                "}",
                "gate x _gate_q_0 {",
                f"  u3_{id(u3_2)}(pi, 0, pi) _gate_q_0;",
                "}",
                "bit[2] c;",
                "qubit[3] _all_qubits;",
                "let q = _all_qubits[0:2];",
                "h q[1];",
                "cx q[1], q[2];",
                "barrier q[0], q[1], q[2];",
                "cx q[0], q[1];",
                "h q[0];",
                "barrier q[0], q[1], q[2];",
                "c[0] = measure q[0];",
                "c[1] = measure q[1];",
                "barrier q[0], q[1], q[2];",
                "if (c[1] == 1) {",
                "  x q[2];",
                "}",
                "if (c[0] == 1) {",
                "  z q[2];",
                "}",
                "",
            ]
        )
        self.assertEqual(
            Exporter(includes=[], basis_gates=["cx", "z", "U"]).dumps(qc),
            expected_qasm,
        )

    def test_teleportation(self):
        """Teleportation with physical qubits"""
        qc = QuantumCircuit(3, 2)
        qc.h(1)
        qc.cx(1, 2)
        qc.barrier()
        qc.cx(0, 1)
        qc.h(0)
        qc.barrier()
        qc.measure([0, 1], [0, 1])
        qc.barrier()
        qc.x(2).c_if(qc.clbits[1], 1)
        qc.z(2).c_if(qc.clbits[0], 1)

        transpiled = transpile(qc, initial_layout=[0, 1, 2])
        first_h = transpiled.data[0][0]
        u2 = first_h.definition.data[0][0]
        u3_1 = u2.definition.data[0][0]
        first_x = transpiled.data[-2][0]
        u3_2 = first_x.definition.data[0][0]
        first_z = transpiled.data[-1][0]
        u1 = first_z.definition.data[0][0]
        u3_3 = u1.definition.data[0][0]

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                "gate cx c, t {",
                "  ctrl @ U(pi, 0, pi) c, t;",
                "}",
                f"gate u3_{id(u3_1)}(_gate_p_0, _gate_p_1, _gate_p_2) _gate_q_0 {{",
                "  U(pi/2, 0, pi) _gate_q_0;",
                "}",
                f"gate u2_{id(u2)}(_gate_p_0, _gate_p_1) _gate_q_0 {{",
                f"  u3_{id(u3_1)}(pi/2, 0, pi) _gate_q_0;",
                "}",
                "gate h _gate_q_0 {",
                f"  u2_{id(u2)}(0, pi) _gate_q_0;",
                "}",
                f"gate u3_{id(u3_2)}(_gate_p_0, _gate_p_1, _gate_p_2) _gate_q_0 {{",
                "  U(pi, 0, pi) _gate_q_0;",
                "}",
                "gate x _gate_q_0 {",
                f"  u3_{id(u3_2)}(pi, 0, pi) _gate_q_0;",
                "}",
                f"gate u3_{id(u3_3)}(_gate_p_0, _gate_p_1, _gate_p_2) _gate_q_0 {{",
                "  U(0, 0, pi) _gate_q_0;",
                "}",
                f"gate u1_{id(u1)}(_gate_p_0) _gate_q_0 {{",
                f"  u3_{id(u3_3)}(0, 0, pi) _gate_q_0;",
                "}",
                "gate z _gate_q_0 {",
                f"  u1_{id(u1)}(pi) _gate_q_0;",
                "}",
                "bit[2] c;",
                "h $1;",
                "cx $1, $2;",
                "barrier $0, $1, $2;",
                "cx $0, $1;",
                "h $0;",
                "barrier $0, $1, $2;",
                "c[0] = measure $0;",
                "c[1] = measure $1;",
                "barrier $0, $1, $2;",
                "if (c[1] == 1) {",
                "  x $2;",
                "}",
                "if (c[0] == 1) {",
                "  z $2;",
                "}",
                "",
            ]
        )
        self.assertEqual(Exporter(includes=[]).dumps(transpiled), expected_qasm)

    def test_custom_gate_with_params_bound_main_call(self):
        """Custom gate with unbound parameters that are bound in the main circuit"""
        parameter0 = Parameter("p0")
        parameter1 = Parameter("p1")

        custom = QuantumCircuit(2, name="custom")
        custom.rz(parameter0, 0)
        custom.rz(parameter1 / 2, 1)

        qr_all_qubits = QuantumRegister(3, "q")
        qr_r = QuantumRegister(3, "r")
        circuit = QuantumCircuit(qr_all_qubits, qr_r)
        circuit.append(custom.to_gate(), [qr_all_qubits[0], qr_r[0]])

        circuit.assign_parameters({parameter0: pi, parameter1: pi / 2}, inplace=True)
        custom_id = id(circuit.data[0][0])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                f"gate custom_{custom_id}(p0, p1) _gate_q_0, _gate_q_1 {{",
                "  rz(pi) _gate_q_0;",
                "  rz(pi/4) _gate_q_1;",
                "}",
                "qubit[6] _all_qubits;",
                "let q = _all_qubits[0:2];",
                "let r = _all_qubits[3:5];",
                f"custom_{custom_id}(pi, pi/2) q[0], r[0];",
                "",
            ]
        )
        self.assertEqual(Exporter().dumps(circuit), expected_qasm)

    def test_no_include(self):
        """Test explicit gate declaration (no include)"""
        q = QuantumRegister(2, "q")
        circuit = QuantumCircuit(q)
        circuit.rz(pi / 2, 0)
        circuit.sx(0)
        circuit.cx(0, 1)

        rz = circuit.data[0][0]
        u1_1 = rz.definition.data[0][0]
        u3_1 = u1_1.definition.data[0][0]
        sx = circuit.data[1][0]
        sdg = sx.definition.data[0][0]
        u1_2 = sdg.definition.data[0][0]
        u3_2 = u1_2.definition.data[0][0]
        h_ = sx.definition.data[1][0]
        u2_1 = h_.definition.data[0][0]
        u3_3 = u2_1.definition.data[0][0]
        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                "gate cx c, t {",
                "  ctrl @ U(pi, 0, pi) c, t;",
                "}",
                f"gate u3_{id(u3_1)}(_gate_p_0, _gate_p_1, _gate_p_2) _gate_q_0 {{",
                "  U(0, 0, pi/2) _gate_q_0;",
                "}",
                f"gate u1_{id(u1_1)}(_gate_p_0) _gate_q_0 {{",
                f"  u3_{id(u3_1)}(0, 0, pi/2) _gate_q_0;",
                "}",
                f"gate rz_{id(rz)}(_gate_p_0) _gate_q_0 {{",
                f"  u1_{id(u1_1)}(pi/2) _gate_q_0;",
                "}",
                f"gate u3_{id(u3_2)}(_gate_p_0, _gate_p_1, _gate_p_2) _gate_q_0 {{",
                "  U(0, 0, -pi/2) _gate_q_0;",
                "}",
                f"gate u1_{id(u1_2)}(_gate_p_0) _gate_q_0 {{",
                f"  u3_{id(u3_2)}(0, 0, -pi/2) _gate_q_0;",
                "}",
                "gate sdg _gate_q_0 {",
                f"  u1_{id(u1_2)}(-pi/2) _gate_q_0;",
                "}",
                f"gate u3_{id(u3_3)}(_gate_p_0, _gate_p_1, _gate_p_2) _gate_q_0 {{",
                "  U(pi/2, 0, pi) _gate_q_0;",
                "}",
                f"gate u2_{id(u2_1)}(_gate_p_0, _gate_p_1) _gate_q_0 {{",
                f"  u3_{id(u3_3)}(pi/2, 0, pi) _gate_q_0;",
                "}",
                "gate h _gate_q_0 {",
                f"  u2_{id(u2_1)}(0, pi) _gate_q_0;",
                "}",
                "gate sx _gate_q_0 {",
                "  sdg _gate_q_0;",
                "  h _gate_q_0;",
                "  sdg _gate_q_0;",
                "}",
                "qubit[2] _all_qubits;",
                "let q = _all_qubits[0:1];",
                f"rz_{id(rz)}(pi/2) q[0];",
                "sx q[0];",
                "cx q[0], q[1];",
                "",
            ]
        )
        self.assertEqual(Exporter(includes=[]).dumps(circuit), expected_qasm)


@ddt
class TestQASM3ExporterFailurePaths(QiskitTestCase):
    """Tests of the failure paths for the exporter."""

    def test_disallow_overlapping_classical_registers_if_no_aliasing(self):
        """Test that the exporter rejects circuits with a classical bit in more than one register if
        the ``alias_classical_registers`` option is set false."""
        qubits = [Qubit() for _ in [None] * 3]
        clbits = [Clbit() for _ in [None] * 5]
        registers = [ClassicalRegister(bits=clbits[:4]), ClassicalRegister(bits=clbits[1:])]
        qc = QuantumCircuit(qubits, *registers)
        exporter = Exporter(alias_classical_registers=False)
        with self.assertRaisesRegex(QASM3ExporterError, r"Clbit .* is in multiple registers.*"):
            exporter.dumps(qc)

    @data([1, 2, 1.1], [1j, 2])
    def test_disallow_for_loops_with_non_integers(self, indices):
        """Test that the exporter rejects ``for`` loops that include non-integer values in their
        index sets."""
        loop_body = QuantumCircuit()
        qc = QuantumCircuit(2, 2)
        qc.for_loop(indices, None, loop_body, [], [])
        exporter = Exporter()
        with self.assertRaisesRegex(
            QASM3ExporterError, r"The values in QASM 3 'for' loops must all be integers.*"
        ):
            exporter.dumps(qc)

    def test_disallow_custom_subroutine_with_parameters(self):
        """Test that the exporter throws an error instead of trying to export a subroutine with
        parameters, while this is not supported."""
        subroutine = QuantumCircuit(1)
        subroutine.rx(Parameter("x"), 0)

        qc = QuantumCircuit(1)
        qc.append(subroutine.to_instruction(), [0], [])

        exporter = Exporter()
        with self.assertRaisesRegex(
            QASM3ExporterError, "Exporting subroutines with parameters is not yet supported"
        ):
            exporter.dumps(qc)

    def test_disallow_opaque_instruction(self):
        """Test that the exporter throws an error instead of trying to export something into a
        ``defcal`` block, while this is not supported."""

        qc = QuantumCircuit(1)
        qc.append(Instruction("opaque", 1, 0, []), [0], [])

        exporter = Exporter()
        with self.assertRaisesRegex(
            QASM3ExporterError, "Exporting opaque instructions .* is not yet supported"
        ):
            exporter.dumps(qc)
