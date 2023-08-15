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
from math import pi
import re
import unittest

from ddt import ddt, data

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit.circuit import Parameter, Qubit, Clbit, Instruction, Gate, Delay, Barrier
from qiskit.circuit.classical import expr
from qiskit.circuit.controlflow import CASE_DEFAULT
from qiskit.test import QiskitTestCase
from qiskit.qasm3 import Exporter, dumps, dump, QASM3ExporterError, ExperimentalFeatures
from qiskit.qasm3.exporter import QASM3Builder
from qiskit.qasm3.printer import BasicPrinter


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
                "qubit[2] q;",
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
        # input circuits.  They can allow false negatives (in which case, update the regex), but to
        # be useful for the tests must _never_ have false positive matches.  We use an explicit
        # space (`\s`) or semicolon rather than the end-of-word `\b` because we want to ensure that
        # the exporter isn't putting out invalid characters as part of the identifiers.
        cls.register_regex = re.compile(
            r"^\s*(let|(qu)?bit(\[\d+\])?)\s+(?P<name>\w+)[\s;]", re.U | re.M
        )
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
            r"(?P<name>\w+)[\s;]",  # Parameter name
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
                "qubit[1] qr1;",
                "qubit[2] qr2;",
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
                "qubit _qubit0;",
                "qubit _qubit1;",
                "qubit _qubit2;",
                "qubit _qubit3;",
                "qubit _qubit4;",
                "qubit _qubit5;",
                "qubit _qubit6;",
                "qubit _qubit7;",
                "qubit _qubit8;",
                "qubit _qubit9;",
                "let first_four = {_qubit0, _qubit1, _qubit2, _qubit3};",
                "let last_five = {_qubit5, _qubit6, _qubit7, _qubit8, _qubit9};",
                "let alternate = {first_four[0], first_four[2], _qubit4, last_five[1], last_five[3]};",
                "let sporadic = {alternate[2], alternate[1], last_five[4]};",
                "",
            ]
        )
        self.assertEqual(Exporter(allow_aliasing=True).dumps(qc), expected_qasm)

    def test_composite_circuit(self):
        """Test with a composite circuit instruction and barriers"""
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
                "qubit[2] qr;",
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
                "qubit[2] qr;",
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
        composite_circ_instr = composite_circ.to_gate()

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
                "gate composite_circ _gate_q_0, _gate_q_1 {",
                "  h _gate_q_0;",
                "  x _gate_q_1;",
                "  cx _gate_q_0, _gate_q_1;",
                "}",
                "bit[2] cr;",
                "qubit[2] qr;",
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
        my_gate_inst1 = my_gate.to_gate()

        my_gate = QuantumCircuit(1, name="my_gate")
        my_gate.x(0)
        my_gate_inst2 = my_gate.to_gate()

        my_gate = QuantumCircuit(1, name="my_gate")
        my_gate.x(0)
        my_gate_inst3 = my_gate.to_gate()

        qr = QuantumRegister(1, name="qr")
        circuit = QuantumCircuit(qr, name="circuit")
        circuit.append(my_gate_inst1, [qr[0]])
        circuit.append(my_gate_inst2, [qr[0]])
        my_gate_inst2_id = id(circuit.data[-1].operation)
        circuit.append(my_gate_inst3, [qr[0]])
        my_gate_inst3_id = id(circuit.data[-1].operation)
        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                "gate my_gate _gate_q_0 {",
                "  h _gate_q_0;",
                "}",
                f"gate my_gate_{my_gate_inst2_id} _gate_q_0 {{",
                "  x _gate_q_0;",
                "}",
                f"gate my_gate_{my_gate_inst3_id} _gate_q_0 {{",
                "  x _gate_q_0;",
                "}",
                "qubit[1] qr;",
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
                "qubit[2] q;",
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
                "qubit[2] q;",
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
                "input float[64] a;",
                "gate custom(a) _gate_q_0 {",
                "  rx(a) _gate_q_0;",
                "}",
                "qubit[1] q;",
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
                "qubit[1] q;",
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
                "gate custom(_gate_p_0, _gate_p_0) _gate_q_0, _gate_q_1 {",
                "  rz(pi) _gate_q_0;",
                "  rz(pi/4) _gate_q_1;",
                "}",
                "qubit[3] q;",
                "qubit[3] r;",
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

        circuit_name_0 = circuit.data[0].operation.definition.name
        circuit_name_1 = circuit.data[1].operation.definition.name

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                f"gate {circuit_name_0} _gate_q_0 {{",
                "  rx(0.5) _gate_q_0;",
                "}",
                f"gate {circuit_name_1} _gate_q_0 {{",
                "  rx(1.0) _gate_q_0;",
                "}",
                "qubit[1] q;",
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
                "qubit[1] q;",
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
                "input float[64] x;",
                "input float[64] y;",
                "gate rzx(x) _gate_q_0, _gate_q_1 {",
                "  h _gate_q_1;",
                "  cx _gate_q_0, _gate_q_1;",
                "  rz(x) _gate_q_1;",
                "  cx _gate_q_0, _gate_q_1;",
                "  h _gate_q_1;",
                "}",
                "qubit[2] q;",
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
                "qubit[2] q;",
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
        custom_gate_id = id(qc.data[-1].operation)
        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                f"gate cx_{custom_gate_id} _gate_q_0, _gate_q_1 {{",
                "  cx _gate_q_0, _gate_q_1;",
                "}",
                "qubit[2] q;",
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
                "gate cx c, t {",
                "  ctrl @ U(pi, 0, pi) c, t;",
                "}",
                "qubit[2] q;",
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
                "gate u3(_gate_p_0, _gate_p_1, _gate_p_2) _gate_q_0 {",
                "  U(pi/2, 0, pi) _gate_q_0;",
                "}",
                "gate u2(_gate_p_0, _gate_p_1) _gate_q_0 {",
                "  u3(pi/2, 0, pi) _gate_q_0;",
                "}",
                "gate h _gate_q_0 {",
                "  u2(0, pi) _gate_q_0;",
                "}",
                "gate cx c, t {",
                "  ctrl @ U(pi, 0, pi) c, t;",
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
                "if (c[1]) {",
                "  x $2;",
                "}",
                "if (c[0]) {",
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
                "if (c[1]) {",
                "  x $2;",
                "}",
                "if (c[0]) {",
                "  z $2;",
                "}",
                "",
            ]
        )
        self.assertEqual(
            Exporter(includes=[], basis_gates=["cx", "z", "U"]).dumps(transpiled),
            expected_qasm,
        )

    def test_opaque_instruction_in_basis_gates(self):
        """Test that an instruction that is set in the basis gates is output verbatim with no
        definition."""
        qc = QuantumCircuit(1)
        qc.x(0)
        qc.append(Gate("my_gate", 1, []), [0], [])

        basis_gates = ["my_gate", "x"]
        transpiled = transpile(qc, initial_layout=[0])
        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                "x $0;",
                "my_gate $0;",
                "",
            ]
        )
        self.assertEqual(
            Exporter(includes=[], basis_gates=basis_gates).dumps(transpiled), expected_qasm
        )

    def test_reset_statement(self):
        """Test that a reset statement gets output into valid QASM 3.  This includes tests of reset
        operations on single qubits and in nested scopes."""
        qreg = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qreg)
        qc.reset(0)
        qc.reset([0, 1])

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                "qubit[2] qr;",
                "reset qr[0];",
                "reset qr[0];",
                "reset qr[1];",
                "",
            ]
        )
        self.assertEqual(Exporter(includes=[]).dumps(qc), expected_qasm)

    def test_delay_statement(self):
        """Test that delay operations get output into valid QASM 3."""
        qreg = QuantumRegister(2, "qr")
        qc = QuantumCircuit(qreg)
        qc.delay(100, qreg[0], unit="ms")
        qc.delay(2, qreg[1], unit="ps")  # "ps" is not a valid unit in OQ3, so we need to convert.

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                "qubit[2] qr;",
                "delay[100ms] qr[0];",
                "delay[2000ns] qr[1];",
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
                "qubit _qubit0;",
                "qubit _qubit1;",
                "qubit[2] qr;",
                "h _qubit0;",
                "h _qubit1;",
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
                "bit _bit0;",
                "bit _bit3;",
                "bit _bit6;",
                "bit[2] cr1;",
                "bit[2] cr2;",
                "qubit[1] qr;",
                "_bit0 = measure qr[0];",
                "cr1[0] = measure qr[0];",
                "cr1[1] = measure qr[0];",
                "_bit3 = measure qr[0];",
                "cr2[0] = measure qr[0];",
                "cr2[1] = measure qr[0];",
                "_bit6 = measure qr[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_classical_register_aliasing(self):
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
                "bit _bit0;",
                "bit _bit1;",
                "bit _bit2;",
                "bit _bit3;",
                "bit _bit4;",
                "bit _bit5;",
                "bit _bit6;",
                "let cr1 = {_bit1, _bit2};",
                "let cr2 = {_bit4, _bit5};",
                "let cr3 = {cr2[1], _bit6};",
                "qubit[1] qr;",
                "_bit0 = measure qr[0];",
                "cr1[0] = measure qr[0];",
                "cr1[1] = measure qr[0];",
                "_bit3 = measure qr[0];",
                "cr2[0] = measure qr[0];",
                "cr3[0] = measure qr[0];",
                "cr3[1] = measure qr[0];",
                "",
            ]
        )
        self.assertEqual(dumps(qc, allow_aliasing=True), expected_qasm)

    def test_old_alias_classical_registers_option(self):
        """Test that the ``alias_classical_registers`` option still functions during its changeover
        period."""
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
                "bit _bit0;",
                "bit _bit1;",
                "bit _bit2;",
                "bit _bit3;",
                "bit _bit4;",
                "bit _bit5;",
                "bit _bit6;",
                "let cr1 = {_bit1, _bit2};",
                "let cr2 = {_bit4, _bit5};",
                "let cr3 = {cr2[1], _bit6};",
                "qubit[1] qr;",
                "_bit0 = measure qr[0];",
                "cr1[0] = measure qr[0];",
                "cr1[1] = measure qr[0];",
                "_bit3 = measure qr[0];",
                "cr2[0] = measure qr[0];",
                "cr3[0] = measure qr[0];",
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
                f"qubit[2] {qr_name};",
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
                f"qubit[2] {qr_name};",
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
                f"qubit[2] {qr_name};",
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
                f"qubit[2] {qr_name};",
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
                "qubit[2] qr;",
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
                "qubit[2] qr;",
                "while (cr == 0) {",
                "  cr[0] = measure qr[0];",
                "  cr[1] = measure qr[1];",
                # Note the reversed bits in the body.
                "  while (!cr[0]) {",
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
                "qubit[2] qr;",
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
                "qubit[2] qr;",
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
                "qubit[2] qr;",
                "if (cr == 0) {",
                "  if (!cr[0]) {",
                "    cr[0] = measure qr[0];",
                "  } else {",
                "    cr[1] = measure qr[1];",
                "  }",
                "} else {",
                "  if (cr[0]) {",
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
                "qubit[2] qr;",
                "if (cr == 0) {",
                "  if (!cr[0]) {",
                "    cr[0] = measure qr[0];",
                "  } else {",
                "    cr[1] = measure qr[1];",
                "  }",
                "} else if (cr[0]) {",
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
            allow_aliasing=False,
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
                "qubit[2] qr;",
                "if (cr == 0) {",
                "  if (!cr[0]) {",
                "    cr[0] = measure qr[0];",
                "  } else {",
                "    cr[1] = measure qr[1];",
                "  }",
                "} else {",
                "  if (cr[0]) {",
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
            allow_aliasing=False,
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
                "qubit[1] q;",
                "for b in [0:1] {",
                "  custom q[0];",
                "}",
                "",
            ]
        )
        self.assertEqual(dumps(qc), expected_qasm)

    def test_registers_have_escaped_names(self):
        """Test that both types of register are emitted with safely escaped names if they begin with
        invalid names. Regression test of gh-9658."""
        qc = QuantumCircuit(
            QuantumRegister(2, name="q_{reg}"), ClassicalRegister(2, name="c_{reg}")
        )
        qc.measure([0, 1], [0, 1])
        out_qasm = dumps(qc)
        matches = {match_["name"] for match_ in self.register_regex.finditer(out_qasm)}
        self.assertEqual(len(matches), 2, msg=f"Observed OQ3 output:\n{out_qasm}")

    def test_parameters_have_escaped_names(self):
        """Test that parameters are emitted with safely escaped names if they begin with invalid
        names. Regression test of gh-9658."""
        qc = QuantumCircuit(1)
        qc.u(Parameter("p_{0}"), 2 * Parameter("p_?0!"), 0, 0)
        out_qasm = dumps(qc)
        matches = {match_["name"] for match_ in self.scalar_parameter_regex.finditer(out_qasm)}
        self.assertEqual(len(matches), 2, msg=f"Observed OQ3 output:\n{out_qasm}")

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
                "input float[64] _measure;",
                "qubit[1] q;",
                "U(2*_measure, 0, 0) q[0];",
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
            self.assertTrue(register_name, msg=f"Observed OQ3:\n{out_qasm}")
            self.assertNotEqual(keyword, register_name["name"])
        with self.subTest("parameter"):
            qc = QuantumCircuit(1)
            param = Parameter(keyword)
            qc.u(param, 0, 0, 0)
            out_qasm = dumps(qc)
            parameter_name = self.scalar_parameter_regex.search(out_qasm)
            self.assertTrue(parameter_name, msg=f"Observed OQ3:\n{out_qasm}")
            self.assertNotEqual(keyword, parameter_name["name"])

    def test_expr_condition(self):
        """Simple test that the conditions of `if`s and `while`s can be `Expr` nodes."""
        bits = [Qubit(), Clbit()]
        cr = ClassicalRegister(2, "cr")

        if_body = QuantumCircuit(1)
        if_body.x(0)

        while_body = QuantumCircuit(1)
        while_body.x(0)

        qc = QuantumCircuit(bits, cr)
        qc.if_test(expr.logic_not(qc.clbits[0]), if_body, [0], [])
        qc.while_loop(expr.equal(cr, 3), while_body, [0], [])

        expected = """\
OPENQASM 3;
include "stdgates.inc";
bit _bit0;
bit[2] cr;
qubit _qubit0;
if (!_bit0) {
  x _qubit0;
}
while (cr == 3) {
  x _qubit0;
}
"""
        self.assertEqual(dumps(qc), expected)

    def test_expr_nested_condition(self):
        """Simple test that the conditions of `if`s and `while`s can be `Expr` nodes when nested,
        and the mapping of inner bits to outer bits is correct."""
        bits = [Qubit(), Clbit(), Clbit()]
        cr = ClassicalRegister(2, "cr")

        inner_if_body = QuantumCircuit(1)
        inner_if_body.x(0)
        outer_if_body = QuantumCircuit(1, 1)
        outer_if_body.if_test(expr.lift(outer_if_body.clbits[0]), inner_if_body, [0], [])

        inner_while_body = QuantumCircuit(1)
        inner_while_body.x(0)
        outer_while_body = QuantumCircuit([Qubit()], cr)
        outer_while_body.while_loop(expr.equal(expr.bit_and(cr, 3), 3), inner_while_body, [0], [])

        qc = QuantumCircuit(bits, cr)
        qc.if_test(expr.logic_not(qc.clbits[0]), outer_if_body, [0], [1])
        qc.while_loop(expr.equal(cr, 3), outer_while_body, [0], cr)

        expected = """\
OPENQASM 3;
include "stdgates.inc";
bit _bit0;
bit _bit1;
bit[2] cr;
qubit _qubit0;
if (!_bit0) {
  if (_bit1) {
    x _qubit0;
  }
}
while (cr == 3) {
  while ((cr & 3) == 3) {
    x _qubit0;
  }
}
"""
        self.assertEqual(dumps(qc), expected)

    def test_expr_associativity_left(self):
        """Test that operations that are in the expression tree in a left-associative form are
        output to OQ3 correctly."""
        body = QuantumCircuit()

        cr1 = ClassicalRegister(3, "cr1")
        cr2 = ClassicalRegister(3, "cr2")
        cr3 = ClassicalRegister(3, "cr3")
        qc = QuantumCircuit(cr1, cr2, cr3)
        qc.if_test(expr.equal(expr.bit_and(expr.bit_and(cr1, cr2), cr3), 7), body.copy(), [], [])
        qc.if_test(expr.equal(expr.bit_or(expr.bit_or(cr1, cr2), cr3), 7), body.copy(), [], [])
        qc.if_test(expr.equal(expr.bit_xor(expr.bit_xor(cr1, cr2), cr3), 7), body.copy(), [], [])
        qc.if_test(expr.logic_and(expr.logic_and(cr1[0], cr1[1]), cr1[2]), body.copy(), [], [])
        qc.if_test(expr.logic_or(expr.logic_or(cr1[0], cr1[1]), cr1[2]), body.copy(), [], [])

        # Note that bitwise operations have lower priority than `==` so there's extra parentheses.
        # All these operators are left-associative in OQ3.
        expected = """\
OPENQASM 3;
include "stdgates.inc";
bit[3] cr1;
bit[3] cr2;
bit[3] cr3;
if ((cr1 & cr2 & cr3) == 7) {
}
if ((cr1 | cr2 | cr3) == 7) {
}
if ((cr1 ^ cr2 ^ cr3) == 7) {
}
if (cr1[0] && cr1[1] && cr1[2]) {
}
if (cr1[0] || cr1[1] || cr1[2]) {
}
"""
        self.assertEqual(dumps(qc), expected)

    def test_expr_associativity_right(self):
        """Test that operations that are in the expression tree in a right-associative form are
        output to OQ3 correctly."""
        body = QuantumCircuit()

        cr1 = ClassicalRegister(3, "cr1")
        cr2 = ClassicalRegister(3, "cr2")
        cr3 = ClassicalRegister(3, "cr3")
        qc = QuantumCircuit(cr1, cr2, cr3)
        qc.if_test(expr.equal(expr.bit_and(cr1, expr.bit_and(cr2, cr3)), 7), body.copy(), [], [])
        qc.if_test(expr.equal(expr.bit_or(cr1, expr.bit_or(cr2, cr3)), 7), body.copy(), [], [])
        qc.if_test(expr.equal(expr.bit_xor(cr1, expr.bit_xor(cr2, cr3)), 7), body.copy(), [], [])
        qc.if_test(expr.logic_and(cr1[0], expr.logic_and(cr1[1], cr1[2])), body.copy(), [], [])
        qc.if_test(expr.logic_or(cr1[0], expr.logic_or(cr1[1], cr1[2])), body.copy(), [], [])

        # Note that bitwise operations have lower priority than `==` so there's extra parentheses.
        # All these operators are left-associative in OQ3, so we need parentheses for them to be
        # parsed correctly.  Mathematically, they're all actually associative in general, so the
        # order doesn't _technically_ matter.
        expected = """\
OPENQASM 3;
include "stdgates.inc";
bit[3] cr1;
bit[3] cr2;
bit[3] cr3;
if ((cr1 & (cr2 & cr3)) == 7) {
}
if ((cr1 | (cr2 | cr3)) == 7) {
}
if ((cr1 ^ (cr2 ^ cr3)) == 7) {
}
if (cr1[0] && (cr1[1] && cr1[2])) {
}
if (cr1[0] || (cr1[1] || cr1[2])) {
}
"""
        self.assertEqual(dumps(qc), expected)

    def test_expr_binding_unary(self):
        """Test that nested unary operators don't insert unnecessary brackets."""
        body = QuantumCircuit()
        cr = ClassicalRegister(2, "cr")
        qc = QuantumCircuit(cr)
        qc.if_test(expr.equal(expr.bit_not(expr.bit_not(cr)), 3), body.copy(), [], [])
        qc.if_test(expr.logic_not(expr.logic_not(cr[0])), body.copy(), [], [])

        expected = """\
OPENQASM 3;
include "stdgates.inc";
bit[2] cr;
if (~~cr == 3) {
}
if (!!cr[0]) {
}
"""
        self.assertEqual(dumps(qc), expected)

    def test_expr_precedence(self):
        """Test that the precedence properties of operators are correctly output."""
        body = QuantumCircuit()
        cr = ClassicalRegister(2, "cr")
        # This tree is _completely_ inside out, so there's brackets needed round every operand.
        inside_out = expr.logic_not(
            expr.less(
                expr.bit_and(
                    expr.bit_xor(expr.bit_or(cr, cr), expr.bit_or(cr, cr)),
                    expr.bit_xor(expr.bit_or(cr, cr), expr.bit_or(cr, cr)),
                ),
                expr.bit_and(
                    expr.bit_xor(expr.bit_or(cr, cr), expr.bit_or(cr, cr)),
                    expr.bit_xor(expr.bit_or(cr, cr), expr.bit_or(cr, cr)),
                ),
            )
        )
        # This one is the other way round - the tightest-binding operations are on the inside, so no
        # brackets should be needed at all except to put in a comparison to a bitwise binary
        # operation, since those bind less tightly than anything that can cast them to a bool.
        outside_in = expr.logic_or(
            expr.logic_and(
                expr.equal(expr.bit_or(cr, cr), expr.bit_and(cr, cr)),
                expr.equal(expr.bit_and(cr, cr), expr.bit_or(cr, cr)),
            ),
            expr.logic_and(
                expr.greater(expr.bit_or(cr, cr), expr.bit_xor(cr, cr)),
                expr.less_equal(expr.bit_xor(cr, cr), expr.bit_or(cr, cr)),
            ),
        )

        # And an extra test of the logical operator order.
        logics = expr.logic_or(
            expr.logic_and(
                expr.logic_or(expr.logic_not(cr[0]), expr.logic_not(cr[0])),
                expr.logic_not(expr.logic_and(cr[0], cr[0])),
            ),
            expr.logic_and(
                expr.logic_not(expr.logic_and(cr[0], cr[0])),
                expr.logic_or(expr.logic_not(cr[0]), expr.logic_not(cr[0])),
            ),
        )

        qc = QuantumCircuit(cr)
        qc.if_test(inside_out, body.copy(), [], [])
        qc.if_test(outside_in, body.copy(), [], [])
        qc.if_test(logics, body.copy(), [], [])

        expected = """\
OPENQASM 3;
include "stdgates.inc";
bit[2] cr;
if (!((((cr | cr) ^ (cr | cr)) & ((cr | cr) ^ (cr | cr)))\
 < (((cr | cr) ^ (cr | cr)) & ((cr | cr) ^ (cr | cr))))) {
}
if ((cr | cr) == (cr & cr) && (cr & cr) == (cr | cr)\
 || (cr | cr) > (cr ^ cr) && (cr ^ cr) <= (cr | cr)) {
}
if ((!cr[0] || !cr[0]) && !(cr[0] && cr[0]) || !(cr[0] && cr[0]) && (!cr[0] || !cr[0])) {
}
"""
        self.assertEqual(dumps(qc), expected)

    def test_no_unnecessary_cast(self):
        """This is a bit of a cross `Expr`-constructor / OQ3-exporter test.  It doesn't really
        matter whether or not the `Expr` constructor functions insert cast nodes into their output
        for the literals (at the time of writing [commit 2616602], they don't because they do some
        type inference) but the OQ3 export definitely shouldn't have them."""
        cr = ClassicalRegister(8, "cr")
        qc = QuantumCircuit(cr)
        # Note that the integer '1' has a minimum bit-width of 1, whereas the register has a width
        # of 8.  We're testing to make sure that there's no spurious cast up from `bit[1]` to
        # `bit[8]`, or anything like that, _whether or not_ the `Expr` node includes one.
        qc.if_test(expr.equal(cr, 1), QuantumCircuit(), [], [])

        expected = """\
OPENQASM 3;
include "stdgates.inc";
bit[8] cr;
if (cr == 1) {
}
"""
        self.assertEqual(dumps(qc), expected)


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
        first_h = qc.h(1)[0].operation
        qc.cx(1, 2)
        qc.barrier()
        qc.cx(0, 1)
        qc.h(0)
        qc.barrier()
        qc.measure([0, 1], [0, 1])
        qc.barrier()
        first_x = qc.x(2).c_if(qc.clbits[1], 1)[0].operation
        qc.z(2).c_if(qc.clbits[0], 1)

        u2 = first_h.definition.data[0].operation
        u3_1 = u2.definition.data[0].operation
        u3_2 = first_x.definition.data[0].operation

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
                "qubit[3] q;",
                "h q[1];",
                "cx q[1], q[2];",
                "barrier q[0], q[1], q[2];",
                "cx q[0], q[1];",
                "h q[0];",
                "barrier q[0], q[1], q[2];",
                "c[0] = measure q[0];",
                "c[1] = measure q[1];",
                "barrier q[0], q[1], q[2];",
                "if (c[1]) {",
                "  x q[2];",
                "}",
                "if (c[0]) {",
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
        first_h = transpiled.data[0].operation
        u2 = first_h.definition.data[0].operation
        u3_1 = u2.definition.data[0].operation
        first_x = transpiled.data[-2].operation
        u3_2 = first_x.definition.data[0].operation
        first_z = transpiled.data[-1].operation
        u1 = first_z.definition.data[0].operation
        u3_3 = u1.definition.data[0].operation

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
                "gate cx c, t {",
                "  ctrl @ U(pi, 0, pi) c, t;",
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
                "if (c[1]) {",
                "  x $2;",
                "}",
                "if (c[0]) {",
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
        custom_id = id(circuit.data[0].operation)

        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
                'include "stdgates.inc";',
                f"gate custom_{custom_id}(_gate_p_0, _gate_p_1) _gate_q_0, _gate_q_1 {{",
                "  rz(pi) _gate_q_0;",
                "  rz(pi/4) _gate_q_1;",
                "}",
                "qubit[3] q;",
                "qubit[3] r;",
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

        rz = circuit.data[0].operation
        u1_1 = rz.definition.data[0].operation
        u3_1 = u1_1.definition.data[0].operation
        sx = circuit.data[1].operation
        sdg = sx.definition.data[0].operation
        u1_2 = sdg.definition.data[0].operation
        u3_2 = u1_2.definition.data[0].operation
        h_ = sx.definition.data[1].operation
        u2_1 = h_.definition.data[0].operation
        u3_3 = u2_1.definition.data[0].operation
        expected_qasm = "\n".join(
            [
                "OPENQASM 3;",
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
                "gate cx c, t {",
                "  ctrl @ U(pi, 0, pi) c, t;",
                "}",
                "qubit[2] q;",
                f"rz_{id(rz)}(pi/2) q[0];",
                "sx q[0];",
                "cx q[0], q[1];",
                "",
            ]
        )
        self.assertEqual(Exporter(includes=[]).dumps(circuit), expected_qasm)

    def test_unusual_conditions(self):
        """Test that special QASM constructs such as ``measure`` are correctly handled when the
        Terra instructions have old-style conditions."""
        qc = QuantumCircuit(3, 2)
        qc.h(0)
        qc.measure(0, 0)
        qc.measure(1, 1).c_if(0, True)
        qc.reset([0, 1]).c_if(0, True)
        with qc.while_loop((qc.clbits[0], True)):
            qc.break_loop().c_if(0, True)
            qc.continue_loop().c_if(0, True)
        # Terra forbids delay and barrier from being conditioned through `c_if`, but in theory they
        # should work fine in a dynamic-circuits sense (although what a conditional barrier _means_
        # is a whole other kettle of fish).
        delay = Delay(16, "dt")
        delay.condition = (qc.clbits[0], True)
        qc.append(delay, [0], [])
        barrier = Barrier(2)
        barrier.condition = (qc.clbits[0], True)
        qc.append(barrier, [0, 1], [])

        expected = """
OPENQASM 3;
include "stdgates.inc";
bit[2] c;
qubit[3] q;
h q[0];
c[0] = measure q[0];
if (c[0]) {
  c[1] = measure q[1];
}
if (c[0]) {
  reset q[0];
}
if (c[0]) {
  reset q[1];
}
while (c[0]) {
  if (c[0]) {
    break;
  }
  if (c[0]) {
    continue;
  }
}
if (c[0]) {
  delay[16dt] q[0];
}
if (c[0]) {
  barrier q[0], q[1];
}"""
        self.assertEqual(dumps(qc).strip(), expected.strip())


class TestExperimentalFeatures(QiskitTestCase):
    """Tests of features that are hidden behind experimental flags."""

    maxDiff = None

    def test_switch_forbidden_without_flag(self):
        """Omitting the feature flag should raise an error."""
        case = QuantumCircuit(1)
        circuit = QuantumCircuit(1, 1)
        circuit.switch(circuit.clbits[0], [((True, False), case)], [0], [])
        with self.assertRaisesRegex(QASM3ExporterError, "'switch' statements are not stabilized"):
            dumps(circuit)

    def test_switch_clbit(self):
        """Test that a switch statement can be constructed with a bit as a condition."""
        qubit = Qubit()
        clbit = Clbit()
        case1 = QuantumCircuit([qubit, clbit])
        case1.x(0)
        case2 = QuantumCircuit([qubit, clbit])
        case2.z(0)
        circuit = QuantumCircuit([qubit, clbit])
        circuit.switch(clbit, [(True, case1), (False, case2)], [0], [0])

        test = dumps(circuit, experimental=ExperimentalFeatures.SWITCH_CASE_V1)
        expected = """\
OPENQASM 3;
include "stdgates.inc";
bit _bit0;
int switch_dummy;
qubit _qubit0;
switch_dummy = _bit0;
switch (switch_dummy) {
  case 1: {
    x _qubit0;
  }
  break;
  case 0: {
    z _qubit0;
  }
  break;
}
"""
        self.assertEqual(test, expected)

    def test_switch_register(self):
        """Test that a switch statement can be constructed with a register as a condition."""
        qubit = Qubit()
        creg = ClassicalRegister(2, "c")
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)
        case3 = QuantumCircuit([qubit], creg)
        case3.z(0)

        circuit = QuantumCircuit([qubit], creg)
        circuit.switch(creg, [(0, case1), (1, case2), (2, case3)], [0], circuit.clbits)

        test = dumps(circuit, experimental=ExperimentalFeatures.SWITCH_CASE_V1)
        expected = """\
OPENQASM 3;
include "stdgates.inc";
bit[2] c;
int switch_dummy;
qubit _qubit0;
switch_dummy = c;
switch (switch_dummy) {
  case 0: {
    x _qubit0;
  }
  break;
  case 1: {
    y _qubit0;
  }
  break;
  case 2: {
    z _qubit0;
  }
  break;
}
"""
        self.assertEqual(test, expected)

    def test_switch_with_default(self):
        """Test that a switch statement can be constructed with a default case at the end."""
        qubit = Qubit()
        creg = ClassicalRegister(2, "c")
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)
        case3 = QuantumCircuit([qubit], creg)
        case3.z(0)

        circuit = QuantumCircuit([qubit], creg)
        circuit.switch(creg, [(0, case1), (1, case2), (CASE_DEFAULT, case3)], [0], circuit.clbits)

        test = dumps(circuit, experimental=ExperimentalFeatures.SWITCH_CASE_V1)
        expected = """\
OPENQASM 3;
include "stdgates.inc";
bit[2] c;
int switch_dummy;
qubit _qubit0;
switch_dummy = c;
switch (switch_dummy) {
  case 0: {
    x _qubit0;
  }
  break;
  case 1: {
    y _qubit0;
  }
  break;
  default: {
    z _qubit0;
  }
  break;
}
"""
        self.assertEqual(test, expected)

    def test_switch_multiple_cases_to_same_block(self):
        """Test that it is possible to add multiple cases that apply to the same block, if they are
        given as a compound value.  This is an allowed special case of block fall-through."""
        qubit = Qubit()
        creg = ClassicalRegister(2, "c")
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)

        circuit = QuantumCircuit([qubit], creg)
        circuit.switch(creg, [(0, case1), ((1, 2), case2)], [0], circuit.clbits)

        test = dumps(circuit, experimental=ExperimentalFeatures.SWITCH_CASE_V1)
        expected = """\
OPENQASM 3;
include "stdgates.inc";
bit[2] c;
int switch_dummy;
qubit _qubit0;
switch_dummy = c;
switch (switch_dummy) {
  case 0: {
    x _qubit0;
  }
  break;
  case 1:
  case 2: {
    y _qubit0;
  }
  break;
}
"""
        self.assertEqual(test, expected)

    def test_multiple_switches_dont_clash_on_dummy(self):
        """Test that having more than one switch statement in the circuit doesn't cause naming
        clashes in the dummy integer value used."""
        qubit = Qubit()
        creg = ClassicalRegister(2, "switch_dummy")
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)

        circuit = QuantumCircuit([qubit], creg)
        circuit.switch(creg, [(0, case1), ((1, 2), case2)], [0], circuit.clbits)
        circuit.switch(creg, [(0, case1), ((1, 2), case2)], [0], circuit.clbits)

        test = dumps(circuit, experimental=ExperimentalFeatures.SWITCH_CASE_V1)
        expected = """\
OPENQASM 3;
include "stdgates.inc";
bit[2] switch_dummy;
int switch_dummy__generated0;
int switch_dummy__generated1;
qubit _qubit0;
switch_dummy__generated0 = switch_dummy;
switch (switch_dummy__generated0) {
  case 0: {
    x _qubit0;
  }
  break;
  case 1:
  case 2: {
    y _qubit0;
  }
  break;
}
switch_dummy__generated1 = switch_dummy;
switch (switch_dummy__generated1) {
  case 0: {
    x _qubit0;
  }
  break;
  case 1:
  case 2: {
    y _qubit0;
  }
  break;
}
"""
        self.assertEqual(test, expected)

    def test_switch_nested_in_if(self):
        """Test that the switch statement works when in a nested scope, including the dummy
        classical variable being declared globally.  This isn't necessary in the OQ3 language, but
        it is universally valid and the IBM QSS stack prefers that.  They're our primary consumers
        of OQ3 strings, so it's best to play nicely with them."""
        qubit = Qubit()
        creg = ClassicalRegister(2, "c")
        case1 = QuantumCircuit([qubit], creg)
        case1.x(0)
        case2 = QuantumCircuit([qubit], creg)
        case2.y(0)

        body = QuantumCircuit([qubit], creg)
        body.switch(creg, [(0, case1), ((1, 2), case2)], [0], body.clbits)

        circuit = QuantumCircuit([qubit], creg)
        circuit.if_else((creg, 1), body.copy(), body, [0], body.clbits)

        test = dumps(circuit, experimental=ExperimentalFeatures.SWITCH_CASE_V1)
        expected = """\
OPENQASM 3;
include "stdgates.inc";
bit[2] c;
int switch_dummy;
int switch_dummy__generated0;
qubit _qubit0;
if (c == 1) {
  switch_dummy = c;
  switch (switch_dummy) {
    case 0: {
      x _qubit0;
    }
    break;
    case 1:
    case 2: {
      y _qubit0;
    }
    break;
  }
} else {
  switch_dummy__generated0 = c;
  switch (switch_dummy__generated0) {
    case 0: {
      x _qubit0;
    }
    break;
    case 1:
    case 2: {
      y _qubit0;
    }
    break;
  }
}
"""
        self.assertEqual(test, expected)

    def test_expr_target(self):
        """Simple test that the target of `switch` can be `Expr` nodes."""
        bits = [Qubit(), Clbit()]
        cr = ClassicalRegister(2, "cr")
        case0 = QuantumCircuit(1)
        case0.x(0)
        case1 = QuantumCircuit(1)
        case1.x(0)
        qc = QuantumCircuit(bits, cr)
        qc.switch(expr.logic_not(bits[1]), [(False, case0)], [0], [])
        qc.switch(expr.bit_and(cr, 3), [(3, case1)], [0], [])

        expected = """\
OPENQASM 3;
include "stdgates.inc";
bit _bit0;
bit[2] cr;
int switch_dummy;
int switch_dummy__generated0;
qubit _qubit0;
switch_dummy = !_bit0;
switch (switch_dummy) {
  case 0: {
    x _qubit0;
  }
  break;
}
switch_dummy__generated0 = cr & 3;
switch (switch_dummy__generated0) {
  case 3: {
    x _qubit0;
  }
  break;
}
"""
        test = dumps(qc, experimental=ExperimentalFeatures.SWITCH_CASE_V1)
        self.assertEqual(test, expected)


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
        with self.assertRaisesRegex(QASM3ExporterError, r"classical registers .* overlap"):
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
            QASM3ExporterError, "Exporting non-unitary instructions is not yet supported"
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
