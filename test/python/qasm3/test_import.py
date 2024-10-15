# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-module-docstring,missing-class-docstring,missing-function-docstring

# Since the import is nearly entirely delegated to an external package, most of the testing is done
# there.  Here we need to test our wrapping behavior for base functionality and exceptions.  We
# don't want to get into a situation where updates to `qiskit_qasm3_import` breaks Terra's test
# suite due to too specific tests on the Terra side.

import os
import tempfile
import unittest
import warnings

from qiskit import qasm3
from qiskit.exceptions import ExperimentalWarning
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Qubit, Clbit
from qiskit.circuit import library as lib
from qiskit.utils import optionals
from test import QiskitTestCase  # pylint: disable=wrong-import-order


@unittest.skipUnless(optionals.HAS_QASM3_IMPORT, "need qiskit-qasm3-import for OpenQASM 3 imports")
class TestOldQASM3Import(QiskitTestCase):
    # These tests are for the `qiskit-qasm3-import` hooks, not the native one.

    def test_import_errors_converted(self):
        with self.assertRaises(qasm3.QASM3ImporterError):
            qasm3.loads("OPENQASM 3.0; qubit[2.5] q;")

    def test_loads_can_succeed(self):
        program = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] qr;
            bit[2] cr;
            h qr[0];
            cx qr[0], qr[1];
            cr[0] = measure qr[0];
            cr[1] = measure qr[1];
        """
        parsed = qasm3.loads(program)
        expected = QuantumCircuit(QuantumRegister(2, "qr"), ClassicalRegister(2, "cr"))
        expected.h(0)
        expected.cx(0, 1)
        expected.measure(0, 0)
        expected.measure(1, 1)
        self.assertEqual(parsed, expected)

    def test_load_can_succeed(self):
        program = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] qr;
            bit[2] cr;
            h qr[0];
            cx qr[0], qr[1];
            cr[0] = measure qr[0];
            cr[1] = measure qr[1];
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "bell.qasm")
            with open(tmp_path, "w") as fptr:
                fptr.write(program)
            parsed = qasm3.load(tmp_path)
        expected = QuantumCircuit(QuantumRegister(2, "qr"), ClassicalRegister(2, "cr"))
        expected.h(0)
        expected.cx(0, 1)
        expected.measure(0, 0)
        expected.measure(1, 1)
        self.assertEqual(parsed, expected)


class TestQASM3Import(QiskitTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._cm = warnings.catch_warnings()
        cls._cm.__enter__()
        # We're knowingly testing the experimental code.
        warnings.filterwarnings("ignore", category=ExperimentalWarning, module="qiskit.qasm3")

    @classmethod
    def tearDownClass(cls):
        cls._cm.__exit__(None, None, None)
        super().tearDownClass()

    def test_load_can_succeed(self):
        """Basic test of `load` - everything else we'll do via `loads` because it's easier."""
        program = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] qr;
            bit[2] cr;
            h qr[0];
            cx qr[0], qr[1];
            cr[0] = measure qr[0];
            cr[1] = measure qr[1];
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "bell.qasm")
            with open(tmp_path, "w") as fptr:
                fptr.write(program)
            parsed = qasm3.load_experimental(tmp_path)
        expected = QuantumCircuit(QuantumRegister(2, "qr"), ClassicalRegister(2, "cr"))
        expected.h(0)
        expected.cx(0, 1)
        expected.measure(0, 0)
        expected.measure(1, 1)
        self.assertEqual(parsed, expected)

    def test_simple_loose_bits(self):
        program = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit q0;
            qubit q1;
            bit c0;
            bit c1;

            h q0;
            cx q0, q1;
            c0 = measure q0;
            c1 = measure q1;
        """
        parsed = qasm3.loads_experimental(program)
        expected = QuantumCircuit([Qubit(), Qubit(), Clbit(), Clbit()])
        expected.h(0)
        expected.cx(0, 1)
        expected.measure(0, 0)
        expected.measure(1, 1)
        self.assertEqual(parsed, expected)

    def test_all_stdlib_gates(self):
        # Notably this doesn't include negative floating-point values yet.  Can be fixed after:
        #   https://github.com/Qiskit/openqasm3_parser/issues/81
        program = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[3] q;
            p(0.5) q[0];
            x q[0];
            y q[0];
            z q[0];
            h q[0];
            s q[0];
            sdg q[0];
            t q[0];
            tdg q[0];
            sx q[0];
            rx(0.5) q[0];
            ry(0.5) q[0];
            rz(0.5) q[0];
            cx q[0], q[1];
            cy q[0], q[1];
            cz q[0], q[1];
            cp(1.5) q[0], q[1];
            crx(0.25) q[0], q[1];
            cry(0.75) q[0], q[1];
            crz(0.5) q[0], q[1];
            ch q[0], q[1];
            swap q[0], q[1];
            ccx q[0], q[1], q[2];
            cswap q[0], q[1], q[2];
            cu(0.25, 0.5, 0.75, 1.0) q[0], q[1];
            CX q[0], q[1];
            phase(0.5) q[0];
            cphase(0.5) q[0], q[1];
            id q[0];
            u1(0.5) q[0];
            u2(0.25, 0.5) q[0];
            u3(0.25, 0.5, 0.75) q[0];
        """
        parsed = qasm3.loads_experimental(program)
        expected = QuantumCircuit(QuantumRegister(3, "q"))
        expected.p(0.5, 0)
        expected.x(0)
        expected.y(0)
        expected.z(0)
        expected.h(0)
        expected.s(0)
        expected.sdg(0)
        expected.t(0)
        expected.tdg(0)
        expected.sx(0)
        expected.rx(0.5, 0)
        expected.ry(0.5, 0)
        expected.rz(0.5, 0)
        expected.cx(0, 1)
        expected.cy(0, 1)
        expected.cz(0, 1)
        expected.cp(1.5, 0, 1)
        expected.crx(0.25, 0, 1)
        expected.cry(0.75, 0, 1)
        expected.crz(0.5, 0, 1)
        expected.ch(0, 1)
        expected.swap(0, 1)
        expected.ccx(0, 1, 2)
        expected.cswap(0, 1, 2)
        expected.cu(0.25, 0.5, 0.75, 1, 0, 1)
        expected.cx(0, 1)
        expected.p(0.5, 0)
        expected.cp(0.5, 0, 1)
        expected.id(0)
        expected.append(lib.U1Gate(0.5), [0], [])
        expected.append(lib.U2Gate(0.25, 0.5), [0], [])
        expected.append(lib.U3Gate(0.25, 0.5, 0.75), [0], [])
        self.assertEqual(parsed, expected)

    def test_barrier(self):
        program = """
            OPENQASM 3.0;
            qubit[2] a;
            qubit b;
            qubit[5] c;
            barrier b;
            barrier a;
            barrier a, b;
            barrier a[0], c[2];
            barrier c[{3, 1, 4}][{2, 0, 1}][0], b;
        """
        parsed = qasm3.loads_experimental(program)
        a, b, c = QuantumRegister(2, "a"), Qubit(), QuantumRegister(5, "c")
        expected = QuantumCircuit(a, [b], c)
        expected.barrier(b)
        expected.barrier(a)
        expected.barrier(a, b)
        expected.barrier(a[0], c[2])
        expected.barrier(c[4], b)
        self.assertEqual(parsed, expected)

    def test_measure(self):
        program = """
            OPENQASM 3.0;
            qubit[2] q0;
            qubit q1;
            qubit[5] q2;
            bit[2] c0;
            bit c1;
            bit[5] c2;
            c0 = measure q0;
            c0[0] = measure q0[0];
            c1 = measure q1;
            c2[{3, 1, 4}] = measure q2[{4, 2, 3}];
        """
        parsed = qasm3.loads_experimental(program)
        q0, q1, q2 = QuantumRegister(2, "q0"), Qubit(), QuantumRegister(5, "q2")
        c0, c1, c2 = ClassicalRegister(2, "c0"), Clbit(), ClassicalRegister(5, "c2")
        expected = QuantumCircuit(q0, [q1], q2, c0, [c1], c2)
        expected.measure(q0, c0)
        expected.measure(q0[0], c0[0])
        expected.measure(q1, c1)
        expected.measure(q2[[4, 2, 3]], c2[[3, 1, 4]])
        self.assertEqual(parsed, expected)

    def test_override_custom_gate_parameterless(self):
        program = """
            OPENQASM 3.0;
            gate my_gate a, b {}
            qubit[2] q;
            my_gate q[0], q[1];
            my_gate q[1], q[0];
        """
        parsed = qasm3.loads_experimental(
            program, custom_gates=[qasm3.CustomGate(lib.CXGate, "my_gate", 0, 2)]
        )
        expected = QuantumCircuit(QuantumRegister(2, "q"))
        expected.cx(0, 1)
        expected.cx(1, 0)
        self.assertEqual(parsed, expected)

    def test_override_custom_gate_parametric(self):
        program = """
            OPENQASM 3.0;
            gate my_crx(ang) a, b {}
            qubit[2] q;
            my_crx(0.5) q[0], q[1];
            my_crx(1.5) q[1], q[0];
        """
        parsed = qasm3.loads_experimental(
            program, custom_gates=[qasm3.CustomGate(lib.CRXGate, "my_crx", 1, 2)]
        )
        expected = QuantumCircuit(QuantumRegister(2, "q"))
        expected.crx(0.5, 0, 1)
        expected.crx(1.5, 1, 0)
        self.assertEqual(parsed, expected)

    def test_set_include_path(self):
        include = """
            gate my_gate a {}
        """
        program = """
            OPENQASM 3.0;
            include "my_include.qasm";
            qubit q0;
            my_gate q0;
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = os.path.join(tmp_dir, "my_include.qasm")
            with open(tmp_path, "w") as fptr:
                fptr.write(include)
            # Can't test for failed import yet due to:
            #   https://github.com/Qiskit/openqasm3_parser/issues/74
            parsed = qasm3.loads_experimental(
                program,
                custom_gates=[qasm3.CustomGate(lib.XGate, "my_gate", 0, 1)],
                include_path=[tmp_dir],
            )
        expected = QuantumCircuit([Qubit()])
        expected.x(0)
        self.assertEqual(parsed, expected)

    def test_gate_broadcast(self):
        program = """
            OPENQASM 3.0;
            include "stdgates.inc";
            qubit[2] q0;
            qubit q1;
            qubit[2] q2;
            h q0;
            cx q0, q1;
            cx q0[0], q2;
            cx q0, q2;
            ccx q0[{1, 0}], q1, q2;
        """
        parsed = qasm3.loads_experimental(program)
        q0, q1, q2 = QuantumRegister(2, "q0"), Qubit(), QuantumRegister(2, "q2")
        expected = QuantumCircuit(q0, [q1], q2)
        expected.h(q0[0])
        expected.h(q0[1])
        #
        expected.cx(q0[0], q1)
        expected.cx(q0[1], q1)
        #
        expected.cx(q0[0], q2[0])
        expected.cx(q0[0], q2[1])
        #
        expected.cx(q0[0], q2[0])
        expected.cx(q0[1], q2[1])
        #
        expected.ccx(q0[1], q1, q2[0])
        expected.ccx(q0[0], q1, q2[1])
        self.assertEqual(parsed, expected)

    def test_custom_gate_inspectable(self):
        """Test that the `CustomGate` object can be inspected programmatically after creation."""
        custom = qasm3.CustomGate(lib.CXGate, "cx", 0, 2)
        self.assertEqual(custom.name, "cx")
        self.assertEqual(custom.num_params, 0)
        self.assertEqual(custom.num_qubits, 2)

        self.assertIsInstance(qasm3.STDGATES_INC_GATES[0], qasm3.CustomGate)
        stdgates = {
            gate.name: (gate.num_params, gate.num_qubits) for gate in qasm3.STDGATES_INC_GATES
        }
        self.assertEqual(stdgates["rx"], (1, 1))
        self.assertEqual(stdgates["cphase"], (1, 2))
