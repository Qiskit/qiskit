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
# there.  Here we need to test our wrapping behaviour for base functionality and exceptions.  We
# don't want to get into a situation where updates to `qiskit_qasm3_import` breaks Terra's test
# suite due to too specific tests on the Terra side.

import os
import tempfile
import unittest

from qiskit import qasm3
from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.test import QiskitTestCase
from qiskit.utils import optionals


class TestQASM3Import(QiskitTestCase):
    @unittest.skipUnless(
        optionals.HAS_QASM3_IMPORT, "need qiskit-qasm3-import for OpenQASM 3 imports"
    )
    def test_import_errors_converted(self):
        with self.assertRaises(qasm3.QASM3ImporterError):
            qasm3.loads("OPENQASM 3.0; qubit[2.5] q;")

    @unittest.skipUnless(
        optionals.HAS_QASM3_IMPORT, "need qiskit-qasm3-import for OpenQASM 3 imports"
    )
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

    @unittest.skipUnless(
        optionals.HAS_QASM3_IMPORT, "need qiskit-qasm3-import for OpenQASM 3 imports"
    )
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
