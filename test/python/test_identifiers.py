# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,broad-except

"""Non-string identifiers for circuit and record identifiers test"""

import unittest

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import QISKitError
# pylint: disable=redefined-builtin
from qiskit import compile, Aer
from .common import QiskitTestCase, requires_cpp_simulator


class TestQobjIdentifiers(QiskitTestCase):
    """Check the Qobj compiled for different backends create names properly"""

    def setUp(self):
        qr = QuantumRegister(2, name="qr2")
        cr = ClassicalRegister(2, name=None)
        qc = QuantumCircuit(qr, cr, name="qc10")
        qc.h(qr[0])
        qc.measure(qr[0], cr[0])
        self.qr_name = qr.name
        self.cr_name = cr.name
        self.circuits = [qc]

    def test_aer_qasm_simulator_py(self):
        backend = Aer.get_backend('qasm_simulator_py')
        qobj = compile(self.circuits, backend=backend)
        exp = qobj.experiments[0]
        c_qasm = exp.header.compiled_circuit_qasm
        self.assertIn(self.qr_name, map(lambda x: x[0], exp.header.qubit_labels))
        self.assertIn(self.qr_name, c_qasm)
        self.assertIn(self.cr_name, map(lambda x: x[0], exp.header.clbit_labels))
        self.assertIn(self.cr_name, c_qasm)

    @requires_cpp_simulator
    def test_aer_clifford_simulator(self):
        backend = Aer.get_backend('clifford_simulator')
        qobj = compile(self.circuits, backend=backend)
        exp = qobj.experiments[0]
        c_qasm = exp.header.compiled_circuit_qasm
        self.assertIn(self.qr_name, map(lambda x: x[0], exp.header.qubit_labels))
        self.assertIn(self.qr_name, c_qasm)
        self.assertIn(self.cr_name, map(lambda x: x[0], exp.header.clbit_labels))
        self.assertIn(self.cr_name, c_qasm)

    @requires_cpp_simulator
    def test_aer_qasm_simulator(self):
        backend = Aer.get_backend('qasm_simulator')
        qobj = compile(self.circuits, backend=backend)
        exp = qobj.experiments[0]
        c_qasm = exp.header.compiled_circuit_qasm
        self.assertIn(self.qr_name, map(lambda x: x[0], exp.header.qubit_labels))
        self.assertIn(self.qr_name, c_qasm)
        self.assertIn(self.cr_name, map(lambda x: x[0], exp.header.clbit_labels))
        self.assertIn(self.cr_name, c_qasm)

    def test_aer_unitary_simulator(self):
        backend = Aer.get_backend('unitary_simulator')
        qobj = compile(self.circuits, backend=backend)
        exp = qobj.experiments[0]
        c_qasm = exp.header.compiled_circuit_qasm
        self.assertIn(self.qr_name, map(lambda x: x[0], exp.header.qubit_labels))
        self.assertIn(self.qr_name, c_qasm)
        self.assertIn(self.cr_name, map(lambda x: x[0], exp.header.clbit_labels))
        self.assertIn(self.cr_name, c_qasm)


class TestAnonymousIds(QiskitTestCase):
    """Test the anonymous use of registers.
    """

    def test_create_anonymous_classical_register(self):
        """ClassicalRegister with no name."""
        cr = ClassicalRegister(size=3)
        self.assertIsInstance(cr, ClassicalRegister)

    def test_create_anonymous_quantum_register(self):
        """QuantumRegister with no name."""
        qr = QuantumRegister(size=3)
        self.assertIsInstance(qr, QuantumRegister)

    def test_create_anonymous_classical_registers(self):
        """Several ClassicalRegister with no name."""
        cr1 = ClassicalRegister(size=3)
        cr2 = ClassicalRegister(size=3)
        self.assertNotEqual(cr1.name, cr2.name)

    def test_create_anonymous_quantum_registers(self):
        """Several QuantumRegister with no name."""
        qr1 = QuantumRegister(size=3)
        qr2 = QuantumRegister(size=3)
        self.assertNotEqual(qr1.name, qr2.name)

    def test_create_anonymous_mixed_registers(self):
        """Several Registers with no name."""
        cr0 = ClassicalRegister(size=3)
        qr0 = QuantumRegister(size=3)
        # Get the current index counte of the registers
        cr_index = int(cr0.name[1:])
        qr_index = int(qr0.name[1:])

        cr1 = ClassicalRegister(size=3)
        _ = QuantumRegister(size=3)
        qr2 = QuantumRegister(size=3)

        # Check that the counters for each kind are incremented separately.
        cr_current = int(cr1.name[1:])
        qr_current = int(qr2.name[1:])
        self.assertEqual(cr_current, cr_index + 1)
        self.assertEqual(qr_current, qr_index + 2)

    def test_create_circuit_noname(self):
        """Create_circuit with no name."""
        qr = QuantumRegister(size=3)
        cr = ClassicalRegister(size=3)
        qc = QuantumCircuit(qr, cr)
        self.assertIsInstance(qc, QuantumCircuit)


class TestInvalidIds(QiskitTestCase):
    """Circuits and records with invalid IDs"""

    def test_invalid_type_circuit_name(self):
        """QuantumCircuit() with invalid type name."""
        qr = QuantumRegister(size=3)
        cr = ClassicalRegister(size=3)
        self.assertRaises(QISKitError, QuantumCircuit, qr, cr, name=1)

    def test_invalid_type_qr_name(self):
        """QuantumRegister() with an invalid type name."""
        self.assertRaises(QISKitError, QuantumRegister, size=3, name=1)

    def test_invalid_type_cr_name(self):
        """ClassicalRegister() with an invalid type name."""
        self.assertRaises(QISKitError, ClassicalRegister, size=3, name=1)

    def test_invalid_qasmname_qr(self):
        """QuantumRegister() with invalid name."""
        self.assertRaises(QISKitError, QuantumRegister, size=3, name='Qr')

    def test_invalid_qasmname_cr(self):
        """ClassicalRegister() with invalid name."""
        self.assertRaises(QISKitError, ClassicalRegister, size=3, name='Cr')


if __name__ == '__main__':
    unittest.main(verbosity=2)
