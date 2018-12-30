# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring,redefined-builtin

"""Non-string identifiers for circuit and record identifiers test"""

import unittest

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit import compile, BasicAer
from qiskit.test import QiskitTestCase


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

    def test_builtin_qasm_simulator_py(self):
        backend = BasicAer.get_backend('qasm_simulator')
        qobj = compile(self.circuits, backend=backend)
        exp = qobj.experiments[0]
        c_qasm = exp.header.compiled_circuit_qasm
        self.assertIn(self.qr_name, map(lambda x: x[0], exp.header.qubit_labels))
        self.assertIn(self.qr_name, c_qasm)
        self.assertIn(self.cr_name, map(lambda x: x[0], exp.header.clbit_labels))
        self.assertIn(self.cr_name, c_qasm)

    def test_builtin_qasm_simulator(self):
        backend = BasicAer.get_backend('qasm_simulator')
        qobj = compile(self.circuits, backend=backend)
        exp = qobj.experiments[0]
        c_qasm = exp.header.compiled_circuit_qasm
        self.assertIn(self.qr_name, map(lambda x: x[0], exp.header.qubit_labels))
        self.assertIn(self.qr_name, c_qasm)
        self.assertIn(self.cr_name, map(lambda x: x[0], exp.header.clbit_labels))
        self.assertIn(self.cr_name, c_qasm)

    def test_builtin_unitary_simulator_py(self):
        backend = BasicAer.get_backend('unitary_simulator')
        qobj = compile(self.circuits, backend=backend)
        exp = qobj.experiments[0]
        c_qasm = exp.header.compiled_circuit_qasm
        self.assertIn(self.qr_name, map(lambda x: x[0], exp.header.qubit_labels))
        self.assertIn(self.qr_name, c_qasm)
        self.assertIn(self.cr_name, map(lambda x: x[0], exp.header.clbit_labels))
        self.assertIn(self.cr_name, c_qasm)


if __name__ == '__main__':
    unittest.main(verbosity=2)
