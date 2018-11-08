# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring, redefined-builtin

import unittest
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import compile, execute
from qiskit import Aer
from .common import QiskitTestCase


class CompileSkipTranslationTest(QiskitTestCase):
    """Test compilation with skip translation."""

    def test_simple_compile(self):
        """Test compile with and without skip_transpiler."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        qc.u1(3.14, qr[0])
        qc.u2(3.14, 1.57, qr[0])
        qc.measure(qr, cr)
        backend = Aer.get_backend('qasm_simulator')
        rtrue = compile(qc, backend, skip_transpiler=True)
        rfalse = compile(qc, backend, skip_transpiler=False)
        self.assertEqual(rtrue.config, rfalse.config)
        self.assertEqual(rtrue.experiments, rfalse.experiments)

    def test_simple_execute(self):
        """Test execute with and without skip_transpiler."""
        qr = QuantumRegister(2)
        cr = ClassicalRegister(2)
        qc = QuantumCircuit(qr, cr)
        qc.u1(3.14, qr[0])
        qc.u2(3.14, 1.57, qr[0])
        qc.measure(qr, cr)
        backend = Aer.get_backend('qasm_simulator')
        rtrue = execute(qc, backend, seed=42, skip_transpiler=True).result()
        rfalse = execute(qc, backend, seed=42, skip_transpiler=False).result()
        self.assertEqual(rtrue.get_counts(), rfalse.get_counts())


if __name__ == '__main__':
    unittest.main()
