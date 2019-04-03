# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=unused-import

"""Test Qiskit's inverse gate operation."""

import os
import tempfile
import unittest

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase
from qiskit.extensions.standard import TGate, SGate


class TestCircuitQasm(QiskitTestCase):
    """QuantumCircuit Qasm tests."""

    def test_circuit_qasm(self):
        """Test circuit qasm() method.
        """
        qr = QuantumRegister(1, 'qr')
        cr = ClassicalRegister(1, 'cr')
        circuit = QuantumCircuit(qr, cr)

        circuit.s(qr)
        circuit.s(qr)
        circuit.append(SGate().inverse(), qr[:])
        circuit.s(qr)
        circuit.append(TGate().inverse(), qr[:])
        circuit.t(qr)
        circuit.measure(qr, cr)
        expected_qasm = """OPENQASM 2.0;
include "qelib1.inc";
qreg qr[1];
creg cr[1];
s qr[0];
s qr[0];
sdg qr[0];
s qr[0];
tdg qr[0];
t qr[0];
measure qr[0] -> cr[0];\n"""
        self.assertEqual(circuit.qasm(), expected_qasm)


if __name__ == '__main__':
    unittest.main()
