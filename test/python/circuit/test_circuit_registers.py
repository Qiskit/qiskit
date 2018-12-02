# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=unused-import

"""Test Qiskit's QuantumCircuit class."""

import os
import tempfile
import unittest

import qiskit.extensions.simulator
from qiskit import Aer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit import QiskitError
from qiskit.quantum_info import state_fidelity
from ..common import QiskitTestCase


class TestCircuitRegisters(QiskitTestCase):
    """QuantumCircuit Registers tests."""

    def test_qregs(self):
        """Test getting quantum registers from circuit.
        """
        qr1 = QuantumRegister(10, "q")
        self.assertEqual(qr1.name, "q")
        self.assertEqual(qr1.size, 10)
        self.assertEqual(type(qr1), QuantumRegister)

    def test_cregs(self):
        """Test getting quantum registers from circuit.
        """
        cr1 = ClassicalRegister(10, "c")
        self.assertEqual(cr1.name, "c")
        self.assertEqual(cr1.size, 10)
        self.assertEqual(type(cr1), ClassicalRegister)

    def test_reg_equal(self):
        """Test getting quantum registers from circuit.
        """
        qr1 = QuantumRegister(1, "q")
        qr2 = QuantumRegister(2, "q")
        cr1 = ClassicalRegister(1, "q")

        self.assertEqual(qr1, qr1)
        self.assertNotEqual(qr1, qr2)
        self.assertNotEqual(qr1, cr1)

    def test_qregs_circuit(self):
        """Test getting quantum registers from circuit.
        """
        qr1 = QuantumRegister(1)
        qr2 = QuantumRegister(2)
        qc = QuantumCircuit(qr1, qr2)
        q_regs = qc.qregs
        self.assertEqual(len(q_regs), 2)
        self.assertEqual(q_regs[0], qr1)
        self.assertEqual(q_regs[1], qr2)

    def test_cregs_circuit(self):
        """Test getting classical registers from circuit.
        """
        cr1 = ClassicalRegister(1)
        cr2 = ClassicalRegister(2)
        cr3 = ClassicalRegister(3)
        qc = QuantumCircuit(cr1, cr2, cr3)
        c_regs = qc.cregs
        self.assertEqual(len(c_regs), 3)
        self.assertEqual(c_regs[0], cr1)
        self.assertEqual(c_regs[1], cr2)
