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
from qiskit import QISKitError
from qiskit.quantum_info import state_fidelity
from ..common import QiskitTestCase


class TestCircuitRegisters(QiskitTestCase):
    """QuantumCircuit Registers tests."""

    def test_qregs(self):
        """Test getting quantum registers from circuit.
        """
        qr1 = QuantumRegister(1)
        qr2 = QuantumRegister(2)
        qc = QuantumCircuit(qr1, qr2)
        q_regs = qc.qregs
        self.assertEqual(len(q_regs), 2)
        self.assertEqual(q_regs[qr1.name], qr1)
        self.assertEqual(q_regs[qr2.name], qr2)

    def test_cregs(self):
        """Test getting classical registers from circuit.
        """
        cr1 = ClassicalRegister(1)
        cr2 = ClassicalRegister(2)
        cr3 = ClassicalRegister(3)
        qc = QuantumCircuit(cr1, cr2, cr3)
        c_regs = qc.cregs
        self.assertEqual(len(c_regs), 3)
        self.assertEqual(c_regs[cr1.name], cr1)
        self.assertEqual(c_regs[cr2.name], cr2)
