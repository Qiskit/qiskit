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
from qiskit import BasicAer
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import execute
from qiskit import QiskitError
from qiskit.quantum_info import state_fidelity
from qiskit.test import QiskitTestCase


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

    def test_basic_slice(self):
        """simple slice test"""
        qr = QuantumRegister(5)
        cr = ClassicalRegister(5)
        self.assertEqual(len(qr[0:3]), 3)
        self.assertEqual(len(cr[0:3]), 3)

    def test_apply_gate_to_slice(self):
        """test applying gate to register slice"""
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        qc = QuantumCircuit(qr, cr)
        qc.x(qr[0])
        qc.y(qr)
        qc.x(qr[0:2])
        qc.h(qr[slice(0, 9, 2)])
        self.log.info(qc.qasm())

    def test_apply_barrier_to_slice(self):
        """test applying barrier to register slice"""
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        qc = QuantumCircuit(qr, cr)
        qc.barrier(qr)
        qc.barrier(qr[0:2])
        self.log.info(qc.qasm())

    def test_apply_ccx_to_slice(self):
        """test applying ccx to register slice"""
        qcontrol = QuantumRegister(10)
        qcontrol2 = QuantumRegister(10)
        qtarget = QuantumRegister(5)
        qtarget2 = QuantumRegister(10)
        qc = QuantumCircuit(qcontrol, qtarget, qcontrol2, qtarget2)
        qc.ccx(qcontrol[1::2], qcontrol[0::2], qtarget)
        qc.barrier(qcontrol)
        qc.ccx(qcontrol[2:0:-1], qcontrol[4:6], qtarget[0:2])
        qc.barrier(qcontrol)
        qc.ccx(qcontrol, qcontrol2, qtarget2)
        qc.barrier(qcontrol)
        qc.ccx(qcontrol[0:2], qcontrol[2:4], qcontrol[5:7])
        self.log.info(qc.qasm())

    def test_apply_ccx_to_non_register(self):
        """test applying ccx to non-register raises"""
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        qc = QuantumCircuit(qr, cr)
        self.assertRaises(qiskit.exceptions.QiskitError, qc.ccx, qc[0:2], qc[2:4], qc[5:7])

    def test_apply_cx_to_non_register(self):
        """test applying ccx to non-register raises"""
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        qc = QuantumCircuit(qr, cr)
        self.assertRaises(qiskit.exceptions.QiskitError, qc.cx, qc[0:2], qc[2:4])

    def test_apply_ch_to_slice(self):
        """test applying ch to slice"""
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        qc = QuantumCircuit(qr, cr)
        qc.ch(qr[0:2], qr[2:4])
        qc.barrier(qr)
        qc.ch(qr[0], qr[1])
        self.log.info(qc.qasm())
