# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for registers"""

import qiskit
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit.test import QiskitTestCase


class TestRegister(QiskitTestCase):
    """Test the mapper."""

    def test_basic_slice(self):
        """simple slice test"""
        qr = QuantumRegister(5)
        cr = ClassicalRegister(5)
        v1 = qr[0:3]
        v2 = cr[0:3]
        qc = QuantumCircuit(qr, cr)
        self.log.info(qc.qasm())

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

    def test_draw(self):
        """test drawing circuit with register slice"""
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        qc = QuantumCircuit(qr, cr)
        qc.y(qr)
        qc.x(qr[0:2])
        qc.h(qr[slice(0, 9, 2)])
        qc.draw()
