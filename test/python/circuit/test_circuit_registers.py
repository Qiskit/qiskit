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
        sli = slice(0, 9, 2)
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        qc = QuantumCircuit(qr, cr)
        qc.h(qr[0:9:2])
        for i, index in enumerate(range(*sli.indices(sli.stop))):
            self.assertEqual(qc.data[i].qargs[0][1], index)

    def test_apply_barrier_to_slice(self):
        """test applying barrier to register slice"""
        n_qubits = 10
        qr = QuantumRegister(n_qubits)
        cr = ClassicalRegister(n_qubits)
        qc = QuantumCircuit(qr, cr)
        qc.barrier(qr)
        # barrier works a little different than normal gates for expansion
        # test full register
        self.assertEqual(len(qc.data), 1)
        self.assertEqual(qc.data[0].name, 'barrier')
        self.assertEqual(len(qc.data[0].qargs), n_qubits)
        for i in range(n_qubits):
            self.assertEqual(qc.data[0].qargs[i][1], i)
        # test slice
        n_qubits = 2
        qc = QuantumCircuit(qr, cr)
        qc.barrier(qr[0:n_qubits])
        self.log.info(qc.qasm())
        self.assertEqual(len(qc.data), 1)
        self.assertEqual(qc.data[0].name, 'barrier')
        self.assertEqual(len(qc.data[0].qargs), n_qubits)
        for i in range(n_qubits):
            self.assertEqual(qc.data[0].qargs[i][1], i)

    def test_apply_ccx_to_slice(self):
        """test applying ccx to register slice"""
        qcontrol = QuantumRegister(10)
        qcontrol2 = QuantumRegister(10)
        qtarget = QuantumRegister(5)
        qtarget2 = QuantumRegister(10)
        qc = QuantumCircuit(qcontrol, qtarget)
        # test slice with skip and full register target
        qc.ccx(qcontrol[1::2], qcontrol[0::2], qtarget)
        self.assertEqual(len(qc.data), 5)
        for i, ictl, gate in zip(range(len(qc.data)), range(0, 10, 2), qc.data):
            self.assertEqual(gate.name, 'ccx')
            self.assertEqual(len(gate.qargs), 3)
            self.assertIn(gate.qargs[0][1], [ictl, ictl+1])
            self.assertIn(gate.qargs[1][1], [ictl, ictl+1])
            self.assertEqual(gate.qargs[2][1], i)
        # test decrementing slice
        qc = QuantumCircuit(qcontrol, qtarget)
        qc.ccx(qcontrol[2:0:-1], qcontrol[4:6], qtarget[0:2])
        self.assertEqual(len(qc.data), 2)
        for gate, ictl1, ictl2, itgt in zip(qc.data, range(2, 0, -1),
                                            range(4, 6), range(0, 2)):
            self.assertEqual(gate.name, 'ccx')
            self.assertEqual(len(gate.qargs), 3)
            self.assertEqual(gate.qargs[0][1], ictl1)
            self.assertEqual(gate.qargs[1][1], ictl2)
            self.assertEqual(gate.qargs[2][1], itgt)
        # test register expansion in ccx
        qc = QuantumCircuit(qcontrol, qcontrol2, qtarget2)
        qc.ccx(qcontrol, qcontrol2, qtarget2)
        for i, gate in enumerate(qc.data):
            self.assertEqual(gate.name, 'ccx')
            self.assertEqual(len(gate.qargs), 3)
            self.assertEqual(gate.qargs[0][1], i)
            self.assertEqual(gate.qargs[1][1], i)
            self.assertEqual(gate.qargs[2][1], i)

    def test_ccx_multicontrol_single_target(self):
        """ccx with multi-qubit controls and single qubit target"""
        qcontrol1 = QuantumRegister(5)
        qcontrol2 = QuantumRegister(5)
        qtarget = QuantumRegister(2)
        qc = QuantumCircuit(qcontrol1, qcontrol2, qtarget)
        qc.ccx(qcontrol1[0:2], qcontrol2[0:2], qtarget[0])
        self.assertEqual(qc.data[0].qargs[-1], qc.data[1].qargs[-1])
        qc.ccx(qcontrol1[0:2], qcontrol2[0:2], qtarget[0:2])
        for i, gate in enumerate(qc.data[2:]):
            self.assertEqual(gate.qargs[0][1], i)

    def test_cswap_on_slice(self):
        """test applying cswap to register slice"""
        qr1 = QuantumRegister(10)
        qr2 = QuantumRegister(5)
        qc = QuantumCircuit(qr1, qr2)
        qc.cswap(qr2[3::-1], qr1[1:9:2], qr1[2:9:2])
        qc.cswap(qr2[0], qr1[1], qr1[2])
        qc.cswap([qr2[0]], [qr1[1]], [qr1[2]])
        self.assertRaises(qiskit.exceptions.QiskitError, qc.cswap, qr2[4::-1],
                          qr1[1:9:2], qr1[2:9:2])

    def test_apply_ccx_to_empty_slice(self):
        """test applying ccx to non-register raises"""
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        qc = QuantumCircuit(qr, cr)
        self.assertRaises(qiskit.exceptions.QiskitError, qc.ccx, qr[2:0], qr[4:2], qr[7:5])

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
        # test slice
        qc = QuantumCircuit(qr, cr)
        ctl_slice = slice(0, 2)
        tgt_slice = slice(2, 4)
        qc.ch(qr[ctl_slice], qr[tgt_slice])
        for gate, ictrl, itgt in zip(qc.data, range(0, 2), range(2, 4)):
            self.assertEqual(gate.name, 'ch')
            self.assertEqual(len(gate.qargs), 2)
            self.assertEqual(gate.qargs[0][1], ictrl)
            self.assertEqual(gate.qargs[1][1], itgt)
        # test single qubit args
        qc = QuantumCircuit(qr, cr)
        qc.ch(qr[0], qr[1])
        self.assertEqual(len(qc.data), 1)
        self.assertEqual(qc.data[0].name, 'ch')
        self.assertEqual(qc.data[0].qargs[0][1], 0)
        self.assertEqual(qc.data[0].qargs[1][1], 1)

    def test_measure_slice(self):
        """test measure slice"""
        qr = QuantumRegister(10)
        cr = ClassicalRegister(10)
        cr2 = ClassicalRegister(5)
        qc = QuantumCircuit(qr, cr)
        qc.measure(qr[0:2], cr[2:4])
        for gate, ictrl, itgt in zip(qc.data, range(0, 2), range(2, 4)):
            self.assertEqual(gate.name, 'measure')
            self.assertEqual(len(gate.qargs), 1)
            self.assertEqual(len(gate.cargs), 1)
            self.assertEqual(gate.qargs[0][1], ictrl)
            self.assertEqual(gate.cargs[0][1], itgt)
        # test single element slice
        qc = QuantumCircuit(qr, cr)
        qc.measure(qr[0:1], cr[2:3])
        for gate, ictrl, itgt in zip(qc.data, range(0, 1), range(2, 3)):
            self.assertEqual(gate.name, 'measure')
            self.assertEqual(len(gate.qargs), 1)
            self.assertEqual(len(gate.cargs), 1)
            self.assertEqual(gate.qargs[0][1], ictrl)
            self.assertEqual(gate.cargs[0][1], itgt)
        # test tuple
        qc = QuantumCircuit(qr, cr)
        qc.measure(qr[0], cr[2])
        self.assertEqual(len(qc.data), 1)
        op = qc.data[0]
        self.assertEqual(op.name, 'measure')
        self.assertEqual(len(op.qargs), 1)
        self.assertEqual(len(op.cargs), 1)
        self.assertTrue(isinstance(op.qargs[0], tuple))
        self.assertTrue(isinstance(op.cargs[0], tuple))
        self.assertEqual(gate.qargs[0][1], 0)
        self.assertEqual(gate.cargs[0][1], 2)
        # test full register
        qc = QuantumCircuit(qr, cr)
        qc.measure(qr, cr)
        for gate, ictrl, itgt in zip(qc.data, range(len(qr)), range(len(cr))):
            self.assertEqual(gate.name, 'measure')
            self.assertEqual(len(gate.qargs), 1)
            self.assertEqual(len(gate.cargs), 1)
            self.assertEqual(gate.qargs[0][1], ictrl)
            self.assertEqual(gate.cargs[0][1], itgt)
        # test mix slice full register
        qc = QuantumCircuit(qr, cr2)
        qc.measure(qr[::2], cr2)
        for gate, ictrl, itgt in zip(qc.data, range(0, 10, 2), range(len(cr2))):
            self.assertEqual(gate.name, 'measure')
            self.assertEqual(len(gate.qargs), 1)
            self.assertEqual(len(gate.cargs), 1)
            self.assertEqual(gate.qargs[0][1], ictrl)
            self.assertEqual(gate.cargs[0][1], itgt)
