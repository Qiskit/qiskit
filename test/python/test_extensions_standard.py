# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=missing-docstring

import unittest

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.qasm import pi
from qiskit._qiskiterror import QISKitError

from .common import QiskitTestCase


class TestStandard1Q(QiskitTestCase):
    """Standard Extension Test. Gates with a single Qubit"""

    def setUp(self):
        self.qr = QuantumRegister(3, "q")
        self.qr2 = QuantumRegister(3, "r")
        self.cr = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(self.qr, self.qr2, self.cr)

    def test_barrier(self):
        self.circuit.barrier(self.qr[1])
        self.assertEqual(len(self.circuit), 1)
        self.assertEqual(self.circuit[0].name, 'barrier')
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_barrier_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.barrier, self.cr[0])
        self.assertRaises(QISKitError, qc.barrier, self.cr)
        self.assertRaises(QISKitError, qc.barrier, (self.qr, 3))
        self.assertRaises(QISKitError, qc.barrier, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.barrier, 0)

    def test_barrier_reg(self):
        self.circuit.barrier(self.qr)
        self.assertEqual(len(self.circuit), 1)
        self.assertEqual(self.circuit[0].name, 'barrier')
        self.assertEqual(self.circuit[0].qargs,
                         [self.qr[0], self.qr[1], self.qr[2]])

    def test_barrier_none(self):
        self.circuit.barrier()
        self.assertEqual(len(self.circuit), 1)
        self.assertEqual(self.circuit[0].name, 'barrier')
        self.assertEqual(self.circuit[0].qargs,
                         [self.qr[0], self.qr[1], self.qr[2],
                          self.qr2[0], self.qr2[1], self.qr2[2]])

    def test_ccx(self):
        self.circuit.ccx(self.qr[0], self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].name, 'ccx')
        self.assertEqual(self.circuit[0].qargs,
                         [self.qr[0], self.qr[1], self.qr[2]])

    def test_ccx_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.ccx, self.cr[0], self.cr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.ccx, self.qr[0], self.qr[0], self.qr[2])
        self.assertRaises(QISKitError, qc.ccx, 0, self.qr[0], self.qr[2])
        self.assertRaises(QISKitError, qc.ccx, (self.qr, 3), self.qr[1], self.qr[2])
        self.assertRaises(QISKitError, qc.ccx, self.cr, self.qr, self.qr)
        self.assertRaises(QISKitError, qc.ccx, 'a', self.qr[1], self.qr[2])

    def test_ch(self):
        self.circuit.ch(self.qr[0], self.qr[1])
        self.assertEqual(self.circuit[0].name, 'ch')
        self.assertEqual(self.circuit[0].qargs, [self.qr[0], self.qr[1]])

    def test_ch_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.ch, self.cr[0], self.cr[1])
        self.assertRaises(QISKitError, qc.ch, self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.ch, 0, self.qr[0])
        self.assertRaises(QISKitError, qc.ch, (self.qr, 3), self.qr[0])
        self.assertRaises(QISKitError, qc.ch, self.cr, self.qr)
        self.assertRaises(QISKitError, qc.ch, 'a', self.qr[1])

    def test_crz(self):
        self.circuit.crz(1, self.qr[0], self.qr[1])
        self.assertEqual(self.circuit[0].name, 'crz')
        self.assertEqual(self.circuit[0].param, [1])
        self.assertEqual(self.circuit[0].qargs, [self.qr[0], self.qr[1]])

    def test_crz_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.crz, 0, self.cr[0], self.cr[1])
        self.assertRaises(QISKitError, qc.crz, 0, self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.crz, 0, 0, self.qr[0])
        # TODO self.assertRaises(QISKitError, qc.crz, self.qr[2], self.qr[1], self.qr[0])
        self.assertRaises(QISKitError, qc.crz, 0, self.qr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.crz, 0, (self.qr, 3), self.qr[1])
        self.assertRaises(QISKitError, qc.crz, 0, self.cr, self.qr)
        # TODO self.assertRaises(QISKitError, qc.crz, 'a', self.qr[1], self.qr[2])

    def test_cswap(self):
        self.circuit.cswap(self.qr[0], self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].name, 'cswap')
        self.assertEqual(self.circuit[0].param, [])
        self.assertEqual(self.circuit[0].qargs, [self.qr[0], self.qr[1], self.qr[2]])

    def test_cswap_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.cswap, self.cr[0], self.cr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.cswap, self.qr[1], self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.cswap, self.qr[1], 0, self.qr[0])
        self.assertRaises(QISKitError, qc.cswap, self.cr[0], self.cr[1], self.qr[0])
        self.assertRaises(QISKitError, qc.cswap, self.qr[0], self.qr[0], self.qr[1])
        self.assertRaises(QISKitError, qc.cswap, 0, self.qr[0], self.qr[1])
        self.assertRaises(QISKitError, qc.cswap, (self.qr, 3), self.qr[0], self.qr[1])
        self.assertRaises(QISKitError, qc.cswap, self.cr, self.qr[0], self.qr[1])
        self.assertRaises(QISKitError, qc.cswap, 'a', self.qr[1], self.qr[2])

    def test_cu1(self):
        self.circuit.cu1(1, self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].name, 'cu1')
        self.assertEqual(self.circuit[0].param, [1])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1], self.qr[2]])

    def test_cu1_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.cu1, self.cr[0], self.cr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.cu1, 1, self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.cu1, self.qr[1], 0, self.qr[0])
        self.assertRaises(QISKitError, qc.cu1, 0, self.cr[0], self.cr[1])
        self.assertRaises(QISKitError, qc.cu1, 0, self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.cu1, 0, 0, self.qr[0])
        # TODO self.assertRaises(QISKitError, qc.cu1, self.qr[2], self.qr[1], self.qr[0])
        self.assertRaises(QISKitError, qc.cu1, 0, self.qr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.cu1, 0, (self.qr, 3), self.qr[1])
        self.assertRaises(QISKitError, qc.cu1, 0, self.cr, self.qr)
        # TODO self.assertRaises(QISKitError, qc.cu1, 'a', self.qr[1], self.qr[2])

    def test_cu3(self):
        self.circuit.cu3(1, 2, 3, self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].name, 'cu3')
        self.assertEqual(self.circuit[0].param, [1, 2, 3])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1], self.qr[2]])

    def test_cu3_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.cu3, 0, 0, self.qr[0], self.qr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.cu3, 0, 0, 0, self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.cu3, 0, 0, self.qr[1], 0, self.qr[0])
        self.assertRaises(QISKitError, qc.cu3, 0, 0, 0, self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.cu3, 0, 0, 0, 0, self.qr[0])
        self.assertRaises(QISKitError, qc.cu3, 0, 0, 0, (self.qr, 3), self.qr[1])
        self.assertRaises(QISKitError, qc.cu3, 0, 0, 0, self.cr, self.qr)
        # TODO self.assertRaises(QISKitError, qc.cu3, 0, 0, 'a', self.qr[1], self.qr[2])

    def test_cx(self):
        self.circuit.cx(self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].name, 'cx')
        self.assertEqual(self.circuit[0].param, [])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1], self.qr[2]])

    def test_cx_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.cx, self.cr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.cx, self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.cx, 0, self.qr[0])
        self.assertRaises(QISKitError, qc.cx, (self.qr, 3), self.qr[0])
        self.assertRaises(QISKitError, qc.cx, self.cr, self.qr)
        self.assertRaises(QISKitError, qc.cx, 'a', self.qr[1])

    def test_cxbase(self):
        self.circuit.cx_base(self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].name, 'CX')
        self.assertEqual(self.circuit[0].param, [])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1], self.qr[2]])

    def test_cxbase_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.cx_base, self.cr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.cx_base, self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.cx_base, 0, self.qr[0])
        self.assertRaises(QISKitError, qc.cx_base, (self.qr, 3), self.qr[0])
        self.assertRaises(QISKitError, qc.cx_base, self.cr, self.qr)
        self.assertRaises(QISKitError, qc.cx_base, 'a', self.qr[1])

    def test_cy(self):
        self.circuit.cy(self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].name, 'cy')
        self.assertEqual(self.circuit[0].param, [])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1], self.qr[2]])

    def test_cy_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.cy, self.cr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.cy, self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.cy, 0, self.qr[0])
        self.assertRaises(QISKitError, qc.cy, (self.qr, 3), self.qr[0])
        self.assertRaises(QISKitError, qc.cy, self.cr, self.qr)
        self.assertRaises(QISKitError, qc.cy, 'a', self.qr[1])

    def test_cz(self):
        self.circuit.cz(self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].name, 'cz')
        self.assertEqual(self.circuit[0].param, [])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1], self.qr[2]])

    def test_cz_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.cz, self.cr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.cz, self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.cz, 0, self.qr[0])
        self.assertRaises(QISKitError, qc.cz, (self.qr, 3), self.qr[0])
        self.assertRaises(QISKitError, qc.cz, self.cr, self.qr)
        self.assertRaises(QISKitError, qc.cz, 'a', self.qr[1])

    def test_h(self):
        self.circuit.h(self.qr[1])
        self.assertEqual(self.circuit[0].name, 'h')
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_h_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.h, self.cr[0])
        self.assertRaises(QISKitError, qc.h, self.cr)
        self.assertRaises(QISKitError, qc.h, (self.qr, 3))
        self.assertRaises(QISKitError, qc.h, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.h, 0)

    def test_h_reg(self):
        instruction_set = self.circuit.h(self.qr)
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'h')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])

    def test_h_reg_inv(self):
        instruction_set = self.circuit.h(self.qr).inverse()
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'h')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])

    def test_iden(self):
        self.circuit.iden(self.qr[1])
        self.assertEqual(self.circuit[0].name, 'id')
        self.assertEqual(self.circuit[0].param, [])

    def test_iden_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.iden, self.cr[0])
        self.assertRaises(QISKitError, qc.iden, self.cr)
        self.assertRaises(QISKitError, qc.iden, (self.qr, 3))
        self.assertRaises(QISKitError, qc.iden, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.iden, 0)

    def test_iden_reg(self):
        instruction_set = self.circuit.iden(self.qr)
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'id')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])

    def test_iden_reg_inv(self):
        instruction_set = self.circuit.iden(self.qr).inverse()
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'id')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])

    def test_rx(self):
        self.circuit.rx(1, self.qr[1])
        self.assertEqual(self.circuit[0].name, 'rx')
        self.assertEqual(self.circuit[0].param, [1])

    def test_rx_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.rx, self.cr[0], self.cr[1])
        self.assertRaises(QISKitError, qc.rx, self.qr[1], 0)
        self.assertRaises(QISKitError, qc.rx, 0, self.cr[0])
        self.assertRaises(QISKitError, qc.rx, 0, 0)
        # TODO self.assertRaises(QISKitError, qc.rx, self.qr[2], self.qr[1])
        self.assertRaises(QISKitError, qc.rx, 0, (self.qr, 3))
        self.assertRaises(QISKitError, qc.rx, 0, self.cr)
        # TODO self.assertRaises(QISKitError, qc.rx, 'a', self.qr[1])
        self.assertRaises(QISKitError, qc.rx, 0, 'a')

    def test_rx_reg(self):
        instruction_set = self.circuit.rx(1, self.qr)
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'rx')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [1])

    def test_rx_reg_inv(self):
        instruction_set = self.circuit.rx(1, self.qr).inverse()
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'rx')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [-1])

    def test_rx_pi(self):
        qc = self.circuit
        qc.rx(pi / 2, self.qr[1])
        self.assertEqual(self.circuit[0].name, 'rx')
        self.assertEqual(self.circuit[0].param, [pi / 2])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_ry(self):
        self.circuit.ry(1, self.qr[1])
        self.assertEqual(self.circuit[0].name, 'ry')
        self.assertEqual(self.circuit[0].param, [1])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_ry_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.ry, self.cr[0], self.cr[1])
        self.assertRaises(QISKitError, qc.ry, self.qr[1], 0)
        self.assertRaises(QISKitError, qc.ry, 0, self.cr[0])
        self.assertRaises(QISKitError, qc.ry, 0, 0)
        # TODO self.assertRaises(QISKitError, qc.ry, self.qr[2], self.qr[1])
        self.assertRaises(QISKitError, qc.ry, 0, (self.qr, 3))
        self.assertRaises(QISKitError, qc.ry, 0, self.cr)
        # TODO self.assertRaises(QISKitError, qc.ry, 'a', self.qr[1])
        self.assertRaises(QISKitError, qc.ry, 0, 'a')

    def test_ry_reg(self):
        instruction_set = self.circuit.ry(1, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'ry')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [1])

    def test_ry_reg_inv(self):
        instruction_set = self.circuit.ry(1, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'ry')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [-1])

    def test_ry_pi(self):
        qc = self.circuit
        qc.ry(pi / 2, self.qr[1])
        self.assertEqual(self.circuit[0].name, 'ry')
        self.assertEqual(self.circuit[0].param, [pi / 2])

    def test_rz(self):
        self.circuit.rz(1, self.qr[1])
        self.assertEqual(self.circuit[0].name, 'rz')
        self.assertEqual(self.circuit[0].param, [1])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_rz_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.rz, self.cr[0], self.cr[1])
        self.assertRaises(QISKitError, qc.rz, self.qr[1], 0)
        self.assertRaises(QISKitError, qc.rz, 0, self.cr[0])
        self.assertRaises(QISKitError, qc.rz, 0, 0)
        # TODO self.assertRaises(QISKitError, qc.rz, self.qr[2], self.qr[1])
        self.assertRaises(QISKitError, qc.rz, 0, (self.qr, 3))
        self.assertRaises(QISKitError, qc.rz, 0, self.cr)
        # TODO self.assertRaises(QISKitError, qc.rz, 'a', self.qr[1])
        self.assertRaises(QISKitError, qc.rz, 0, 'a')

    def test_rz_reg(self):
        instruction_set = self.circuit.rz(1, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'rz')
        self.assertEqual(instruction_set.instructions[2].param, [1])

    def test_rz_reg_inv(self):
        instruction_set = self.circuit.rz(1, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'rz')
        self.assertEqual(instruction_set.instructions[2].param, [-1])

    def test_rz_pi(self):
        self.circuit.rz(pi / 2, self.qr[1])
        self.assertEqual(self.circuit[0].name, 'rz')
        self.assertEqual(self.circuit[0].param, [pi / 2])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_s(self):
        self.circuit.s(self.qr[1])
        self.assertEqual(self.circuit[0].name, 's')
        self.assertEqual(self.circuit[0].param, [])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_s_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.s, self.cr[0])
        self.assertRaises(QISKitError, qc.s, self.cr)
        self.assertRaises(QISKitError, qc.s, (self.qr, 3))
        self.assertRaises(QISKitError, qc.s, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.s, 0)

    def test_s_reg(self):
        instruction_set = self.circuit.s(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 's')
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_s_reg_inv(self):
        instruction_set = self.circuit.s(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'sdg')
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_sdg(self):
        self.circuit.sdg(self.qr[1])
        self.assertEqual(self.circuit[0].name, 'sdg')
        self.assertEqual(self.circuit[0].param, [])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_sdg_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.sdg, self.cr[0])
        self.assertRaises(QISKitError, qc.sdg, self.cr)
        self.assertRaises(QISKitError, qc.sdg, (self.qr, 3))
        self.assertRaises(QISKitError, qc.sdg, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.sdg, 0)

    def test_sdg_reg(self):
        instruction_set = self.circuit.sdg(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'sdg')
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_sdg_reg_inv(self):
        instruction_set = self.circuit.sdg(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 's')
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_swap(self):
        self.circuit.swap(self.qr[1], self.qr[2])
        self.assertEqual(self.circuit[0].name, 'swap')
        self.assertEqual(self.circuit[0].param, [])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1], self.qr[2]])

    def test_swap_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.swap, self.cr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.swap, self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.swap, 0, self.qr[0])
        self.assertRaises(QISKitError, qc.swap, (self.qr, 3), self.qr[0])
        self.assertRaises(QISKitError, qc.swap, self.cr, self.qr)
        self.assertRaises(QISKitError, qc.swap, 'a', self.qr[1])
        self.assertRaises(QISKitError, qc.swap, self.qr, self.qr2[1])
        self.assertRaises(QISKitError, qc.swap, self.qr[1], self.qr2)

    def test_t(self):
        self.assertRaises(QISKitError, self.circuit.t, self.cr[0])
        self.assertRaises(QISKitError, self.circuit.t, 1)
        self.circuit.t(self.qr[1])
        self.assertEqual(self.circuit[0].name, 't')
        self.assertEqual(self.circuit[0].param, [])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_t_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.t, self.cr[0])
        self.assertRaises(QISKitError, qc.t, self.cr)
        self.assertRaises(QISKitError, qc.t, (self.qr, 3))
        self.assertRaises(QISKitError, qc.t, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.t, 0)

    def test_t_reg(self):
        instruction_set = self.circuit.t(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 't')
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_t_reg_inv(self):
        instruction_set = self.circuit.t(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'tdg')
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_tdg(self):
        self.assertRaises(QISKitError, self.circuit.tdg, self.cr[0])
        self.assertRaises(QISKitError, self.circuit.tdg, 1)
        self.circuit.tdg(self.qr[1])
        self.assertEqual(self.circuit[0].name, 'tdg')
        self.assertEqual(self.circuit[0].param, [])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_tdg_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.tdg, self.cr[0])
        self.assertRaises(QISKitError, qc.tdg, self.cr)
        self.assertRaises(QISKitError, qc.tdg, (self.qr, 3))
        self.assertRaises(QISKitError, qc.tdg, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.tdg, 0)

    def test_tdg_reg(self):
        instruction_set = self.circuit.tdg(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'tdg')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_tdg_reg_inv(self):
        instruction_set = self.circuit.tdg(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 't')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_u1(self):
        self.circuit.u1(1, self.qr[1])
        self.assertEqual(self.circuit[0].name, 'u1')
        self.assertEqual(self.circuit[0].param, [1])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_u1_invalid(self):
        qc = self.circuit
        # CHECKME? self.assertRaises(QISKitError, qc.u1, self.cr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.u1, self.cr[0], self.cr[1])
        self.assertRaises(QISKitError, qc.u1, self.qr[1], 0)
        self.assertRaises(QISKitError, qc.u1, 0, self.cr[0])
        self.assertRaises(QISKitError, qc.u1, 0, 0)
        # TODO self.assertRaises(QISKitError, qc.u1, self.qr[2], self.qr[1])
        self.assertRaises(QISKitError, qc.u1, 0, (self.qr, 3))
        self.assertRaises(QISKitError, qc.u1, 0, self.cr)
        # TODO self.assertRaises(QISKitError, qc.u1, 'a', self.qr[1])
        self.assertRaises(QISKitError, qc.u1, 0, 'a')

    def test_u1_reg(self):
        instruction_set = self.circuit.u1(1, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'u1')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [1])

    def test_u1_reg_inv(self):
        instruction_set = self.circuit.u1(1, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'u1')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [-1])

    def test_u1_pi(self):
        qc = self.circuit
        qc.u1(pi / 2, self.qr[1])
        self.assertEqual(self.circuit[0].name, 'u1')
        self.assertEqual(self.circuit[0].param, [pi / 2])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_u2(self):
        self.circuit.u2(1, 2, self.qr[1])
        self.assertEqual(self.circuit[0].name, 'u2')
        self.assertEqual(self.circuit[0].param, [1, 2])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_u2_invalid(self):
        qc = self.circuit
        # TODO self.assertRaises(QISKitError, qc.u2, 0, self.cr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.u2, 0, self.cr[0], self.cr[1])
        self.assertRaises(QISKitError, qc.u2, 0, self.qr[1], 0)
        self.assertRaises(QISKitError, qc.u2, 0, 0, self.cr[0])
        self.assertRaises(QISKitError, qc.u2, 0, 0, 0)
        # TODO self.assertRaises(QISKitError, qc.u2, 0, self.qr[2], self.qr[1])
        self.assertRaises(QISKitError, qc.u2, 0, 0, (self.qr, 3))
        self.assertRaises(QISKitError, qc.u2, 0, 0, self.cr)
        # TODO self.assertRaises(QISKitError, qc.u2, 0, 'a', self.qr[1])
        self.assertRaises(QISKitError, qc.u2, 0, 0, 'a')

    def test_u2_reg(self):
        instruction_set = self.circuit.u2(1, 2, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'u2')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [1, 2])

    def test_u2_reg_inv(self):
        instruction_set = self.circuit.u2(1, 2, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'u2')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [-pi - 2, -1 + pi])

    def test_u2_pi(self):
        self.circuit.u2(pi / 2, 0.3 * pi, self.qr[1])
        self.assertEqual(self.circuit[0].name, 'u2')
        self.assertEqual(self.circuit[0].param, [pi / 2, 0.3 * pi])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_u3(self):
        self.circuit.u3(1, 2, 3, self.qr[1])
        self.assertEqual(self.circuit[0].name, 'u3')
        self.assertEqual(self.circuit[0].param, [1, 2, 3])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_u3_invalid(self):
        qc = self.circuit
        # CHECKME? self.assertRaises(QISKitError, qc.u3, 0, self.cr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.u3, 0, 0, self.cr[0], self.cr[1])
        self.assertRaises(QISKitError, qc.u3, 0, 0, self.qr[1], 0)
        self.assertRaises(QISKitError, qc.u3, 0, 0, 0, self.cr[0])
        self.assertRaises(QISKitError, qc.u3, 0, 0, 0, 0)
        # TODO self.assertRaises(QISKitError, qc.u3, 0, 0, self.qr[2], self.qr[1])
        self.assertRaises(QISKitError, qc.u3, 0, 0, 0, (self.qr, 3))
        self.assertRaises(QISKitError, qc.u3, 0, 0, 0, self.cr)
        # TODO self.assertRaises(QISKitError, qc.u3, 0, 0, 'a', self.qr[1])
        self.assertRaises(QISKitError, qc.u3, 0, 0, 0, 'a')

    def test_u3_reg(self):
        instruction_set = self.circuit.u3(1, 2, 3, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'u3')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [1, 2, 3])

    def test_u3_reg_inv(self):
        instruction_set = self.circuit.u3(1, 2, 3, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'u3')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [-1, -3, -2])

    def test_u3_pi(self):
        self.circuit.u3(pi, pi / 2, 0.3 * pi, self.qr[1])
        self.assertEqual(self.circuit[0].name, 'u3')
        self.assertEqual(self.circuit[0].param, [pi, pi / 2, 0.3 * pi])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_ubase(self):
        self.circuit.u_base(1, 2, 3, self.qr[1])
        self.assertEqual(self.circuit[0].name, 'U')
        self.assertEqual(self.circuit[0].param, [1, 2, 3])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_ubase_invalid(self):
        qc = self.circuit
        # CHECKME? self.assertRaises(QISKitError, qc.u_base, 0, self.cr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.u_base, 0, 0, self.cr[0], self.cr[1])
        self.assertRaises(QISKitError, qc.u_base, 0, 0, self.qr[1], 0)
        self.assertRaises(QISKitError, qc.u_base, 0, 0, 0, self.cr[0])
        self.assertRaises(QISKitError, qc.u_base, 0, 0, 0, 0)
        # TODO self.assertRaises(QISKitError, qc.u_base, 0, 0, self.qr[2], self.qr[1])
        self.assertRaises(QISKitError, qc.u_base, 0, 0, 0, (self.qr, 3))
        self.assertRaises(QISKitError, qc.u_base, 0, 0, 0, self.cr)
        # TODO self.assertRaises(QISKitError, qc.u_base, 0, 0, 'a', self.qr[1])
        self.assertRaises(QISKitError, qc.u_base, 0, 0, 0, 'a')

    def test_ubase_reg(self):
        instruction_set = self.circuit.u_base(1, 2, 3, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'U')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [1, 2, 3])

    def test_ubase_reg_inv(self):
        instruction_set = self.circuit.u_base(1, 2, 3, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'U')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [-1, -3, -2])

    def test_ubase_pi(self):
        self.circuit.u_base(pi, pi / 2, 0.3 * pi, self.qr[1])
        self.assertEqual(self.circuit[0].name, 'U')
        self.assertEqual(self.circuit[0].param, [pi, pi / 2, 0.3 * pi])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_x(self):
        self.circuit.x(self.qr[1])
        self.assertEqual(self.circuit[0].name, 'x')
        self.assertEqual(self.circuit[0].param, [])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_x_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.x, self.cr[0])
        self.assertRaises(QISKitError, qc.x, self.cr)
        self.assertRaises(QISKitError, qc.x, (self.qr, 3))
        self.assertRaises(QISKitError, qc.x, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.x, 0)

    def test_x_reg(self):
        instruction_set = self.circuit.x(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'x')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_x_reg_inv(self):
        instruction_set = self.circuit.x(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'x')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_y(self):
        self.circuit.y(self.qr[1])
        self.assertEqual(self.circuit[0].name, 'y')
        self.assertEqual(self.circuit[0].param, [])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_y_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.y, self.cr[0])
        self.assertRaises(QISKitError, qc.y, self.cr)
        self.assertRaises(QISKitError, qc.y, (self.qr, 3))
        self.assertRaises(QISKitError, qc.y, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.y, 0)

    def test_y_reg(self):
        instruction_set = self.circuit.y(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'y')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_y_reg_inv(self):
        instruction_set = self.circuit.y(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'y')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_z(self):
        self.circuit.z(self.qr[1])
        self.assertEqual(self.circuit[0].name, 'z')
        self.assertEqual(self.circuit[0].param, [])
        self.assertEqual(self.circuit[0].qargs, [self.qr[1]])

    def test_rzz(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.rzz, 0.1, self.cr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.rzz, 0.1, self.qr[0], self.qr[0])

    def test_z_reg(self):
        instruction_set = self.circuit.z(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'z')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_z_reg_inv(self):
        instruction_set = self.circuit.z(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'z')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])


class TestStandard2Q(QiskitTestCase):
    """Standard Extension Test. Gates with two Qubits"""

    def setUp(self):
        self.qr = QuantumRegister(3, "q")
        self.qr2 = QuantumRegister(3, "r")
        self.cr = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(self.qr, self.qr2, self.cr)

    def test_barrier_reg_bit(self):
        self.circuit.barrier(self.qr, self.qr2[0])
        self.assertEqual(len(self.circuit), 1)
        self.assertEqual(self.circuit[0].name, 'barrier')
        self.assertEqual(self.circuit[0].qargs,
                         [self.qr[0], self.qr[1], self.qr[2], self.qr2[0]])

    def test_ch_reg_reg(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'ch')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_ch_reg_reg_inv(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'ch')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_ch_reg_bit(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'ch')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_ch_reg_bit_inv(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'ch')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_ch_bit_reg(self):
        instruction_set = self.circuit.ch(self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'ch')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_crz_reg_reg(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [1])

    def test_crz_reg_reg_inv(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [-1])

    def test_crz_reg_bit(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [1])

    def test_crz_reg_bit_inv(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [-1])

    def test_crz_bit_reg(self):
        instruction_set = self.circuit.crz(1, self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [1])

    def test_crz_bit_reg_inv(self):
        instruction_set = self.circuit.crz(1, self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [-1])

    def test_cu1_reg_reg(self):
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [1])

    def test_cu1_reg_reg_inv(self):
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [-1])

    def test_cu1_reg_bit(self):
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [1])

    def test_cu1_reg_bit_inv(self):
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [-1])

    def test_cu1_bit_reg(self):
        instruction_set = self.circuit.cu1(1, self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [1])

    def test_cu1_bit_reg_inv(self):
        instruction_set = self.circuit.cu1(1, self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [-1])

    def test_cu3_reg_reg(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [1, 2, 3])

    def test_cu3_reg_reg_inv(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [-1, -3, -2])

    def test_cu3_reg_bit(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [1, 2, 3])

    def test_cu3_reg_bit_inv(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [-1, -3, -2])

    def test_cu3_bit_reg(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [1, 2, 3])

    def test_cu3_bit_reg_inv(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [-1, -3, -2])

    def test_cx_reg_reg(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cx_reg_reg_inv(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cx_reg_bit(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cx_reg_bit_inv(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cx_bit_reg(self):
        instruction_set = self.circuit.cx(self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cx_bit_reg_inv(self):
        instruction_set = self.circuit.cx(self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cxbase_reg_reg(self):
        instruction_set = self.circuit.cx_base(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'CX')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cxbase_reg_reg_inv(self):
        instruction_set = self.circuit.cx_base(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'CX')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cxbase_reg_bit(self):
        instruction_set = self.circuit.cx_base(self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'CX')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cxbase_reg_bit_inv(self):
        instruction_set = self.circuit.cx_base(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'CX')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cxbase_bit_reg(self):
        instruction_set = self.circuit.cx_base(self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'CX')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cxbase_bit_reg_inv(self):
        instruction_set = self.circuit.cx_base(self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'CX')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cy_reg_reg(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cy_reg_reg_inv(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cy_reg_bit(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cy_reg_bit_inv(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cy_bit_reg(self):
        instruction_set = self.circuit.cy(self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cy_bit_reg_inv(self):
        instruction_set = self.circuit.cy(self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cz_reg_reg(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cz_reg_reg_inv(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cz_reg_bit(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cz_reg_bit_inv(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cz_bit_reg(self):
        instruction_set = self.circuit.cz(self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cz_bit_reg_inv(self):
        instruction_set = self.circuit.cz(self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_swap_reg_reg(self):
        instruction_set = self.circuit.swap(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'swap')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_swap_reg_reg_inv(self):
        instruction_set = self.circuit.swap(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'swap')
        self.assertEqual(instruction_set.instructions[1].qargs, [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])


class TestStandard3Q(QiskitTestCase):
    """Standard Extension Test. Gates with three Qubits"""

    def setUp(self):
        self.qr = QuantumRegister(3, "q")
        self.qr2 = QuantumRegister(3, "r")
        self.qr3 = QuantumRegister(3, "s")
        self.cr = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(self.qr, self.qr2, self.qr3, self.cr)

    def test_ccx_reg_reg_reg(self):
        instruction_set = self.circuit.ccx(self.qr, self.qr2, self.qr3)
        self.assertEqual(instruction_set.instructions[0].name, 'ccx')
        self.assertEqual(instruction_set.instructions[1].qargs,
                         [self.qr[1], self.qr2[1], self.qr3[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_ccx_reg_reg_inv(self):
        instruction_set = self.circuit.ccx(self.qr, self.qr2, self.qr3).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'ccx')
        self.assertEqual(instruction_set.instructions[1].qargs,
                         [self.qr[1], self.qr2[1], self.qr3[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cswap_reg_reg_reg(self):
        instruction_set = self.circuit.cswap(self.qr, self.qr2, self.qr3)
        self.assertEqual(instruction_set.instructions[0].name, 'cswap')
        self.assertEqual(instruction_set.instructions[1].qargs,
                         [self.qr[1], self.qr2[1], self.qr3[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])

    def test_cswap_reg_reg_inv(self):
        instruction_set = self.circuit.cswap(self.qr, self.qr2, self.qr3).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cswap')
        self.assertEqual(instruction_set.instructions[1].qargs,
                         [self.qr[1], self.qr2[1], self.qr3[1]])
        self.assertEqual(instruction_set.instructions[2].param, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
