# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=missing-docstring

import unittest

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.qasm import pi
from qiskit.exceptions import QiskitError
from qiskit.test import QiskitTestCase


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
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'barrier')
        self.assertEqual(qargs, [self.qr[1]])

    def test_barrier_wires(self):
        self.circuit.barrier(1)
        self.assertEqual(len(self.circuit), 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'barrier')
        self.assertEqual(qargs, [self.qr[1]])

    def test_barrier_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.barrier, self.cr[0])
        self.assertRaises(QiskitError, qc.barrier, self.cr)
        self.assertRaises(QiskitError, qc.barrier, (self.qr, 'a'))
        self.assertRaises(QiskitError, qc.barrier, .0)

    def test_conditional_barrier_invalid(self):
        qc = self.circuit
        barrier = qc.barrier(self.qr)
        self.assertRaises(QiskitError, barrier.c_if, self.cr, 0)

    def test_barrier_reg(self):
        self.circuit.barrier(self.qr)
        self.assertEqual(len(self.circuit), 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'barrier')
        self.assertEqual(qargs, [self.qr[0], self.qr[1], self.qr[2]])

    def test_barrier_none(self):
        self.circuit.barrier()
        self.assertEqual(len(self.circuit), 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'barrier')
        self.assertEqual(qargs, [self.qr[0], self.qr[1], self.qr[2],
                                 self.qr2[0], self.qr2[1], self.qr2[2]])

    def test_ccx(self):
        self.circuit.ccx(self.qr[0], self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'ccx')
        self.assertEqual(qargs, [self.qr[0], self.qr[1], self.qr[2]])

    def test_ccx_wires(self):
        self.circuit.ccx(0, 1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'ccx')
        self.assertEqual(qargs, [self.qr[0], self.qr[1], self.qr[2]])

    def test_ccx_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.ccx, self.cr[0], self.cr[1], self.cr[2])
        self.assertRaises(QiskitError, qc.ccx, self.qr[0], self.qr[0], self.qr[2])
        self.assertRaises(QiskitError, qc.ccx, 0.0, self.qr[0], self.qr[2])
        self.assertRaises(QiskitError, qc.ccx, self.cr, self.qr, self.qr)
        self.assertRaises(QiskitError, qc.ccx, 'a', self.qr[1], self.qr[2])

    def test_ch(self):
        self.circuit.ch(self.qr[0], self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'ch')
        self.assertEqual(qargs, [self.qr[0], self.qr[1]])

    def test_ch_wires(self):
        self.circuit.ch(0, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'ch')
        self.assertEqual(qargs, [self.qr[0], self.qr[1]])

    def test_ch_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.ch, self.cr[0], self.cr[1])
        self.assertRaises(QiskitError, qc.ch, self.qr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.ch, .0, self.qr[0])
        self.assertRaises(QiskitError, qc.ch, (self.qr, 3), self.qr[0])
        self.assertRaises(QiskitError, qc.ch, self.cr, self.qr)
        self.assertRaises(QiskitError, qc.ch, 'a', self.qr[1])

    def test_crz(self):
        self.circuit.crz(1, self.qr[0], self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'crz')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[0], self.qr[1]])

    def test_crz_wires(self):
        self.circuit.crz(1, 0, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'crz')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[0], self.qr[1]])

    def test_crz_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.crz, 0, self.cr[0], self.cr[1])
        self.assertRaises(QiskitError, qc.crz, 0, self.qr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.crz, 0, .0, self.qr[0])
        self.assertRaises(QiskitError, qc.crz, self.qr[2], self.qr[1], self.qr[0])
        self.assertRaises(QiskitError, qc.crz, 0, self.qr[1], self.cr[2])
        self.assertRaises(QiskitError, qc.crz, 0, (self.qr, 3), self.qr[1])
        self.assertRaises(QiskitError, qc.crz, 0, self.cr, self.qr)
        # TODO self.assertRaises(QiskitError, qc.crz, 'a', self.qr[1], self.qr[2])

    def test_cswap(self):
        self.circuit.cswap(self.qr[0], self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cswap')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[0], self.qr[1], self.qr[2]])

    def test_cswap_wires(self):
        self.circuit.cswap(0, 1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cswap')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[0], self.qr[1], self.qr[2]])

    def test_cswap_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.cswap, self.cr[0], self.cr[1], self.cr[2])
        self.assertRaises(QiskitError, qc.cswap, self.qr[1], self.qr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.cswap, self.qr[1], .0, self.qr[0])
        self.assertRaises(QiskitError, qc.cswap, self.cr[0], self.cr[1], self.qr[0])
        self.assertRaises(QiskitError, qc.cswap, self.qr[0], self.qr[0], self.qr[1])
        self.assertRaises(QiskitError, qc.cswap, .0, self.qr[0], self.qr[1])
        self.assertRaises(QiskitError, qc.cswap, (self.qr, 3), self.qr[0], self.qr[1])
        self.assertRaises(QiskitError, qc.cswap, self.cr, self.qr[0], self.qr[1])
        self.assertRaises(QiskitError, qc.cswap, 'a', self.qr[1], self.qr[2])

    def test_cu1(self):
        self.circuit.cu1(1, self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cu1')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cu1_wires(self):
        self.circuit.cu1(1, 1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cu1')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cu1_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.cu1, self.cr[0], self.cr[1], self.cr[2])
        self.assertRaises(QiskitError, qc.cu1, 1, self.qr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.cu1, self.qr[1], 0, self.qr[0])
        self.assertRaises(QiskitError, qc.cu1, 0, self.cr[0], self.cr[1])
        self.assertRaises(QiskitError, qc.cu1, 0, self.qr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.cu1, 0, .0, self.qr[0])
        self.assertRaises(QiskitError, qc.cu1, self.qr[2], self.qr[1], self.qr[0])
        self.assertRaises(QiskitError, qc.cu1, 0, self.qr[1], self.cr[2])
        self.assertRaises(QiskitError, qc.cu1, 0, (self.qr, 3), self.qr[1])
        self.assertRaises(QiskitError, qc.cu1, 0, self.cr, self.qr)
        # TODO self.assertRaises(QiskitError, qc.cu1, 'a', self.qr[1], self.qr[2])

    def test_cu3(self):
        self.circuit.cu3(1, 2, 3, self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cu3')
        self.assertEqual(op.params, [1, 2, 3])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cu3_wires(self):
        self.circuit.cu3(1, 2, 3, 1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cu3')
        self.assertEqual(op.params, [1, 2, 3])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cu3_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.cu3, 0, 0, self.qr[0], self.qr[1], self.cr[2])
        self.assertRaises(QiskitError, qc.cu3, 0, 0, 0, self.qr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.cu3, 0, 0, self.qr[1], 0, self.qr[0])
        self.assertRaises(QiskitError, qc.cu3, 0, 0, 0, self.qr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.cu3, 0, 0, 0, .0, self.qr[0])
        self.assertRaises(QiskitError, qc.cu3, 0, 0, 0, (self.qr, 3), self.qr[1])
        self.assertRaises(QiskitError, qc.cu3, 0, 0, 0, self.cr, self.qr)
        # TODO self.assertRaises(QiskitError, qc.cu3, 0, 0, 'a', self.qr[1], self.qr[2])

    def test_cx(self):
        self.circuit.cx(self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cx')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cx_wires(self):
        self.circuit.cx(1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cx')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cx_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.cx, self.cr[1], self.cr[2])
        self.assertRaises(QiskitError, qc.cx, self.qr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.cx, .0, self.qr[0])
        self.assertRaises(QiskitError, qc.cx, (self.qr, 3), self.qr[0])
        self.assertRaises(QiskitError, qc.cx, self.cr, self.qr)
        self.assertRaises(QiskitError, qc.cx, 'a', self.qr[1])

    def test_cy(self):
        self.circuit.cy(self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cy')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cy_wires(self):
        self.circuit.cy(1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cy')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cy_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.cy, self.cr[1], self.cr[2])
        self.assertRaises(QiskitError, qc.cy, self.qr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.cy, .0, self.qr[0])
        self.assertRaises(QiskitError, qc.cy, (self.qr, 3), self.qr[0])
        self.assertRaises(QiskitError, qc.cy, self.cr, self.qr)
        self.assertRaises(QiskitError, qc.cy, 'a', self.qr[1])

    def test_cz(self):
        self.circuit.cz(self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cz')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cz_wires(self):
        self.circuit.cz(1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'cz')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_cz_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.cz, self.cr[1], self.cr[2])
        self.assertRaises(QiskitError, qc.cz, self.qr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.cz, .0, self.qr[0])
        self.assertRaises(QiskitError, qc.cz, (self.qr, 3), self.qr[0])
        self.assertRaises(QiskitError, qc.cz, self.cr, self.qr)
        self.assertRaises(QiskitError, qc.cz, 'a', self.qr[1])

    def test_h(self):
        self.circuit.h(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'h')
        self.assertEqual(qargs, [self.qr[1]])

    def test_h_wires(self):
        self.circuit.h(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'h')
        self.assertEqual(qargs, [self.qr[1]])

    def test_h_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.h, self.cr[0])
        self.assertRaises(QiskitError, qc.h, self.cr)
        self.assertRaises(QiskitError, qc.h, (self.qr, 3))
        self.assertRaises(QiskitError, qc.h, (self.qr, 'a'))
        self.assertRaises(QiskitError, qc.h, .0)

    def test_h_reg(self):
        instruction_set = self.circuit.h(self.qr)
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'h')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])

    def test_h_reg_inv(self):
        instruction_set = self.circuit.h(self.qr).inverse()
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'h')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])

    def test_iden(self):
        self.circuit.iden(self.qr[1])
        op, _, _ = self.circuit[0]
        self.assertEqual(op.name, 'id')
        self.assertEqual(op.params, [])

    def test_iden_wires(self):
        self.circuit.iden(1)
        op, _, _ = self.circuit[0]
        self.assertEqual(op.name, 'id')
        self.assertEqual(op.params, [])

    def test_iden_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.iden, self.cr[0])
        self.assertRaises(QiskitError, qc.iden, self.cr)
        self.assertRaises(QiskitError, qc.iden, (self.qr, 3))
        self.assertRaises(QiskitError, qc.iden, (self.qr, 'a'))
        self.assertRaises(QiskitError, qc.iden, .0)

    def test_iden_reg(self):
        instruction_set = self.circuit.iden(self.qr)
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'id')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])

    def test_iden_reg_inv(self):
        instruction_set = self.circuit.iden(self.qr).inverse()
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'id')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])

    def test_rx(self):
        self.circuit.rx(1, self.qr[1])
        op, _, _ = self.circuit[0]
        self.assertEqual(op.name, 'rx')
        self.assertEqual(op.params, [1])

    def test_rx_wires(self):
        self.circuit.rx(1, 1)
        op, _, _ = self.circuit[0]
        self.assertEqual(op.name, 'rx')
        self.assertEqual(op.params, [1])

    def test_rx_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.rx, self.cr[0], self.cr[1])
        self.assertRaises(QiskitError, qc.rx, self.qr[1], 0)
        self.assertRaises(QiskitError, qc.rx, 0, self.cr[0])
        self.assertRaises(QiskitError, qc.rx, 0, .0)
        self.assertRaises(QiskitError, qc.rx, self.qr[2], self.qr[1])
        self.assertRaises(QiskitError, qc.rx, 0, (self.qr, 3))
        self.assertRaises(QiskitError, qc.rx, 0, self.cr)
        # TODO self.assertRaises(QiskitError, qc.rx, 'a', self.qr[1])
        self.assertRaises(QiskitError, qc.rx, 0, 'a')

    def test_rx_reg(self):
        instruction_set = self.circuit.rx(1, self.qr)
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'rx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_rx_reg_inv(self):
        instruction_set = self.circuit.rx(1, self.qr).inverse()
        self.assertEqual(len(instruction_set.instructions), 3)
        self.assertEqual(instruction_set.instructions[0].name, 'rx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_rx_pi(self):
        qc = self.circuit
        qc.rx(pi / 2, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'rx')
        self.assertEqual(op.params, [pi / 2])
        self.assertEqual(qargs, [self.qr[1]])

    def test_ry(self):
        self.circuit.ry(1, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'ry')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1]])

    def test_ry_wires(self):
        self.circuit.ry(1, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'ry')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1]])

    def test_ry_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.ry, self.cr[0], self.cr[1])
        self.assertRaises(QiskitError, qc.ry, self.qr[1], 0)
        self.assertRaises(QiskitError, qc.ry, 0, self.cr[0])
        self.assertRaises(QiskitError, qc.ry, 0, .0)
        self.assertRaises(QiskitError, qc.ry, self.qr[2], self.qr[1])
        self.assertRaises(QiskitError, qc.ry, 0, (self.qr, 3))
        self.assertRaises(QiskitError, qc.ry, 0, self.cr)
        # TODO self.assertRaises(QiskitError, qc.ry, 'a', self.qr[1])
        self.assertRaises(QiskitError, qc.ry, 0, 'a')

    def test_ry_reg(self):
        instruction_set = self.circuit.ry(1, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'ry')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_ry_reg_inv(self):
        instruction_set = self.circuit.ry(1, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'ry')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_ry_pi(self):
        qc = self.circuit
        qc.ry(pi / 2, self.qr[1])
        op, _, _ = self.circuit[0]
        self.assertEqual(op.name, 'ry')
        self.assertEqual(op.params, [pi / 2])

    def test_rz(self):
        self.circuit.rz(1, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'rz')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1]])

    def test_rz_wires(self):
        self.circuit.rz(1, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'rz')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1]])

    def test_rz_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.rz, self.cr[0], self.cr[1])
        self.assertRaises(QiskitError, qc.rz, self.qr[1], 0)
        self.assertRaises(QiskitError, qc.rz, 0, self.cr[0])
        self.assertRaises(QiskitError, qc.rz, 0, .0)
        self.assertRaises(QiskitError, qc.rz, self.qr[2], self.qr[1])
        self.assertRaises(QiskitError, qc.rz, 0, (self.qr, 3))
        self.assertRaises(QiskitError, qc.rz, 0, self.cr)
        # TODO self.assertRaises(QiskitError, qc.rz, 'a', self.qr[1])
        self.assertRaises(QiskitError, qc.rz, 0, 'a')

    def test_rz_reg(self):
        instruction_set = self.circuit.rz(1, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'rz')
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_rz_reg_inv(self):
        instruction_set = self.circuit.rz(1, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'rz')
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_rz_pi(self):
        self.circuit.rz(pi / 2, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'rz')
        self.assertEqual(op.params, [pi / 2])
        self.assertEqual(qargs, [self.qr[1]])

    def test_rzz(self):
        self.circuit.rzz(1, self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'rzz')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_rzz_wires(self):
        self.circuit.rzz(1, 1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'rzz')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_rzz_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.rzz, 1, self.cr[1], self.cr[2])
        self.assertRaises(QiskitError, qc.rzz, 1, self.qr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.rzz, 1, .0, self.qr[0])
        self.assertRaises(QiskitError, qc.rzz, 1, (self.qr, 3), self.qr[0])
        self.assertRaises(QiskitError, qc.rzz, 1, self.cr, self.qr)
        self.assertRaises(QiskitError, qc.rzz, 1, 'a', self.qr[1])
        self.assertRaises(QiskitError, qc.rzz, 0.1, self.cr[1], self.cr[2])
        self.assertRaises(QiskitError, qc.rzz, 0.1, self.qr[0], self.qr[0])

    def test_s(self):
        self.circuit.s(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 's')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_s_wires(self):
        self.circuit.s(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 's')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_s_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.s, self.cr[0])
        self.assertRaises(QiskitError, qc.s, self.cr)
        self.assertRaises(QiskitError, qc.s, (self.qr, 3))
        self.assertRaises(QiskitError, qc.s, (self.qr, 'a'))
        self.assertRaises(QiskitError, qc.s, .0)

    def test_s_reg(self):
        instruction_set = self.circuit.s(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 's')
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_s_reg_inv(self):
        instruction_set = self.circuit.s(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'sdg')
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_sdg(self):
        self.circuit.sdg(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'sdg')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_sdg_wires(self):
        self.circuit.sdg(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'sdg')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_sdg_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.sdg, self.cr[0])
        self.assertRaises(QiskitError, qc.sdg, self.cr)
        self.assertRaises(QiskitError, qc.sdg, (self.qr, 3))
        self.assertRaises(QiskitError, qc.sdg, (self.qr, 'a'))
        self.assertRaises(QiskitError, qc.sdg, .0)

    def test_sdg_reg(self):
        instruction_set = self.circuit.sdg(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'sdg')
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_sdg_reg_inv(self):
        instruction_set = self.circuit.sdg(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 's')
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_swap(self):
        self.circuit.swap(self.qr[1], self.qr[2])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'swap')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_swap_wires(self):
        self.circuit.swap(1, 2)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'swap')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1], self.qr[2]])

    def test_swap_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.swap, self.cr[1], self.cr[2])
        self.assertRaises(QiskitError, qc.swap, self.qr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.swap, .0, self.qr[0])
        self.assertRaises(QiskitError, qc.swap, (self.qr, 3), self.qr[0])
        self.assertRaises(QiskitError, qc.swap, self.cr, self.qr)
        self.assertRaises(QiskitError, qc.swap, 'a', self.qr[1])
        self.assertRaises(QiskitError, qc.swap, self.qr, self.qr2[[1, 2]])
        self.assertRaises(QiskitError, qc.swap, self.qr[:2], self.qr2)

    def test_t(self):
        self.circuit.t(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 't')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_t_wire(self):
        self.circuit.t(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 't')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_t_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.t, self.cr[0])
        self.assertRaises(QiskitError, qc.t, self.cr)
        self.assertRaises(QiskitError, qc.t, (self.qr, 3))
        self.assertRaises(QiskitError, qc.t, (self.qr, 'a'))
        self.assertRaises(QiskitError, qc.t, .0)

    def test_t_reg(self):
        instruction_set = self.circuit.t(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 't')
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_t_reg_inv(self):
        instruction_set = self.circuit.t(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'tdg')
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_tdg(self):
        self.circuit.tdg(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'tdg')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_tdg_wires(self):
        self.circuit.tdg(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'tdg')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_tdg_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.tdg, self.cr[0])
        self.assertRaises(QiskitError, qc.tdg, self.cr)
        self.assertRaises(QiskitError, qc.tdg, (self.qr, 3))
        self.assertRaises(QiskitError, qc.tdg, (self.qr, 'a'))
        self.assertRaises(QiskitError, qc.tdg, .0)

    def test_tdg_reg(self):
        instruction_set = self.circuit.tdg(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'tdg')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_tdg_reg_inv(self):
        instruction_set = self.circuit.tdg(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 't')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_u0(self):
        self.circuit.u0(1, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u0')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u0_wires(self):
        self.circuit.u0(1, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u0')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u0_invalid(self):
        qc = self.circuit
        # CHECKME? self.assertRaises(QiskitError, qc.u0, self.cr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.u0, self.cr[0], self.cr[1])
        self.assertRaises(QiskitError, qc.u0, self.qr[1], 0)
        self.assertRaises(QiskitError, qc.u0, 0, self.cr[0])
        self.assertRaises(QiskitError, qc.u0, 0, .0)
        self.assertRaises(QiskitError, qc.u0, self.qr[2], self.qr[1])
        self.assertRaises(QiskitError, qc.u0, 0, (self.qr, 3))
        self.assertRaises(QiskitError, qc.u0, 0, self.cr)
        # TODO self.assertRaises(QiskitError, qc.u0, 'a', self.qr[1])
        self.assertRaises(QiskitError, qc.u0, 0, 'a')

    def test_u0_reg(self):
        instruction_set = self.circuit.u0(1, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'u0')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_u0_reg_inv(self):
        instruction_set = self.circuit.u0(1, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'u0')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_u0_pi(self):
        qc = self.circuit
        qc.u0(pi / 2, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u0')
        self.assertEqual(op.params, [pi / 2])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u1(self):
        self.circuit.u1(1, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u1')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u1_wires(self):
        self.circuit.u1(1, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u1')
        self.assertEqual(op.params, [1])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u1_invalid(self):
        qc = self.circuit
        # CHECKME? self.assertRaises(QiskitError, qc.u1, self.cr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.u1, self.cr[0], self.cr[1])
        self.assertRaises(QiskitError, qc.u1, self.qr[1], 0)
        self.assertRaises(QiskitError, qc.u1, 0, self.cr[0])
        self.assertRaises(QiskitError, qc.u1, 0, .0)
        self.assertRaises(QiskitError, qc.u1, self.qr[2], self.qr[1])
        self.assertRaises(QiskitError, qc.u1, 0, (self.qr, 3))
        self.assertRaises(QiskitError, qc.u1, 0, self.cr)
        # TODO self.assertRaises(QiskitError, qc.u1, 'a', self.qr[1])
        self.assertRaises(QiskitError, qc.u1, 0, 'a')

    def test_u1_reg(self):
        instruction_set = self.circuit.u1(1, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'u1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_u1_reg_inv(self):
        instruction_set = self.circuit.u1(1, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'u1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_u1_pi(self):
        qc = self.circuit
        qc.u1(pi / 2, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u1')
        self.assertEqual(op.params, [pi / 2])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u2(self):
        self.circuit.u2(1, 2, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u2')
        self.assertEqual(op.params, [1, 2])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u2_wires(self):
        self.circuit.u2(1, 2, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u2')
        self.assertEqual(op.params, [1, 2])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u2_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.u2, 0, self.cr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.u2, 0, self.cr[0], self.cr[1])
        self.assertRaises(QiskitError, qc.u2, 0, self.qr[1], 0)
        self.assertRaises(QiskitError, qc.u2, 0, 0, self.cr[0])
        self.assertRaises(QiskitError, qc.u2, 0, 0, .0)
        self.assertRaises(QiskitError, qc.u2, 0, self.qr[2], self.qr[1])
        self.assertRaises(QiskitError, qc.u2, 0, 0, (self.qr, 3))
        self.assertRaises(QiskitError, qc.u2, 0, 0, self.cr)
        # TODO self.assertRaises(QiskitError, qc.u2, 0, 'a', self.qr[1])
        self.assertRaises(QiskitError, qc.u2, 0, 0, 'a')

    def test_u2_reg(self):
        instruction_set = self.circuit.u2(1, 2, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'u2')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1, 2])

    def test_u2_reg_inv(self):
        instruction_set = self.circuit.u2(1, 2, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'u2')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-pi - 2, -1 + pi])

    def test_u2_pi(self):
        self.circuit.u2(pi / 2, 0.3 * pi, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u2')
        self.assertEqual(op.params, [pi / 2, 0.3 * pi])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u3(self):
        self.circuit.u3(1, 2, 3, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u3')
        self.assertEqual(op.params, [1, 2, 3])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u3_wires(self):
        self.circuit.u3(1, 2, 3, 1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u3')
        self.assertEqual(op.params, [1, 2, 3])
        self.assertEqual(qargs, [self.qr[1]])

    def test_u3_invalid(self):
        qc = self.circuit
        # TODO self.assertRaises(QiskitError, qc.u3, 0, self.cr[0], self.qr[0])
        self.assertRaises(QiskitError, qc.u3, 0, 0, self.cr[0], self.cr[1])
        self.assertRaises(QiskitError, qc.u3, 0, 0, self.qr[1], 0)
        self.assertRaises(QiskitError, qc.u3, 0, 0, 0, self.cr[0])
        self.assertRaises(QiskitError, qc.u3, 0, 0, 0, .0)
        self.assertRaises(QiskitError, qc.u3, 0, 0, self.qr[2], self.qr[1])
        self.assertRaises(QiskitError, qc.u3, 0, 0, 0, (self.qr, 3))
        self.assertRaises(QiskitError, qc.u3, 0, 0, 0, self.cr)
        # TODO self.assertRaises(QiskitError, qc.u3, 0, 0, 'a', self.qr[1])
        self.assertRaises(QiskitError, qc.u3, 0, 0, 0, 'a')

    def test_u3_reg(self):
        instruction_set = self.circuit.u3(1, 2, 3, self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'u3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1, 2, 3])

    def test_u3_reg_inv(self):
        instruction_set = self.circuit.u3(1, 2, 3, self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'u3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1, -3, -2])

    def test_u3_pi(self):
        self.circuit.u3(pi, pi / 2, 0.3 * pi, self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'u3')
        self.assertEqual(op.params, [pi, pi / 2, 0.3 * pi])
        self.assertEqual(qargs, [self.qr[1]])

    def test_x(self):
        self.circuit.x(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'x')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_x_wires(self):
        self.circuit.x(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'x')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_x_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.x, self.cr[0])
        self.assertRaises(QiskitError, qc.x, self.cr)
        self.assertRaises(QiskitError, qc.x, (self.qr, 'a'))
        self.assertRaises(QiskitError, qc.x, 0.0)

    def test_x_reg(self):
        instruction_set = self.circuit.x(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'x')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_x_reg_inv(self):
        instruction_set = self.circuit.x(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'x')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_y(self):
        self.circuit.y(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'y')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_y_wires(self):
        self.circuit.y(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'y')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_y_invalid(self):
        qc = self.circuit
        self.assertRaises(QiskitError, qc.y, self.cr[0])
        self.assertRaises(QiskitError, qc.y, self.cr)
        self.assertRaises(QiskitError, qc.y, (self.qr, 'a'))
        self.assertRaises(QiskitError, qc.y, 0.0)

    def test_y_reg(self):
        instruction_set = self.circuit.y(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'y')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_y_reg_inv(self):
        instruction_set = self.circuit.y(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'y')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_z(self):
        self.circuit.z(self.qr[1])
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'z')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_z_wires(self):
        self.circuit.z(1)
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'z')
        self.assertEqual(op.params, [])
        self.assertEqual(qargs, [self.qr[1]])

    def test_z_reg(self):
        instruction_set = self.circuit.z(self.qr)
        self.assertEqual(instruction_set.instructions[0].name, 'z')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_z_reg_inv(self):
        instruction_set = self.circuit.z(self.qr).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'z')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])


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
        op, qargs, _ = self.circuit[0]
        self.assertEqual(op.name, 'barrier')
        self.assertEqual(qargs, [self.qr[0], self.qr[1], self.qr[2], self.qr2[0]])

    def test_ch_reg_reg(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'ch')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_ch_reg_reg_inv(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'ch')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_ch_reg_bit(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'ch')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_ch_reg_bit_inv(self):
        instruction_set = self.circuit.ch(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'ch')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_ch_bit_reg(self):
        instruction_set = self.circuit.ch(self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'ch')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_crz_reg_reg(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_crz_reg_reg_inv(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_crz_reg_bit(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_crz_reg_bit_inv(self):
        instruction_set = self.circuit.crz(1, self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_crz_bit_reg(self):
        instruction_set = self.circuit.crz(1, self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_crz_bit_reg_inv(self):
        instruction_set = self.circuit.crz(1, self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'crz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_cu1_reg_reg(self):
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_cu1_reg_reg_inv(self):
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_cu1_reg_bit(self):
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_cu1_reg_bit_inv(self):
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_cu1_bit_reg(self):
        instruction_set = self.circuit.cu1(1, self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1])

    def test_cu1_bit_reg_inv(self):
        instruction_set = self.circuit.cu1(1, self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu1')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1])

    def test_cu3_reg_reg(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1, 2, 3])

    def test_cu3_reg_reg_inv(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1, -3, -2])

    def test_cu3_reg_bit(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1, 2, 3])

    def test_cu3_reg_bit_inv(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1, -3, -2])

    def test_cu3_bit_reg(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [1, 2, 3])

    def test_cu3_bit_reg_inv(self):
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cu3')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [-1, -3, -2])

    def test_cx_reg_reg(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cx_reg_reg_inv(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cx_reg_bit(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cx_reg_bit_inv(self):
        instruction_set = self.circuit.cx(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cx_bit_reg(self):
        instruction_set = self.circuit.cx(self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cx_bit_reg_inv(self):
        instruction_set = self.circuit.cx(self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cx')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cy_reg_reg(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cy_reg_reg_inv(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cy_reg_bit(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cy_reg_bit_inv(self):
        instruction_set = self.circuit.cy(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cy_bit_reg(self):
        instruction_set = self.circuit.cy(self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cy_bit_reg_inv(self):
        instruction_set = self.circuit.cy(self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cy')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cz_reg_reg(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cz_reg_reg_inv(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cz_reg_bit(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2[1])
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cz_reg_bit_inv(self):
        instruction_set = self.circuit.cz(self.qr, self.qr2[1]).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cz_bit_reg(self):
        instruction_set = self.circuit.cz(self.qr[1], self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cz_bit_reg_inv(self):
        instruction_set = self.circuit.cz(self.qr[1], self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cz')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_swap_reg_reg(self):
        instruction_set = self.circuit.swap(self.qr, self.qr2)
        self.assertEqual(instruction_set.instructions[0].name, 'swap')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_swap_reg_reg_inv(self):
        instruction_set = self.circuit.swap(self.qr, self.qr2).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'swap')
        self.assertEqual(instruction_set.qargs[1], [self.qr[1], self.qr2[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])


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
        self.assertEqual(instruction_set.qargs[1],
                         [self.qr[1], self.qr2[1], self.qr3[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_ccx_reg_reg_inv(self):
        instruction_set = self.circuit.ccx(self.qr, self.qr2, self.qr3).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'ccx')
        self.assertEqual(instruction_set.qargs[1],
                         [self.qr[1], self.qr2[1], self.qr3[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cswap_reg_reg_reg(self):
        instruction_set = self.circuit.cswap(self.qr, self.qr2, self.qr3)
        self.assertEqual(instruction_set.instructions[0].name, 'cswap')
        self.assertEqual(instruction_set.qargs[1],
                         [self.qr[1], self.qr2[1], self.qr3[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])

    def test_cswap_reg_reg_inv(self):
        instruction_set = self.circuit.cswap(self.qr, self.qr2, self.qr3).inverse()
        self.assertEqual(instruction_set.instructions[0].name, 'cswap')
        self.assertEqual(instruction_set.qargs[1],
                         [self.qr[1], self.qr2[1], self.qr3[1]])
        self.assertEqual(instruction_set.instructions[2].params, [])


if __name__ == '__main__':
    unittest.main(verbosity=2)
