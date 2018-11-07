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

from qiskit.extensions.standard.barrier import Barrier
from qiskit.extensions.standard.ccx import ToffoliGate
from qiskit.extensions.standard.ch import CHGate
from qiskit.extensions.standard.crz import CrzGate
from qiskit.extensions.standard.cswap import FredkinGate
from qiskit.extensions.standard.cu1 import Cu1Gate
from qiskit.extensions.standard.cu3 import Cu3Gate
from qiskit.extensions.standard.cx import CnotGate
from qiskit.extensions.standard.cxbase import CXBase
from qiskit.extensions.standard.cy import CyGate
from qiskit.extensions.standard.cz import CzGate
from qiskit.extensions.standard.h import HGate
from qiskit.extensions.standard.iden import IdGate
from qiskit.extensions.standard.rx import RXGate
from qiskit.extensions.standard.ry import RYGate
from qiskit.extensions.standard.rz import RZGate
from qiskit.extensions.standard.s import SGate, SdgGate
from qiskit.extensions.standard.swap import SwapGate
from qiskit.extensions.standard.t import TGate, TdgGate
from qiskit.extensions.standard.u1 import U1Gate
from qiskit.extensions.standard.u2 import U2Gate
from qiskit.extensions.standard.u3 import U3Gate
from qiskit.extensions.standard.ubase import UBase
from qiskit.extensions.standard.x import XGate
from qiskit.extensions.standard.y import YGate
from qiskit.extensions.standard.z import ZGate
from qiskit.extensions.standard.rzz import RZZGate

from .common import QiskitTestCase


class StandardExtensionTest(QiskitTestCase):
    def assertResult(self, type_, qasm_txt, type_inv=None, qasm_txt_inv=None):
        """
        Assert the single gate in self.circuit is of the type type_, the QASM
        representation matches qasm_txt and the QASM representation of
        inverse matches qasm_txt_inv and type_inv.

        Args:
            type_ (type): a gate type.
            qasm_txt (str): QASM representation of the gate.
            type_inv (type): a inverse gate type. If None, same as type_.
            qasm_txt_inv (str): QASM representation of the inverse gate. If None, same as qasm_txt.
        """
        if not qasm_txt_inv:
            qasm_txt_inv = qasm_txt
        if not type_inv:
            type_inv = type_
        circuit = self.circuit
        self.assertEqual(type(circuit[0]), type_)
        self.assertQasm(qasm_txt)
        circuit[0].reapply(circuit)
        self.assertQasm(qasm_txt + '\n' + qasm_txt)
        self.assertEqual(circuit[0].inverse(), circuit[0])
        self.assertQasm(qasm_txt_inv + '\n' + qasm_txt)
        self.assertEqual(type(circuit[0]), type_inv)

    def assertStmtsType(self, stmts, type_):
        """
        Assert a list of statements stmts is of a type type_.

        Args:
            stmts (list): list of statements.
            type_ (type): a gate type.
        """
        for stmt in stmts:
            self.assertEqual(type(stmt), type_)

    def assertQasm(self, qasm_txt, offset=1):
        """
        Assert the QASM representation of the circuit self.circuit includes
        the text qasm_txt in the right position (which can be adjusted by
        offset)

        Args:
            qasm_txt (str): a string with QASM code
            offset (int): the offset in which qasm_txt should be found.
        """
        circuit = self.circuit
        c_txt = len(qasm_txt)
        self.assertIn('\n' + qasm_txt + '\n', circuit.qasm())
        # pylint: disable=no-member
        self.assertEqual(self.c_header + c_txt + offset, len(circuit.qasm()))


class TestStandard1Q(StandardExtensionTest):
    """Standard Extension Test. Gates with a single Qubit"""

    def setUp(self):
        self.qr = QuantumRegister(3, "q")
        self.qr2 = QuantumRegister(3, "r")
        self.cr = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(self.qr, self.qr2, self.cr)
        self.c_header = 69  # length of the header

    def test_barrier(self):
        self.circuit.barrier(self.qr[1])
        qasm_txt = 'barrier q[1];'
        self.assertResult(Barrier, qasm_txt)

    def test_barrier_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.barrier, self.cr[0])
        self.assertRaises(QISKitError, qc.barrier, self.cr)
        self.assertRaises(QISKitError, qc.barrier, (self.qr, 3))
        self.assertRaises(QISKitError, qc.barrier, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.barrier, 0)

    def test_barrier_reg(self):
        self.circuit.barrier(self.qr)
        qasm_txt = 'barrier q[0],q[1],q[2];'
        self.assertResult(Barrier, qasm_txt)

    def test_barrier_none(self):
        self.circuit.barrier()
        qasm_txt = 'barrier q[0],q[1],q[2],r[0],r[1],r[2];'
        self.assertResult(Barrier, qasm_txt)

    def test_ccx(self):
        self.circuit.ccx(self.qr[0], self.qr[1], self.qr[2])
        qasm_txt = 'ccx q[0],q[1],q[2];'
        self.assertResult(ToffoliGate, qasm_txt)

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
        qasm_txt = 'ch q[0],q[1];'
        self.assertResult(CHGate, qasm_txt)

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
        self.assertResult(CrzGate, 'crz(1) q[0],q[1];', qasm_txt_inv='crz(-1) q[0],q[1];')

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
        qasm_txt = 'cswap q[0],q[1],q[2];'
        self.assertResult(FredkinGate, qasm_txt)

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
        self.assertResult(Cu1Gate, 'cu1(1) q[1],q[2];', qasm_txt_inv='cu1(-1) q[1],q[2];')

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
        self.assertResult(Cu3Gate, 'cu3(1,2,3) q[1],q[2];', qasm_txt_inv='cu3(-1,-3,-2) q[1],q[2];')

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
        qasm_txt = 'cx q[1],q[2];'
        self.assertResult(CnotGate, qasm_txt)

    def test_cx_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.cx, self.cr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.cx, self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.cx, 0, self.qr[0])
        self.assertRaises(QISKitError, qc.cx, (self.qr, 3), self.qr[0])
        self.assertRaises(QISKitError, qc.cx, self.cr, self.qr)
        self.assertRaises(QISKitError, qc.cx, 'a', self.qr[1])

    def test_cxbase(self):
        qasm_txt = 'CX q[1],q[2];'
        self.circuit.cx_base(self.qr[1], self.qr[2])
        self.assertResult(CXBase, qasm_txt)

    def test_cxbase_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.cx_base, self.cr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.cx_base, self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.cx_base, 0, self.qr[0])
        self.assertRaises(QISKitError, qc.cx_base, (self.qr, 3), self.qr[0])
        self.assertRaises(QISKitError, qc.cx_base, self.cr, self.qr)
        self.assertRaises(QISKitError, qc.cx_base, 'a', self.qr[1])

    def test_cy(self):
        qasm_txt = 'cy q[1],q[2];'
        self.circuit.cy(self.qr[1], self.qr[2])
        self.assertResult(CyGate, qasm_txt)

    def test_cy_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.cy, self.cr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.cy, self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.cy, 0, self.qr[0])
        self.assertRaises(QISKitError, qc.cy, (self.qr, 3), self.qr[0])
        self.assertRaises(QISKitError, qc.cy, self.cr, self.qr)
        self.assertRaises(QISKitError, qc.cy, 'a', self.qr[1])

    def test_cz(self):
        qasm_txt = 'cz q[1],q[2];'
        self.circuit.cz(self.qr[1], self.qr[2])
        self.assertResult(CzGate, qasm_txt)

    def test_cz_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.cz, self.cr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.cz, self.qr[0], self.qr[0])
        self.assertRaises(QISKitError, qc.cz, 0, self.qr[0])
        self.assertRaises(QISKitError, qc.cz, (self.qr, 3), self.qr[0])
        self.assertRaises(QISKitError, qc.cz, self.cr, self.qr)
        self.assertRaises(QISKitError, qc.cz, 'a', self.qr[1])

    def test_h(self):
        qasm_txt = 'h q[1];'
        self.circuit.h(self.qr[1])
        self.assertResult(HGate, qasm_txt)

    def test_h_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.h, self.cr[0])
        self.assertRaises(QISKitError, qc.h, self.cr)
        self.assertRaises(QISKitError, qc.h, (self.qr, 3))
        self.assertRaises(QISKitError, qc.h, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.h, 0)

    def test_h_reg(self):
        qasm_txt = 'h q[0];\nh q[1];\nh q[2];'
        instruction_set = self.circuit.h(self.qr)
        self.assertStmtsType(instruction_set.instructions, HGate)
        self.assertQasm(qasm_txt)

    def test_h_reg_inv(self):
        qasm_txt = 'h q[0];\nh q[1];\nh q[2];'
        instruction_set = self.circuit.h(self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, HGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 22)

    def test_iden(self):
        self.circuit.iden(self.qr[1])
        self.assertResult(IdGate, 'id q[1];')

    def test_iden_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.iden, self.cr[0])
        self.assertRaises(QISKitError, qc.iden, self.cr)
        self.assertRaises(QISKitError, qc.iden, (self.qr, 3))
        self.assertRaises(QISKitError, qc.iden, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.iden, 0)

    def test_iden_reg(self):
        qasm_txt = 'id q[0];\nid q[1];\nid q[2];'
        instruction_set = self.circuit.iden(self.qr)
        self.assertStmtsType(instruction_set.instructions, IdGate)
        self.assertQasm(qasm_txt)

    def test_iden_reg_inv(self):
        qasm_txt = 'id q[0];\nid q[1];\nid q[2];'
        instruction_set = self.circuit.iden(self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, IdGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 25)

    def test_rx(self):
        self.circuit.rx(1, self.qr[1])
        self.assertResult(RXGate, 'rx(1) q[1];', qasm_txt_inv='rx(-1) q[1];')

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
        qasm_txt = 'rx(1) q[0];\nrx(1) q[1];\nrx(1) q[2];'
        instruction_set = self.circuit.rx(1, self.qr)
        self.assertStmtsType(instruction_set.instructions, RXGate)
        self.assertQasm(qasm_txt)

    def test_rx_reg_inv(self):
        qasm_txt = 'rx(-1) q[0];\nrx(-1) q[1];\nrx(-1) q[2];'
        instruction_set = self.circuit.rx(1, self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, RXGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 37)

    def test_rx_pi(self):
        qc = self.circuit
        qc.rx(pi / 2, self.qr[1])
        self.assertResult(RXGate, 'rx(pi/2) q[1];', qasm_txt_inv='rx(-pi/2) q[1];')

    def test_ry(self):
        self.circuit.ry(1, self.qr[1])
        self.assertResult(RYGate, 'ry(1) q[1];', qasm_txt_inv='ry(-1) q[1];')

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
        qasm_txt = 'ry(1) q[0];\nry(1) q[1];\nry(1) q[2];'
        instruction_set = self.circuit.ry(1, self.qr)
        self.assertStmtsType(instruction_set.instructions, RYGate)
        self.assertQasm(qasm_txt)

    def test_ry_reg_inv(self):
        qasm_txt = 'ry(-1) q[0];\nry(-1) q[1];\nry(-1) q[2];'
        instruction_set = self.circuit.ry(1, self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, RYGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 37)

    def test_ry_pi(self):
        qc = self.circuit
        qc.ry(pi / 2, self.qr[1])
        self.assertResult(RYGate, 'ry(pi/2) q[1];', qasm_txt_inv='ry(-pi/2) q[1];')

    def test_rz(self):
        self.circuit.rz(1, self.qr[1])
        self.assertResult(RZGate, 'rz(1) q[1];', qasm_txt_inv='rz(-1) q[1];')

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
        qasm_txt = 'rz(1) q[0];\nrz(1) q[1];\nrz(1) q[2];'
        instruction_set = self.circuit.rz(1, self.qr)
        self.assertStmtsType(instruction_set.instructions, RZGate)
        self.assertQasm(qasm_txt)

    def test_rz_reg_inv(self):
        qasm_txt = 'rz(-1) q[0];\nrz(-1) q[1];\nrz(-1) q[2];'
        instruction_set = self.circuit.rz(1, self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, RZGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 37)

    def test_rz_pi(self):
        qc = self.circuit
        qc.rz(pi / 2, self.qr[1])
        self.assertResult(RZGate, 'rz(pi/2) q[1];', qasm_txt_inv='rz(-pi/2) q[1];')

    def test_s(self):
        self.circuit.s(self.qr[1])
        self.assertResult(SGate, 's q[1];', type_inv=SdgGate, qasm_txt_inv='sdg q[1];')

    def test_s_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.s, self.cr[0])
        self.assertRaises(QISKitError, qc.s, self.cr)
        self.assertRaises(QISKitError, qc.s, (self.qr, 3))
        self.assertRaises(QISKitError, qc.s, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.s, 0)

    def test_s_reg(self):
        qasm_txt = 's q[0];\ns q[1];\ns q[2];'
        instruction_set = self.circuit.s(self.qr)
        self.assertStmtsType(instruction_set.instructions, SGate)
        self.assertQasm(qasm_txt)

    def test_s_reg_inv(self):
        qasm_txt = 'sdg q[0];\nsdg q[1];\nsdg q[2];'
        instruction_set = self.circuit.s(self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, SdgGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 28)

    def test_sdg(self):
        self.circuit.sdg(self.qr[1])
        self.assertResult(SdgGate, 'sdg q[1];', type_inv=SGate, qasm_txt_inv='s q[1];')

    def test_sdg_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.sdg, self.cr[0])
        self.assertRaises(QISKitError, qc.sdg, self.cr)
        self.assertRaises(QISKitError, qc.sdg, (self.qr, 3))
        self.assertRaises(QISKitError, qc.sdg, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.sdg, 0)

    def test_sdg_reg(self):
        qasm_txt = 'sdg q[0];\nsdg q[1];\nsdg q[2];'
        instruction_set = self.circuit.sdg(self.qr)
        self.assertStmtsType(instruction_set.instructions, SdgGate)
        self.assertQasm(qasm_txt)

    def test_sdg_reg_inv(self):
        qasm_txt = 's q[0];\ns q[1];\ns q[2];'
        instruction_set = self.circuit.sdg(self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, SGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 22)

    def test_swap(self):
        self.circuit.swap(self.qr[1], self.qr[2])
        self.assertResult(SwapGate, 'swap q[1],q[2];', qasm_txt_inv='swap q[1],q[2];')

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
        qc = self.circuit
        self.assertRaises(QISKitError, qc.t, self.cr[0])
        # TODO self.assertRaises(QISKitError, qc.t, 1)
        qc.t(self.qr[1])
        self.assertResult(TGate, 't q[1];', type_inv=TdgGate, qasm_txt_inv='tdg q[1];')

    def test_t_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.t, self.cr[0])
        self.assertRaises(QISKitError, qc.t, self.cr)
        self.assertRaises(QISKitError, qc.t, (self.qr, 3))
        self.assertRaises(QISKitError, qc.t, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.t, 0)

    def test_t_reg(self):
        qasm_txt = 't q[0];\nt q[1];\nt q[2];'
        instruction_set = self.circuit.t(self.qr)
        self.assertStmtsType(instruction_set.instructions, TGate)
        self.assertQasm(qasm_txt)

    def test_t_reg_inv(self):
        qasm_txt = 'tdg q[0];\ntdg q[1];\ntdg q[2];'
        instruction_set = self.circuit.t(self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, TdgGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 28)

    def test_tdg(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.tdg, self.cr[0])
        # TODO self.assertRaises(QISKitError, qc.tdg, 1)
        qc.tdg(self.qr[1])
        self.assertResult(TdgGate, 'tdg q[1];', type_inv=TGate, qasm_txt_inv='t q[1];')

    def test_tdg_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.tdg, self.cr[0])
        self.assertRaises(QISKitError, qc.tdg, self.cr)
        self.assertRaises(QISKitError, qc.tdg, (self.qr, 3))
        self.assertRaises(QISKitError, qc.tdg, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.tdg, 0)

    def test_tdg_reg(self):
        qasm_txt = 'tdg q[0];\ntdg q[1];\ntdg q[2];'
        instruction_set = self.circuit.tdg(self.qr)
        self.assertStmtsType(instruction_set.instructions, TdgGate)
        self.assertQasm(qasm_txt)

    def test_tdg_reg_inv(self):
        qasm_txt = 't q[0];\nt q[1];\nt q[2];'
        instruction_set = self.circuit.tdg(self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, TGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 22)

    def test_u1(self):
        self.circuit.u1(1, self.qr[1])
        self.assertResult(U1Gate, 'u1(1) q[1];', qasm_txt_inv='u1(-1) q[1];')

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
        qasm_txt = 'u1(1) q[0];\nu1(1) q[1];\nu1(1) q[2];'
        instruction_set = self.circuit.u1(1, self.qr)
        self.assertStmtsType(instruction_set.instructions, U1Gate)
        self.assertQasm(qasm_txt)

    def test_u1_reg_inv(self):
        qasm_txt = 'u1(-1) q[0];\nu1(-1) q[1];\nu1(-1) q[2];'
        instruction_set = self.circuit.u1(1, self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, U1Gate)
        self.assertQasm(qasm_txt)

    def test_u1_pi(self):
        qc = self.circuit
        qc.u1(pi / 2, self.qr[1])
        self.assertResult(U1Gate, 'u1(pi/2) q[1];', qasm_txt_inv='u1(-pi/2) q[1];')

    def test_u2(self):
        self.circuit.u2(1, 2, self.qr[1])
        self.assertResult(U2Gate, 'u2(1,2) q[1];', qasm_txt_inv='u2(-pi - 2,-1 + pi) q[1];')

    def test_u2_invalid(self):
        qc = self.circuit
        # CHECKME? self.assertRaises(QISKitError, qc.u2, 0, self.cr[0], self.qr[0])
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
        qasm_txt = 'u2(1,2) q[0];\nu2(1,2) q[1];\nu2(1,2) q[2];'
        instruction_set = self.circuit.u2(1, 2, self.qr)
        self.assertStmtsType(instruction_set.instructions, U2Gate)
        self.assertQasm(qasm_txt)

    def test_u2_reg_inv(self):
        qasm_txt = 'u2(-pi - 2,-1 + pi) q[0];\nu2(-pi - 2,-1 + pi) q[1];\nu2(-pi - 2,-1 + pi) q[2];'
        instruction_set = self.circuit.u2(1, 2, self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, U2Gate)
        self.assertQasm(qasm_txt)

    def test_u2_pi(self):
        qc = self.circuit
        qc.u2(pi / 2, 0.3 * pi, self.qr[1])
        self.assertResult(U2Gate, 'u2(pi/2,0.3*pi) q[1];', qasm_txt_inv='u2(-1.3*pi,pi/2) q[1];')

    def test_u3(self):
        self.circuit.u3(1, 2, 3, self.qr[1])
        self.assertResult(U3Gate, 'u3(1,2,3) q[1];', qasm_txt_inv='u3(-1,-3,-2) q[1];')

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
        qasm_txt = 'u3(1,2,3) q[0];\nu3(1,2,3) q[1];\nu3(1,2,3) q[2];'
        instruction_set = self.circuit.u3(1, 2, 3, self.qr)
        self.assertStmtsType(instruction_set.instructions, U3Gate)
        self.assertQasm(qasm_txt)

    def test_u3_reg_inv(self):
        qasm_txt = 'u3(-1,-3,-2) q[0];\nu3(-1,-3,-2) q[1];\nu3(-1,-3,-2) q[2];'
        instruction_set = self.circuit.u3(1, 2, 3, self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, U3Gate)
        self.assertQasm(qasm_txt)

    def test_u3_pi(self):
        qc = self.circuit
        qc.u3(pi, pi / 2, 0.3 * pi, self.qr[1])
        self.assertResult(U3Gate, 'u3(pi,pi/2,0.3*pi) q[1];',
                          qasm_txt_inv='u3(-pi,-0.3*pi,-pi/2) q[1];')

    def test_ubase(self):
        self.circuit.u_base(1, 2, 3, self.qr[1])
        self.assertResult(UBase, 'U(1,2,3) q[1];', qasm_txt_inv='U(-1,-3,-2) q[1];')

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
        qasm_txt = 'U(1,2,3) q[0];\nU(1,2,3) q[1];\nU(1,2,3) q[2];'
        instruction_set = self.circuit.u_base(1, 2, 3, self.qr)
        self.assertStmtsType(instruction_set.instructions, UBase)
        self.assertQasm(qasm_txt)

    def test_ubase_reg_inv(self):
        qasm_txt = 'U(-1,-3,-2) q[0];\nU(-1,-3,-2) q[1];\nU(-1,-3,-2) q[2];'
        instruction_set = self.circuit.u_base(1, 2, 3, self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, UBase)
        self.assertQasm(qasm_txt)

    def test_ubase_pi(self):
        qc = self.circuit
        qc.u_base(pi, pi / 2, 0.3 * pi, self.qr[1])
        self.assertResult(UBase, 'U(pi,pi/2,0.3*pi) q[1];',
                          qasm_txt_inv='U(-pi,-0.3*pi,-pi/2) q[1];')

    def test_x(self):
        self.circuit.x(self.qr[1])
        self.assertResult(XGate, 'x q[1];', qasm_txt_inv='x q[1];')

    def test_x_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.x, self.cr[0])
        self.assertRaises(QISKitError, qc.x, self.cr)
        self.assertRaises(QISKitError, qc.x, (self.qr, 3))
        self.assertRaises(QISKitError, qc.x, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.x, 0)

    def test_x_reg(self):
        qasm_txt = 'x q[0];\nx q[1];\nx q[2];'
        instruction_set = self.circuit.x(self.qr)
        self.assertStmtsType(instruction_set.instructions, XGate)
        self.assertQasm(qasm_txt)

    def test_x_reg_inv(self):
        qasm_txt = 'x q[0];\nx q[1];\nx q[2];'
        instruction_set = self.circuit.x(self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, XGate)
        self.assertQasm(qasm_txt)

    def test_y(self):
        self.circuit.y(self.qr[1])
        self.assertResult(YGate, 'y q[1];')

    def test_y_invalid(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.y, self.cr[0])
        self.assertRaises(QISKitError, qc.y, self.cr)
        self.assertRaises(QISKitError, qc.y, (self.qr, 3))
        self.assertRaises(QISKitError, qc.y, (self.qr, 'a'))
        self.assertRaises(QISKitError, qc.y, 0)

    def test_y_reg(self):
        qasm_txt = 'y q[0];\ny q[1];\ny q[2];'
        instruction_set = self.circuit.y(self.qr)
        self.assertStmtsType(instruction_set.instructions, YGate)
        self.assertQasm(qasm_txt)

    def test_y_reg_inv(self):
        qasm_txt = 'y q[0];\ny q[1];\ny q[2];'
        instruction_set = self.circuit.y(self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, YGate)
        self.assertQasm(qasm_txt)

    def test_z(self):
        self.circuit.z(self.qr[1])
        self.assertResult(ZGate, 'z q[1];')

    def test_rzz(self):
        qc = self.circuit
        self.assertRaises(QISKitError, qc.rzz, 0.1, self.cr[1], self.cr[2])
        self.assertRaises(QISKitError, qc.rzz, 0.1, self.qr[0], self.qr[0])
        qc.rzz(pi / 2, self.qr[1], self.qr[2])
        self.assertResult(RZZGate, 'rzz(pi/2) q[1],q[2];', qasm_txt_inv='rzz(-pi/2) q[1],q[2];')

    def test_z_reg(self):
        qasm_txt = 'z q[0];\nz q[1];\nz q[2];'
        instruction_set = self.circuit.z(self.qr)
        self.assertStmtsType(instruction_set.instructions, ZGate)
        self.assertQasm(qasm_txt)

    def test_z_reg_inv(self):
        qasm_txt = 'z q[0];\nz q[1];\nz q[2];'
        instruction_set = self.circuit.z(self.qr).inverse()
        self.assertStmtsType(instruction_set.instructions, ZGate)
        self.assertQasm(qasm_txt)


class TestStandard2Q(StandardExtensionTest):
    """Standard Extension Test. Gates with two Qubits"""

    def setUp(self):
        self.qr = QuantumRegister(3, "q")
        self.qr2 = QuantumRegister(3, "r")
        self.cr = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(self.qr, self.qr2, self.cr)
        self.c_header = 69  # length of the header

    def test_barrier_none(self):
        self.circuit.barrier()
        qasm_txt = 'barrier q[0],q[1],q[2],r[0],r[1],r[2];'
        self.assertResult(Barrier, qasm_txt)

    def test_barrier_reg_bit(self):
        self.circuit.barrier(self.qr, self.qr2[0])
        qasm_txt = 'barrier q[0],q[1],q[2],r[0];'
        self.assertResult(Barrier, qasm_txt)

    def test_ch_reg_reg(self):
        qasm_txt = 'ch q[0],r[0];\nch q[1],r[1];\nch q[2],r[2];'
        instruction_set = self.circuit.ch(self.qr, self.qr2)
        self.assertStmtsType(instruction_set.instructions, CHGate)
        self.assertQasm(qasm_txt)

    def test_ch_reg_reg_inv(self):
        qasm_txt = 'ch q[0],r[0];\nch q[1],r[1];\nch q[2],r[2];'
        instruction_set = self.circuit.ch(self.qr, self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, CHGate)
        self.assertQasm(qasm_txt)

    def test_ch_reg_bit(self):
        qasm_txt = 'ch q[0],r[1];\nch q[1],r[1];\nch q[2],r[1];'
        instruction_set = self.circuit.ch(self.qr, self.qr2[1])
        self.assertStmtsType(instruction_set.instructions, CHGate)
        self.assertQasm(qasm_txt)

    def test_ch_reg_bit_inv(self):
        qasm_txt = 'ch q[0],r[1];\nch q[1],r[1];\nch q[2],r[1];'
        instruction_set = self.circuit.ch(self.qr, self.qr2[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, CHGate)
        self.assertQasm(qasm_txt)

    def test_ch_bit_reg(self):
        qasm_txt = 'ch q[1],r[0];\nch q[1],r[1];\nch q[1],r[2];'
        instruction_set = self.circuit.ch(self.qr[1], self.qr2)
        self.assertStmtsType(instruction_set.instructions, CHGate)
        self.assertQasm(qasm_txt)

    def test_ch_bit_reg_inv(self):
        qasm_txt = 'ch q[1],r[0];\nch q[1],r[1];\nch q[1],r[2];'
        instruction_set = self.circuit.ch(self.qr[1], self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, CHGate)
        self.assertQasm(qasm_txt)

    def test_crz_reg_reg(self):
        qasm_txt = 'crz(1) q[0],r[0];\ncrz(1) q[1],r[1];\ncrz(1) q[2],r[2];'
        instruction_set = self.circuit.crz(1, self.qr, self.qr2)
        self.assertStmtsType(instruction_set.instructions, CrzGate)
        self.assertQasm(qasm_txt)

    def test_crz_reg_reg_inv(self):
        qasm_txt = 'crz(-1) q[0],r[0];\ncrz(-1) q[1],r[1];\ncrz(-1) q[2],r[2];'
        instruction_set = self.circuit.crz(1, self.qr, self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, CrzGate)
        self.assertQasm(qasm_txt)

    def test_crz_reg_bit(self):
        qasm_txt = 'crz(1) q[0],r[1];\ncrz(1) q[1],r[1];\ncrz(1) q[2],r[1];'
        instruction_set = self.circuit.crz(1, self.qr, self.qr2[1])
        self.assertStmtsType(instruction_set.instructions, CrzGate)
        self.assertQasm(qasm_txt)

    def test_crz_reg_bit_inv(self):
        qasm_txt = 'crz(-1) q[0],r[1];\ncrz(-1) q[1],r[1];\ncrz(-1) q[2],r[1];'
        instruction_set = self.circuit.crz(1, self.qr, self.qr2[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, CrzGate)
        self.assertQasm(qasm_txt)

    def test_crz_bit_reg(self):
        qasm_txt = 'crz(1) q[1],r[0];\ncrz(1) q[1],r[1];\ncrz(1) q[1],r[2];'
        instruction_set = self.circuit.crz(1, self.qr[1], self.qr2)
        self.assertStmtsType(instruction_set.instructions, CrzGate)
        self.assertQasm(qasm_txt)

    def test_crz_bit_reg_inv(self):
        qasm_txt = 'crz(-1) q[1],r[0];\ncrz(-1) q[1],r[1];\ncrz(-1) q[1],r[2];'
        instruction_set = self.circuit.crz(1, self.qr[1], self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, CrzGate)
        self.assertQasm(qasm_txt)

    def test_cu1_reg_reg(self):
        qasm_txt = 'cu1(1) q[0],r[0];\ncu1(1) q[1],r[1];\ncu1(1) q[2],r[2];'
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2)
        self.assertStmtsType(instruction_set.instructions, Cu1Gate)
        self.assertQasm(qasm_txt)

    def test_cu1_reg_reg_inv(self):
        qasm_txt = 'cu1(-1) q[0],r[0];\ncu1(-1) q[1],r[1];\ncu1(-1) q[2],r[2];'
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, Cu1Gate)
        self.assertQasm(qasm_txt)

    def test_cu1_reg_bit(self):
        qasm_txt = 'cu1(1) q[0],r[1];\ncu1(1) q[1],r[1];\ncu1(1) q[2],r[1];'
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2[1])
        self.assertStmtsType(instruction_set.instructions, Cu1Gate)
        self.assertQasm(qasm_txt)

    def test_cu1_reg_bit_inv(self):
        qasm_txt = 'cu1(-1) q[0],r[1];\ncu1(-1) q[1],r[1];\ncu1(-1) q[2],r[1];'
        instruction_set = self.circuit.cu1(1, self.qr, self.qr2[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, Cu1Gate)
        self.assertQasm(qasm_txt)

    def test_cu1_bit_reg(self):
        qasm_txt = 'cu1(1) q[1],r[0];\ncu1(1) q[1],r[1];\ncu1(1) q[1],r[2];'
        instruction_set = self.circuit.cu1(1, self.qr[1], self.qr2)
        self.assertStmtsType(instruction_set.instructions, Cu1Gate)
        self.assertQasm(qasm_txt)

    def test_cu1_bit_reg_inv(self):
        qasm_txt = 'cu1(-1) q[1],r[0];\ncu1(-1) q[1],r[1];\ncu1(-1) q[1],r[2];'
        instruction_set = self.circuit.cu1(1, self.qr[1], self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, Cu1Gate)
        self.assertQasm(qasm_txt)

    def test_cu3_reg_reg(self):
        qasm_txt = 'cu3(1,2,3) q[0],r[0];\ncu3(1,2,3) q[1],r[1];\ncu3(1,2,3) q[2],r[2];'
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2)
        self.assertStmtsType(instruction_set.instructions, Cu3Gate)
        self.assertQasm(qasm_txt)

    def test_cu3_reg_reg_inv(self):
        qasm_txt = 'cu3(-1,-3,-2) q[0],r[0];\ncu3(-1,-3,-2) q[1],r[1];\ncu3(-1,-3,-2) q[2],r[2];'
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, Cu3Gate)
        self.assertQasm(qasm_txt)

    def test_cu3_reg_bit(self):
        qasm_txt = 'cu3(1,2,3) q[0],r[1];\ncu3(1,2,3) q[1],r[1];\ncu3(1,2,3) q[2],r[1];'
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2[1])
        self.assertStmtsType(instruction_set.instructions, Cu3Gate)
        self.assertQasm(qasm_txt)

    def test_cu3_reg_bit_inv(self):
        qasm_txt = 'cu3(-1,-3,-2) q[0],r[1];\ncu3(-1,-3,-2) q[1],r[1];\ncu3(-1,-3,-2) q[2],r[1];'
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr, self.qr2[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, Cu3Gate)
        self.assertQasm(qasm_txt)

    def test_cu3_bit_reg(self):
        qasm_txt = 'cu3(1,2,3) q[1],r[0];\ncu3(1,2,3) q[1],r[1];\ncu3(1,2,3) q[1],r[2];'
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr[1], self.qr2)
        self.assertStmtsType(instruction_set.instructions, Cu3Gate)
        self.assertQasm(qasm_txt)

    def test_cu3_bit_reg_inv(self):
        qasm_txt = 'cu3(-1,-3,-2) q[1],r[0];\ncu3(-1,-3,-2) q[1],r[1];\ncu3(-1,-3,-2) q[1],r[2];'
        instruction_set = self.circuit.cu3(1, 2, 3, self.qr[1], self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, Cu3Gate)
        self.assertQasm(qasm_txt)

    def test_cx_reg_reg(self):
        qasm_txt = 'cx q[0],r[0];\ncx q[1],r[1];\ncx q[2],r[2];'
        instruction_set = self.circuit.cx(self.qr, self.qr2)
        self.assertStmtsType(instruction_set.instructions, CnotGate)
        self.assertQasm(qasm_txt)

    def test_cx_reg_reg_inv(self):
        qasm_txt = 'cx q[0],r[0];\ncx q[1],r[1];\ncx q[2],r[2];'
        instruction_set = self.circuit.cx(self.qr, self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, CnotGate)
        self.assertQasm(qasm_txt)

    def test_cx_reg_bit(self):
        qasm_txt = 'cx q[0],r[1];\ncx q[1],r[1];\ncx q[2],r[1];'
        instruction_set = self.circuit.cx(self.qr, self.qr2[1])
        self.assertStmtsType(instruction_set.instructions, CnotGate)
        self.assertQasm(qasm_txt)

    def test_cx_reg_bit_inv(self):
        qasm_txt = 'cx q[0],r[1];\ncx q[1],r[1];\ncx q[2],r[1];'
        instruction_set = self.circuit.cx(self.qr, self.qr2[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, CnotGate)
        self.assertQasm(qasm_txt)

    def test_cx_bit_reg(self):
        qasm_txt = 'cx q[1],r[0];\ncx q[1],r[1];\ncx q[1],r[2];'
        instruction_set = self.circuit.cx(self.qr[1], self.qr2)
        self.assertStmtsType(instruction_set.instructions, CnotGate)
        self.assertQasm(qasm_txt)

    def test_cx_bit_reg_inv(self):
        qasm_txt = 'cx q[1],r[0];\ncx q[1],r[1];\ncx q[1],r[2];'
        instruction_set = self.circuit.cx(self.qr[1], self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, CnotGate)
        self.assertQasm(qasm_txt)

    def test_cxbase_reg_reg(self):
        qasm_txt = 'CX q[0],r[0];\nCX q[1],r[1];\nCX q[2],r[2];'
        instruction_set = self.circuit.cx_base(self.qr, self.qr2)
        self.assertStmtsType(instruction_set.instructions, CXBase)
        self.assertQasm(qasm_txt)

    def test_cxbase_reg_reg_inv(self):
        qasm_txt = 'CX q[0],r[0];\nCX q[1],r[1];\nCX q[2],r[2];'
        instruction_set = self.circuit.cx_base(self.qr, self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, CXBase)
        self.assertQasm(qasm_txt)

    def test_cxbase_reg_bit(self):
        qasm_txt = 'CX q[0],r[1];\nCX q[1],r[1];\nCX q[2],r[1];'
        instruction_set = self.circuit.cx_base(self.qr, self.qr2[1])
        self.assertStmtsType(instruction_set.instructions, CXBase)
        self.assertQasm(qasm_txt)

    def test_cxbase_reg_bit_inv(self):
        qasm_txt = 'CX q[0],r[1];\nCX q[1],r[1];\nCX q[2],r[1];'
        instruction_set = self.circuit.cx_base(self.qr, self.qr2[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, CXBase)
        self.assertQasm(qasm_txt)

    def test_cxbase_bit_reg(self):
        qasm_txt = 'CX q[1],r[0];\nCX q[1],r[1];\nCX q[1],r[2];'
        instruction_set = self.circuit.cx_base(self.qr[1], self.qr2)
        self.assertStmtsType(instruction_set.instructions, CXBase)
        self.assertQasm(qasm_txt)

    def test_cxbase_bit_reg_inv(self):
        qasm_txt = 'CX q[1],r[0];\nCX q[1],r[1];\nCX q[1],r[2];'
        instruction_set = self.circuit.cx_base(self.qr[1], self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, CXBase)
        self.assertQasm(qasm_txt)

    def test_cy_reg_reg(self):
        qasm_txt = 'cy q[0],r[0];\ncy q[1],r[1];\ncy q[2],r[2];'
        instruction_set = self.circuit.cy(self.qr, self.qr2)
        self.assertStmtsType(instruction_set.instructions, CyGate)
        self.assertQasm(qasm_txt)

    def test_cy_reg_reg_inv(self):
        qasm_txt = 'cy q[0],r[0];\ncy q[1],r[1];\ncy q[2],r[2];'
        instruction_set = self.circuit.cy(self.qr, self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, CyGate)
        self.assertQasm(qasm_txt)

    def test_cy_reg_bit(self):
        qasm_txt = 'cy q[0],r[1];\ncy q[1],r[1];\ncy q[2],r[1];'
        instruction_set = self.circuit.cy(self.qr, self.qr2[1])
        self.assertStmtsType(instruction_set.instructions, CyGate)
        self.assertQasm(qasm_txt)

    def test_cy_reg_bit_inv(self):
        qasm_txt = 'cy q[0],r[1];\ncy q[1],r[1];\ncy q[2],r[1];'
        instruction_set = self.circuit.cy(self.qr, self.qr2[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, CyGate)
        self.assertQasm(qasm_txt)

    def test_cy_bit_reg(self):
        qasm_txt = 'cy q[1],r[0];\ncy q[1],r[1];\ncy q[1],r[2];'
        instruction_set = self.circuit.cy(self.qr[1], self.qr2)
        self.assertStmtsType(instruction_set.instructions, CyGate)
        self.assertQasm(qasm_txt)

    def test_cy_bit_reg_inv(self):
        qasm_txt = 'cy q[1],r[0];\ncy q[1],r[1];\ncy q[1],r[2];'
        instruction_set = self.circuit.cy(self.qr[1], self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, CyGate)
        self.assertQasm(qasm_txt)

    def test_cz_reg_reg(self):
        qasm_txt = 'cz q[0],r[0];\ncz q[1],r[1];\ncz q[2],r[2];'
        instruction_set = self.circuit.cz(self.qr, self.qr2)
        self.assertStmtsType(instruction_set.instructions, CzGate)
        self.assertQasm(qasm_txt)

    def test_cz_reg_reg_inv(self):
        qasm_txt = 'cz q[0],r[0];\ncz q[1],r[1];\ncz q[2],r[2];'
        instruction_set = self.circuit.cz(self.qr, self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, CzGate)
        self.assertQasm(qasm_txt)

    def test_cz_reg_bit(self):
        qasm_txt = 'cz q[0],r[1];\ncz q[1],r[1];\ncz q[2],r[1];'
        instruction_set = self.circuit.cz(self.qr, self.qr2[1])
        self.assertStmtsType(instruction_set.instructions, CzGate)
        self.assertQasm(qasm_txt)

    def test_cz_reg_bit_inv(self):
        qasm_txt = 'cz q[0],r[1];\ncz q[1],r[1];\ncz q[2],r[1];'
        instruction_set = self.circuit.cz(self.qr, self.qr2[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, CzGate)
        self.assertQasm(qasm_txt)

    def test_cz_bit_reg(self):
        qasm_txt = 'cz q[1],r[0];\ncz q[1],r[1];\ncz q[1],r[2];'
        instruction_set = self.circuit.cz(self.qr[1], self.qr2)
        self.assertStmtsType(instruction_set.instructions, CzGate)
        self.assertQasm(qasm_txt)

    def test_cz_bit_reg_inv(self):
        qasm_txt = 'cz q[1],r[0];\ncz q[1],r[1];\ncz q[1],r[2];'
        instruction_set = self.circuit.cz(self.qr[1], self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, CzGate)
        self.assertQasm(qasm_txt)

    def test_swap_reg_reg(self):
        qasm_txt = 'swap q[0],r[0];\nswap q[1],r[1];\nswap q[2],r[2];'
        instruction_set = self.circuit.swap(self.qr, self.qr2)
        self.assertStmtsType(instruction_set.instructions, SwapGate)
        self.assertQasm(qasm_txt)

    def test_swap_reg_reg_inv(self):
        qasm_txt = 'swap q[0],r[0];\nswap q[1],r[1];\nswap q[2],r[2];'
        instruction_set = self.circuit.swap(self.qr, self.qr2).inverse()
        self.assertStmtsType(instruction_set.instructions, SwapGate)
        self.assertQasm(qasm_txt)


class TestStandard3Q(StandardExtensionTest):
    """Standard Extension Test. Gates with three Qubits"""

    def setUp(self):
        self.qr = QuantumRegister(3, "q")
        self.qr2 = QuantumRegister(3, "r")
        self.qr3 = QuantumRegister(3, "s")
        self.cr = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(self.qr, self.qr2, self.qr3, self.cr)
        self.c_header = 80  # length of the header

    def test_barrier_none(self):
        self.circuit.barrier()
        qasm_txt = 'barrier q[0],q[1],q[2],r[0],r[1],r[2],s[0],s[1],s[2];'
        self.assertResult(Barrier, qasm_txt)

    def test_ccx_reg_reg_reg(self):
        qasm_txt = 'ccx q[0],r[0],s[0];\nccx q[1],r[1],s[1];\nccx q[2],r[2],s[2];'
        instruction_set = self.circuit.ccx(self.qr, self.qr2, self.qr3)
        self.assertStmtsType(instruction_set.instructions, ToffoliGate)
        self.assertQasm(qasm_txt)

    def test_ccx_reg_reg_inv(self):
        qasm_txt = 'ccx q[0],r[0],s[0];\nccx q[1],r[1],s[1];\nccx q[2],r[2],s[2];'
        instruction_set = self.circuit.ccx(self.qr, self.qr2, self.qr3).inverse()
        self.assertStmtsType(instruction_set.instructions, ToffoliGate)
        self.assertQasm(qasm_txt)

    def test_cswap_reg_reg_reg(self):
        qasm_txt = 'cswap q[0],r[0],s[0];\n' \
                   'cswap q[1],r[1],s[1];\n' \
                   'cswap q[2],r[2],s[2];'
        instruction_set = self.circuit.cswap(self.qr, self.qr2, self.qr3)
        self.assertStmtsType(instruction_set.instructions, FredkinGate)
        self.assertQasm(qasm_txt)

    def test_cswap_reg_reg_inv(self):
        qasm_txt = 'cswap q[0],r[0],s[0];\n' \
                   'cswap q[1],r[1],s[1];\n' \
                   'cswap q[2],r[2],s[2];'
        instruction_set = self.circuit.cswap(self.qr, self.qr2, self.qr3).inverse()
        self.assertStmtsType(instruction_set.instructions, FredkinGate)
        self.assertQasm(qasm_txt)


if __name__ == '__main__':
    unittest.main(verbosity=2)
