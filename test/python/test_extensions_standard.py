# -*- coding: utf-8 -*-
# pylint: disable=invalid-name,missing-docstring

# Copyright 2017 IBM RESEARCH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================

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
from qiskit.extensions.standard.s import SGate
from qiskit.extensions.standard.swap import SwapGate
from qiskit.extensions.standard.t import TGate
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
    def assertResult(self, t, qasm_txt, qasm_txt_):
        """
        Assert the single gate in self.circuit is of the type t, the QASM
        representation matches qasm_txt and the QASM representation of
        inverse maches qasm_txt_.

        Args:
            t (type): a gate type.
            qasm_txt (str): QASM representation of the gate.
            qasm_txt_ (str): QASM representation of the inverse gate.
        """
        c = self.circuit
        self.assertEqual(type(c[0]), t)
        self.assertQasm(qasm_txt)
        c[0].reapply(c)
        self.assertQasm(qasm_txt + '\n' + qasm_txt)
        self.assertEqual(c[0].inverse(), c[0])
        self.assertQasm(qasm_txt_ + '\n' + qasm_txt)

    def assertStmtsType(self, stmts, t):
        """
        Assert a list of statements stmts is of a type t.

        Args:
            stmts (list): list of statements.
            t (type): a gate type.
        """
        for stmt in stmts:
            self.assertEqual(type(stmt), t)

    def assertQasm(self, qasm_txt, offset=1):
        """
        Assert the QASM representation of the circuit self.circuit includes
        the text qasm_txt in the right position (which can be adjusted by
        offset)

        Args:
            qasm_txt (str): a string with QASM code
            offset (int): the offset in which qasm_txt should be found.
        """
        c = self.circuit
        c_txt = len(qasm_txt)
        self.assertIn('\n' + qasm_txt + '\n', c.qasm())
        self.assertEqual(self.c_header + c_txt + offset, len(c.qasm()))  # pylint: disable=no-member


class TestStandard1Q(StandardExtensionTest):
    """Standard Extension Test. Gates with a single Qubit"""

    def setUp(self):
        self.q = QuantumRegister(3, "q")
        self.r = QuantumRegister(3, "r")
        self.c = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(self.q, self.r, self.c)
        self.c_header = 69  # lenght of the header

    def test_barrier(self):
        self.circuit.barrier(self.q[1])
        qasm_txt = 'barrier q[1];'
        self.assertResult(Barrier, qasm_txt, qasm_txt)

    def test_barrier_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.barrier, self.c[0])
        self.assertRaises(QISKitError, c.barrier, self.c)
        self.assertRaises(QISKitError, c.barrier, (self.q, 3))
        self.assertRaises(QISKitError, c.barrier, (self.q, 'a'))
        self.assertRaises(QISKitError, c.barrier, 0)

    def test_barrier_reg(self):
        self.circuit.barrier(self.q)
        qasm_txt = 'barrier q[0],q[1],q[2];'
        self.assertResult(Barrier, qasm_txt, qasm_txt)

    def test_barrier_None(self):
        self.circuit.barrier()
        qasm_txt = 'barrier q[0],q[1],q[2],r[0],r[1],r[2];'
        self.assertResult(Barrier, qasm_txt, qasm_txt)

    def test_ccx(self):
        self.circuit.ccx(self.q[0], self.q[1], self.q[2])
        qasm_txt = 'ccx q[0],q[1],q[2];'
        self.assertResult(ToffoliGate, qasm_txt, qasm_txt)

    def test_ccx_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.ccx, self.c[0], self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.ccx, self.q[0], self.q[0], self.q[2])
        self.assertRaises(QISKitError, c.ccx, 0, self.q[0], self.q[2])
        self.assertRaises(QISKitError, c.ccx, (self.q, 3), self.q[1], self.q[2])
        self.assertRaises(QISKitError, c.ccx, self.c, self.q, self.q)
        self.assertRaises(QISKitError, c.ccx, 'a', self.q[1], self.q[2])

    def test_ch(self):
        self.circuit.ch(self.q[0], self.q[1])
        qasm_txt = 'ch q[0],q[1];'
        self.assertResult(CHGate, qasm_txt, qasm_txt)

    def test_ch_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.ch, self.c[0], self.c[1])
        self.assertRaises(QISKitError, c.ch, self.q[0], self.q[0])
        self.assertRaises(QISKitError, c.ch, 0, self.q[0])
        self.assertRaises(QISKitError, c.ch, (self.q, 3), self.q[0])
        self.assertRaises(QISKitError, c.ch, self.c, self.q)
        self.assertRaises(QISKitError, c.ch, 'a', self.q[1])

    def test_crz(self):
        self.circuit.crz(1, self.q[0], self.q[1])
        self.assertResult(CrzGate, 'crz(1) q[0],q[1];', 'crz(-1) q[0],q[1];')

    def test_crz_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.crz, 0, self.c[0], self.c[1])
        self.assertRaises(QISKitError, c.crz, 0, self.q[0], self.q[0])
        self.assertRaises(QISKitError, c.crz, 0, 0, self.q[0])
        # TODO self.assertRaises(QISKitError, c.crz, self.q[2], self.q[1], self.q[0])
        self.assertRaises(QISKitError, c.crz, 0, self.q[1], self.c[2])
        self.assertRaises(QISKitError, c.crz, 0, (self.q, 3), self.q[1])
        self.assertRaises(QISKitError, c.crz, 0, self.c, self.q)
        # TODO self.assertRaises(QISKitError, c.crz, 'a', self.q[1], self.q[2])

    def test_cswap(self):
        self.circuit.cswap(self.q[0], self.q[1], self.q[2])
        qasm_txt = 'cx q[2],q[1];\nccx q[0],q[1],q[2];\ncx q[2],q[1];'
        self.assertResult(FredkinGate, qasm_txt, qasm_txt)

    def test_cswap_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.cswap, self.c[0], self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cswap, self.q[1], self.q[0], self.q[0])
        self.assertRaises(QISKitError, c.cswap, self.q[1], 0, self.q[0])
        self.assertRaises(QISKitError, c.cswap, self.c[0], self.c[1], self.q[0])
        self.assertRaises(QISKitError, c.cswap, self.q[0], self.q[0], self.q[1])
        self.assertRaises(QISKitError, c.cswap, 0, self.q[0], self.q[1])
        self.assertRaises(QISKitError, c.cswap, (self.q, 3), self.q[0], self.q[1])
        self.assertRaises(QISKitError, c.cswap, self.c, self.q[0], self.q[1])
        self.assertRaises(QISKitError, c.cswap, 'a', self.q[1], self.q[2])

    def test_cu1(self):
        self.circuit.cu1(1, self.q[1], self.q[2])
        self.assertResult(Cu1Gate, 'cu1(1) q[1],q[2];', 'cu1(-1) q[1],q[2];')

    def test_cu1_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.cu1, self.c[0], self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cu1, 1, self.q[0], self.q[0])
        self.assertRaises(QISKitError, c.cu1, self.q[1], 0, self.q[0])
        self.assertRaises(QISKitError, c.cu1, 0, self.c[0], self.c[1])
        self.assertRaises(QISKitError, c.cu1, 0, self.q[0], self.q[0])
        self.assertRaises(QISKitError, c.cu1, 0, 0, self.q[0])
        # TODO self.assertRaises(QISKitError, c.cu1, self.q[2], self.q[1], self.q[0])
        self.assertRaises(QISKitError, c.cu1, 0, self.q[1], self.c[2])
        self.assertRaises(QISKitError, c.cu1, 0, (self.q, 3), self.q[1])
        self.assertRaises(QISKitError, c.cu1, 0, self.c, self.q)
        # TODO self.assertRaises(QISKitError, c.cu1, 'a', self.q[1], self.q[2])

    def test_cu3(self):
        self.circuit.cu3(1, 2, 3, self.q[1], self.q[2])
        self.assertResult(Cu3Gate, 'cu3(1,2,3) q[1],q[2];', 'cu3(-1,-3,-2) q[1],q[2];')

    def test_cu3_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.cu3, 0, 0, self.q[0], self.q[1], self.c[2])
        self.assertRaises(QISKitError, c.cu3, 0, 0, 0, self.q[0], self.q[0])
        self.assertRaises(QISKitError, c.cu3, 0, 0, self.q[1], 0, self.q[0])
        self.assertRaises(QISKitError, c.cu3, 0, 0, 0, self.q[0], self.q[0])
        self.assertRaises(QISKitError, c.cu3, 0, 0, 0, 0, self.q[0])
        self.assertRaises(QISKitError, c.cu3, 0, 0, 0, (self.q, 3), self.q[1])
        self.assertRaises(QISKitError, c.cu3, 0, 0, 0, self.c, self.q)
        # TODO self.assertRaises(QISKitError, c.cu3, 0, 0, 'a', self.q[1], self.q[2])

    def test_cx(self):
        self.circuit.cx(self.q[1], self.q[2])
        qasm_txt = 'cx q[1],q[2];'
        self.assertResult(CnotGate, qasm_txt, qasm_txt)

    def test_cx_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.cx, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cx, self.q[0], self.q[0])
        self.assertRaises(QISKitError, c.cx, 0, self.q[0])
        self.assertRaises(QISKitError, c.cx, (self.q, 3), self.q[0])
        self.assertRaises(QISKitError, c.cx, self.c, self.q)
        self.assertRaises(QISKitError, c.cx, 'a', self.q[1])

    def test_cxbase(self):
        qasm_txt = 'CX q[1],q[2];'
        self.circuit.cx_base(self.q[1], self.q[2])
        self.assertResult(CXBase, qasm_txt, qasm_txt)

    def test_cxbase_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.cx_base, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cx_base, self.q[0], self.q[0])
        self.assertRaises(QISKitError, c.cx_base, 0, self.q[0])
        self.assertRaises(QISKitError, c.cx_base, (self.q, 3), self.q[0])
        self.assertRaises(QISKitError, c.cx_base, self.c, self.q)
        self.assertRaises(QISKitError, c.cx_base, 'a', self.q[1])

    def test_cy(self):
        qasm_txt = 'cy q[1],q[2];'
        self.circuit.cy(self.q[1], self.q[2])
        self.assertResult(CyGate, qasm_txt, qasm_txt)

    def test_cy_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.cy, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cy, self.q[0], self.q[0])
        self.assertRaises(QISKitError, c.cy, 0, self.q[0])
        self.assertRaises(QISKitError, c.cy, (self.q, 3), self.q[0])
        self.assertRaises(QISKitError, c.cy, self.c, self.q)
        self.assertRaises(QISKitError, c.cy, 'a', self.q[1])

    def test_cz(self):
        qasm_txt = 'cz q[1],q[2];'
        self.circuit.cz(self.q[1], self.q[2])
        self.assertResult(CzGate, qasm_txt, qasm_txt)

    def test_cz_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.cz, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cz, self.q[0], self.q[0])
        self.assertRaises(QISKitError, c.cz, 0, self.q[0])
        self.assertRaises(QISKitError, c.cz, (self.q, 3), self.q[0])
        self.assertRaises(QISKitError, c.cz, self.c, self.q)
        self.assertRaises(QISKitError, c.cz, 'a', self.q[1])

    def test_h(self):
        qasm_txt = 'h q[1];'
        self.circuit.h(self.q[1])
        self.assertResult(HGate, qasm_txt, qasm_txt)

    def test_h_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.h, self.c[0])
        self.assertRaises(QISKitError, c.h, self.c)
        self.assertRaises(QISKitError, c.h, (self.q, 3))
        self.assertRaises(QISKitError, c.h, (self.q, 'a'))
        self.assertRaises(QISKitError, c.h, 0)

    def test_h_reg(self):
        qasm_txt = 'h q[0];\nh q[1];\nh q[2];'
        instruction_set = self.circuit.h(self.q)
        self.assertStmtsType(instruction_set.instructions, HGate)
        self.assertQasm(qasm_txt)

    def test_h_reg_inv(self):
        qasm_txt = 'h q[0];\nh q[1];\nh q[2];'
        instruction_set = self.circuit.h(self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, HGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 22)

    def test_iden(self):
        self.circuit.iden(self.q[1])
        self.assertResult(IdGate, 'id q[1];', 'id q[1];')

    def test_iden_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.iden, self.c[0])
        self.assertRaises(QISKitError, c.iden, self.c)
        self.assertRaises(QISKitError, c.iden, (self.q, 3))
        self.assertRaises(QISKitError, c.iden, (self.q, 'a'))
        self.assertRaises(QISKitError, c.iden, 0)

    def test_iden_reg(self):
        qasm_txt = 'id q[0];\nid q[1];\nid q[2];'
        instruction_set = self.circuit.iden(self.q)
        self.assertStmtsType(instruction_set.instructions, IdGate)
        self.assertQasm(qasm_txt)

    def test_iden_reg_inv(self):
        qasm_txt = 'id q[0];\nid q[1];\nid q[2];'
        instruction_set = self.circuit.iden(self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, IdGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 25)

    def test_rx(self):
        self.circuit.rx(1, self.q[1])
        self.assertResult(RXGate, 'rx(1) q[1];', 'rx(-1) q[1];')

    def test_rx_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.rx, self.c[0], self.c[1])
        self.assertRaises(QISKitError, c.rx, self.q[1], 0)
        self.assertRaises(QISKitError, c.rx, 0, self.c[0])
        self.assertRaises(QISKitError, c.rx, 0, 0)
        # TODO self.assertRaises(QISKitError, c.rx, self.q[2], self.q[1])
        self.assertRaises(QISKitError, c.rx, 0, (self.q, 3))
        self.assertRaises(QISKitError, c.rx, 0, self.c)
        # TODO self.assertRaises(QISKitError, c.rx, 'a', self.q[1])
        self.assertRaises(QISKitError, c.rx, 0, 'a')

    def test_rx_reg(self):
        qasm_txt = 'rx(1) q[0];\nrx(1) q[1];\nrx(1) q[2];'
        instruction_set = self.circuit.rx(1, self.q)
        self.assertStmtsType(instruction_set.instructions, RXGate)
        self.assertQasm(qasm_txt)

    def test_rx_reg_inv(self):
        qasm_txt = 'rx(-1) q[0];\nrx(-1) q[1];\nrx(-1) q[2];'
        instruction_set = self.circuit.rx(1, self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, RXGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 37)

    def test_rx_pi(self):
        c = self.circuit
        c.rx(pi / 2, self.q[1])
        self.assertResult(RXGate, 'rx(pi/2) q[1];', 'rx(-pi/2) q[1];')

    def test_ry(self):
        self.circuit.ry(1, self.q[1])
        self.assertResult(RYGate, 'ry(1) q[1];', 'ry(-1) q[1];')

    def test_ry_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.ry, self.c[0], self.c[1])
        self.assertRaises(QISKitError, c.ry, self.q[1], 0)
        self.assertRaises(QISKitError, c.ry, 0, self.c[0])
        self.assertRaises(QISKitError, c.ry, 0, 0)
        # TODO self.assertRaises(QISKitError, c.ry, self.q[2], self.q[1])
        self.assertRaises(QISKitError, c.ry, 0, (self.q, 3))
        self.assertRaises(QISKitError, c.ry, 0, self.c)
        # TODO self.assertRaises(QISKitError, c.ry, 'a', self.q[1])
        self.assertRaises(QISKitError, c.ry, 0, 'a')

    def test_ry_reg(self):
        qasm_txt = 'ry(1) q[0];\nry(1) q[1];\nry(1) q[2];'
        instruction_set = self.circuit.ry(1, self.q)
        self.assertStmtsType(instruction_set.instructions, RYGate)
        self.assertQasm(qasm_txt)

    def test_ry_reg_inv(self):
        qasm_txt = 'ry(-1) q[0];\nry(-1) q[1];\nry(-1) q[2];'
        instruction_set = self.circuit.ry(1, self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, RYGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 37)

    def test_ry_pi(self):
        c = self.circuit
        c.ry(pi / 2, self.q[1])
        self.assertResult(RYGate, 'ry(pi/2) q[1];', 'ry(-pi/2) q[1];')

    def test_rz(self):
        self.circuit.rz(1, self.q[1])
        self.assertResult(RZGate, 'rz(1) q[1];', 'rz(-1) q[1];')

    def test_rz_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.rz, self.c[0], self.c[1])
        self.assertRaises(QISKitError, c.rz, self.q[1], 0)
        self.assertRaises(QISKitError, c.rz, 0, self.c[0])
        self.assertRaises(QISKitError, c.rz, 0, 0)
        # TODO self.assertRaises(QISKitError, c.rz, self.q[2], self.q[1])
        self.assertRaises(QISKitError, c.rz, 0, (self.q, 3))
        self.assertRaises(QISKitError, c.rz, 0, self.c)
        # TODO self.assertRaises(QISKitError, c.rz, 'a', self.q[1])
        self.assertRaises(QISKitError, c.rz, 0, 'a')

    def test_rz_reg(self):
        qasm_txt = 'rz(1) q[0];\nrz(1) q[1];\nrz(1) q[2];'
        instruction_set = self.circuit.rz(1, self.q)
        self.assertStmtsType(instruction_set.instructions, RZGate)
        self.assertQasm(qasm_txt)

    def test_rz_reg_inv(self):
        qasm_txt = 'rz(-1) q[0];\nrz(-1) q[1];\nrz(-1) q[2];'
        instruction_set = self.circuit.rz(1, self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, RZGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 37)

    def test_rz_pi(self):
        c = self.circuit
        c.rz(pi / 2, self.q[1])
        self.assertResult(RZGate, 'rz(pi/2) q[1];', 'rz(-pi/2) q[1];')

    def test_s(self):
        self.circuit.s(self.q[1])
        self.assertResult(SGate, 's q[1];', 'sdg q[1];')

    def test_s_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.s, self.c[0])
        self.assertRaises(QISKitError, c.s, self.c)
        self.assertRaises(QISKitError, c.s, (self.q, 3))
        self.assertRaises(QISKitError, c.s, (self.q, 'a'))
        self.assertRaises(QISKitError, c.s, 0)

    def test_s_reg(self):
        qasm_txt = 's q[0];\ns q[1];\ns q[2];'
        instruction_set = self.circuit.s(self.q)
        self.assertStmtsType(instruction_set.instructions, SGate)
        self.assertQasm(qasm_txt)

    def test_s_reg_inv(self):
        qasm_txt = 'sdg q[0];\nsdg q[1];\nsdg q[2];'
        instruction_set = self.circuit.s(self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, SGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 28)

    def test_sdg(self):
        self.circuit.sdg(self.q[1])
        self.assertResult(SGate, 'sdg q[1];', 's q[1];')

    def test_sdg_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.sdg, self.c[0])
        self.assertRaises(QISKitError, c.sdg, self.c)
        self.assertRaises(QISKitError, c.sdg, (self.q, 3))
        self.assertRaises(QISKitError, c.sdg, (self.q, 'a'))
        self.assertRaises(QISKitError, c.sdg, 0)

    def test_sdg_reg(self):
        qasm_txt = 'sdg q[0];\nsdg q[1];\nsdg q[2];'
        instruction_set = self.circuit.sdg(self.q)
        self.assertStmtsType(instruction_set.instructions, SGate)
        self.assertQasm(qasm_txt)

    def test_sdg_reg_inv(self):
        qasm_txt = 's q[0];\ns q[1];\ns q[2];'
        instruction_set = self.circuit.sdg(self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, SGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 22)

    def test_swap(self):
        self.circuit.swap(self.q[1], self.q[2])
        self.assertResult(SwapGate, 'swap q[1],q[2];', 'swap q[1],q[2];')

    def test_swap_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.swap, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.swap, self.q[0], self.q[0])
        self.assertRaises(QISKitError, c.swap, 0, self.q[0])
        self.assertRaises(QISKitError, c.swap, (self.q, 3), self.q[0])
        self.assertRaises(QISKitError, c.swap, self.c, self.q)
        self.assertRaises(QISKitError, c.swap, 'a', self.q[1])
        self.assertRaises(QISKitError, c.swap, self.q, self.r[1])
        self.assertRaises(QISKitError, c.swap, self.q[1], self.r)

    def test_t(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.t, self.c[0])
        # TODO self.assertRaises(QISKitError, c.t, 1)
        c.t(self.q[1])
        self.assertResult(TGate, 't q[1];', 'tdg q[1];')

    def test_t_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.t, self.c[0])
        self.assertRaises(QISKitError, c.t, self.c)
        self.assertRaises(QISKitError, c.t, (self.q, 3))
        self.assertRaises(QISKitError, c.t, (self.q, 'a'))
        self.assertRaises(QISKitError, c.t, 0)

    def test_t_reg(self):
        qasm_txt = 't q[0];\nt q[1];\nt q[2];'
        instruction_set = self.circuit.t(self.q)
        self.assertStmtsType(instruction_set.instructions, TGate)
        self.assertQasm(qasm_txt)

    def test_t_reg_inv(self):
        qasm_txt = 'tdg q[0];\ntdg q[1];\ntdg q[2];'
        instruction_set = self.circuit.t(self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, TGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 28)

    def test_tdg(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.tdg, self.c[0])
        # TODO self.assertRaises(QISKitError, c.tdg, 1)
        c.tdg(self.q[1])
        self.assertResult(TGate, 'tdg q[1];', 't q[1];')

    def test_tdg_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.tdg, self.c[0])
        self.assertRaises(QISKitError, c.tdg, self.c)
        self.assertRaises(QISKitError, c.tdg, (self.q, 3))
        self.assertRaises(QISKitError, c.tdg, (self.q, 'a'))
        self.assertRaises(QISKitError, c.tdg, 0)

    def test_tdg_reg(self):
        qasm_txt = 'tdg q[0];\ntdg q[1];\ntdg q[2];'
        instruction_set = self.circuit.tdg(self.q)
        self.assertStmtsType(instruction_set.instructions, TGate)
        self.assertQasm(qasm_txt)

    def test_tdg_reg_inv(self):
        qasm_txt = 't q[0];\nt q[1];\nt q[2];'
        instruction_set = self.circuit.tdg(self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, TGate)
        self.assertQasm(qasm_txt, offset=len(qasm_txt) - 22)

    def test_u1(self):
        self.circuit.u1(1, self.q[1])
        self.assertResult(U1Gate, 'u1(1) q[1];', 'u1(-1) q[1];')

    def test_u1_invalid(self):
        c = self.circuit
        # CHECKME? self.assertRaises(QISKitError, c.u1, self.c[0], self.q[0])
        self.assertRaises(QISKitError, c.u1, self.c[0], self.c[1])
        self.assertRaises(QISKitError, c.u1, self.q[1], 0)
        self.assertRaises(QISKitError, c.u1, 0, self.c[0])
        self.assertRaises(QISKitError, c.u1, 0, 0)
        # TODO self.assertRaises(QISKitError, c.u1, self.q[2], self.q[1])
        self.assertRaises(QISKitError, c.u1, 0, (self.q, 3))
        self.assertRaises(QISKitError, c.u1, 0, self.c)
        # TODO self.assertRaises(QISKitError, c.u1, 'a', self.q[1])
        self.assertRaises(QISKitError, c.u1, 0, 'a')

    def test_u1_reg(self):
        qasm_txt = 'u1(1) q[0];\nu1(1) q[1];\nu1(1) q[2];'
        instruction_set = self.circuit.u1(1, self.q)
        self.assertStmtsType(instruction_set.instructions, U1Gate)
        self.assertQasm(qasm_txt)

    def test_u1_reg_inv(self):
        qasm_txt = 'u1(-1) q[0];\nu1(-1) q[1];\nu1(-1) q[2];'
        instruction_set = self.circuit.u1(1, self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, U1Gate)
        self.assertQasm(qasm_txt)

    def test_u1_pi(self):
        c = self.circuit
        c.u1(pi / 2, self.q[1])
        self.assertResult(U1Gate, 'u1(pi/2) q[1];', 'u1(-pi/2) q[1];')

    def test_u2(self):
        self.circuit.u2(1, 2, self.q[1])
        self.assertResult(U2Gate, 'u2(1,2) q[1];', 'u2(-pi - 2,-1 + pi) q[1];')

    def test_u2_invalid(self):
        c = self.circuit
        # CHECKME? self.assertRaises(QISKitError, c.u2, 0, self.c[0], self.q[0])
        self.assertRaises(QISKitError, c.u2, 0, self.c[0], self.c[1])
        self.assertRaises(QISKitError, c.u2, 0, self.q[1], 0)
        self.assertRaises(QISKitError, c.u2, 0, 0, self.c[0])
        self.assertRaises(QISKitError, c.u2, 0, 0, 0)
        # TODO self.assertRaises(QISKitError, c.u2, 0, self.q[2], self.q[1])
        self.assertRaises(QISKitError, c.u2, 0, 0, (self.q, 3))
        self.assertRaises(QISKitError, c.u2, 0, 0, self.c)
        # TODO self.assertRaises(QISKitError, c.u2, 0, 'a', self.q[1])
        self.assertRaises(QISKitError, c.u2, 0, 0, 'a')

    def test_u2_reg(self):
        qasm_txt = 'u2(1,2) q[0];\nu2(1,2) q[1];\nu2(1,2) q[2];'
        instruction_set = self.circuit.u2(1, 2, self.q)
        self.assertStmtsType(instruction_set.instructions, U2Gate)
        self.assertQasm(qasm_txt)

    def test_u2_reg_inv(self):
        qasm_txt = 'u2(-pi - 2,-1 + pi) q[0];\nu2(-pi - 2,-1 + pi) q[1];\nu2(-pi - 2,-1 + pi) q[2];'
        instruction_set = self.circuit.u2(1, 2, self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, U2Gate)
        self.assertQasm(qasm_txt)

    def test_u2_pi(self):
        c = self.circuit
        c.u2(pi / 2, 0.3 * pi, self.q[1])
        self.assertResult(U2Gate, 'u2(pi/2,0.3*pi) q[1];', 'u2(-1.3*pi,pi/2) q[1];')

    def test_u3(self):
        self.circuit.u3(1, 2, 3, self.q[1])
        self.assertResult(U3Gate, 'u3(1,2,3) q[1];', 'u3(-1,-3,-2) q[1];')

    def test_u3_invalid(self):
        c = self.circuit
        # CHECKME? self.assertRaises(QISKitError, c.u3, 0, self.c[0], self.q[0])
        self.assertRaises(QISKitError, c.u3, 0, 0, self.c[0], self.c[1])
        self.assertRaises(QISKitError, c.u3, 0, 0, self.q[1], 0)
        self.assertRaises(QISKitError, c.u3, 0, 0, 0, self.c[0])
        self.assertRaises(QISKitError, c.u3, 0, 0, 0, 0)
        # TODO self.assertRaises(QISKitError, c.u3, 0, 0, self.q[2], self.q[1])
        self.assertRaises(QISKitError, c.u3, 0, 0, 0, (self.q, 3))
        self.assertRaises(QISKitError, c.u3, 0, 0, 0, self.c)
        # TODO self.assertRaises(QISKitError, c.u3, 0, 0, 'a', self.q[1])
        self.assertRaises(QISKitError, c.u3, 0, 0, 0, 'a')

    def test_u3_reg(self):
        qasm_txt = 'u3(1,2,3) q[0];\nu3(1,2,3) q[1];\nu3(1,2,3) q[2];'
        instruction_set = self.circuit.u3(1, 2, 3, self.q)
        self.assertStmtsType(instruction_set.instructions, U3Gate)
        self.assertQasm(qasm_txt)

    def test_u3_reg_inv(self):
        qasm_txt = 'u3(-1,-3,-2) q[0];\nu3(-1,-3,-2) q[1];\nu3(-1,-3,-2) q[2];'
        instruction_set = self.circuit.u3(1, 2, 3, self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, U3Gate)
        self.assertQasm(qasm_txt)

    def test_u3_pi(self):
        c = self.circuit
        c.u3(pi, pi / 2, 0.3 * pi, self.q[1])
        self.assertResult(U3Gate, 'u3(pi,pi/2,0.3*pi) q[1];', 'u3(-pi,-0.3*pi,-pi/2) q[1];')

    def test_ubase(self):
        self.circuit.u_base(1, 2, 3, self.q[1])
        self.assertResult(UBase, 'U(1,2,3) q[1];', 'U(-1,-3,-2) q[1];')

    def test_ubase_invalid(self):
        c = self.circuit
        # CHECKME? self.assertRaises(QISKitError, c.u_base, 0, self.c[0], self.q[0])
        self.assertRaises(QISKitError, c.u_base, 0, 0, self.c[0], self.c[1])
        self.assertRaises(QISKitError, c.u_base, 0, 0, self.q[1], 0)
        self.assertRaises(QISKitError, c.u_base, 0, 0, 0, self.c[0])
        self.assertRaises(QISKitError, c.u_base, 0, 0, 0, 0)
        # TODO self.assertRaises(QISKitError, c.u_base, 0, 0, self.q[2], self.q[1])
        self.assertRaises(QISKitError, c.u_base, 0, 0, 0, (self.q, 3))
        self.assertRaises(QISKitError, c.u_base, 0, 0, 0, self.c)
        # TODO self.assertRaises(QISKitError, c.u_base, 0, 0, 'a', self.q[1])
        self.assertRaises(QISKitError, c.u_base, 0, 0, 0, 'a')

    def test_ubase_reg(self):
        qasm_txt = 'U(1,2,3) q[0];\nU(1,2,3) q[1];\nU(1,2,3) q[2];'
        instruction_set = self.circuit.u_base(1, 2, 3, self.q)
        self.assertStmtsType(instruction_set.instructions, UBase)
        self.assertQasm(qasm_txt)

    def test_ubase_reg_inv(self):
        qasm_txt = 'U(-1,-3,-2) q[0];\nU(-1,-3,-2) q[1];\nU(-1,-3,-2) q[2];'
        instruction_set = self.circuit.u_base(1, 2, 3, self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, UBase)
        self.assertQasm(qasm_txt)

    def test_ubase_pi(self):
        c = self.circuit
        c.u_base(pi, pi / 2, 0.3 * pi, self.q[1])
        self.assertResult(UBase, 'U(pi,pi/2,0.3*pi) q[1];', 'U(-pi,-0.3*pi,-pi/2) q[1];')

    def test_x(self):
        self.circuit.x(self.q[1])
        self.assertResult(XGate, 'x q[1];', 'x q[1];')

    def test_x_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.x, self.c[0])
        self.assertRaises(QISKitError, c.x, self.c)
        self.assertRaises(QISKitError, c.x, (self.q, 3))
        self.assertRaises(QISKitError, c.x, (self.q, 'a'))
        self.assertRaises(QISKitError, c.x, 0)

    def test_x_reg(self):
        qasm_txt = 'x q[0];\nx q[1];\nx q[2];'
        instruction_set = self.circuit.x(self.q)
        self.assertStmtsType(instruction_set.instructions, XGate)
        self.assertQasm(qasm_txt)

    def test_x_reg_inv(self):
        qasm_txt = 'x q[0];\nx q[1];\nx q[2];'
        instruction_set = self.circuit.x(self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, XGate)
        self.assertQasm(qasm_txt)

    def test_y(self):
        self.circuit.y(self.q[1])
        self.assertResult(YGate, 'y q[1];', 'y q[1];')

    def test_y_invalid(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.y, self.c[0])
        self.assertRaises(QISKitError, c.y, self.c)
        self.assertRaises(QISKitError, c.y, (self.q, 3))
        self.assertRaises(QISKitError, c.y, (self.q, 'a'))
        self.assertRaises(QISKitError, c.y, 0)

    def test_y_reg(self):
        qasm_txt = 'y q[0];\ny q[1];\ny q[2];'
        instruction_set = self.circuit.y(self.q)
        self.assertStmtsType(instruction_set.instructions, YGate)
        self.assertQasm(qasm_txt)

    def test_y_reg_inv(self):
        qasm_txt = 'y q[0];\ny q[1];\ny q[2];'
        instruction_set = self.circuit.y(self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, YGate)
        self.assertQasm(qasm_txt)

    def test_z(self):
        self.circuit.z(self.q[1])
        self.assertResult(ZGate, 'z q[1];', 'z q[1];')

    def test_rzz(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.rzz, 0.1, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.rzz, 0.1, self.q[0], self.q[0])
        c.rzz(pi/2, self.q[1], self.q[2])
        self.assertResult(RZZGate, 'rzz(pi/2) q[1],q[2];', 'rzz(-pi/2) q[1],q[2];')

    def assertResult(self, t, qasm_txt, qasm_txt_):
        """
        t: type
        qasm_txt: qasm representation
        qasm_txt_: qasm representation of inverse
        """
        c = self.circuit
        self.assertRaises(QISKitError, c.z, self.c[0])
        self.assertRaises(QISKitError, c.z, self.c)
        self.assertRaises(QISKitError, c.z, (self.q, 3))
        self.assertRaises(QISKitError, c.z, (self.q, 'a'))
        self.assertRaises(QISKitError, c.z, 0)

    def test_z_reg(self):
        qasm_txt = 'z q[0];\nz q[1];\nz q[2];'
        instruction_set = self.circuit.z(self.q)
        self.assertStmtsType(instruction_set.instructions, ZGate)
        self.assertQasm(qasm_txt)

    def test_z_reg_inv(self):
        qasm_txt = 'z q[0];\nz q[1];\nz q[2];'
        instruction_set = self.circuit.z(self.q).inverse()
        self.assertStmtsType(instruction_set.instructions, ZGate)
        self.assertQasm(qasm_txt)


class TestStandard2Q(StandardExtensionTest):
    """Standard Extension Test. Gates with two Qubits"""

    def setUp(self):
        self.q = QuantumRegister(3, "q")
        self.r = QuantumRegister(3, "r")
        self.c = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(self.q, self.r, self.c)
        self.c_header = 69  # lenght of the header

    def test_barrier_None(self):
        self.circuit.barrier()
        qasm_txt = 'barrier q[0],q[1],q[2],r[0],r[1],r[2];'
        self.assertResult(Barrier, qasm_txt, qasm_txt)

    def test_barrier_reg_bit(self):
        self.circuit.barrier(self.q, self.r[0])
        qasm_txt = 'barrier q[0],q[1],q[2],r[0];'
        self.assertResult(Barrier, qasm_txt, qasm_txt)

    def test_ch_reg_reg(self):
        qasm_txt = 'ch q[0],r[0];\nch q[1],r[1];\nch q[2],r[2];'
        instruction_set = self.circuit.ch(self.q, self.r)
        self.assertStmtsType(instruction_set.instructions, CHGate)
        self.assertQasm(qasm_txt)

    def test_ch_reg_reg_inv(self):
        qasm_txt = 'ch q[0],r[0];\nch q[1],r[1];\nch q[2],r[2];'
        instruction_set = self.circuit.ch(self.q, self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, CHGate)
        self.assertQasm(qasm_txt)

    def test_ch_reg_bit(self):
        qasm_txt = 'ch q[0],r[1];\nch q[1],r[1];\nch q[2],r[1];'
        instruction_set = self.circuit.ch(self.q, self.r[1])
        self.assertStmtsType(instruction_set.instructions, CHGate)
        self.assertQasm(qasm_txt)

    def test_ch_reg_bit_inv(self):
        qasm_txt = 'ch q[0],r[1];\nch q[1],r[1];\nch q[2],r[1];'
        instruction_set = self.circuit.ch(self.q, self.r[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, CHGate)
        self.assertQasm(qasm_txt)

    def test_ch_bit_reg(self):
        qasm_txt = 'ch q[1],r[0];\nch q[1],r[1];\nch q[1],r[2];'
        instruction_set = self.circuit.ch(self.q[1], self.r)
        self.assertStmtsType(instruction_set.instructions, CHGate)
        self.assertQasm(qasm_txt)

    def test_ch_bit_reg_inv(self):
        qasm_txt = 'ch q[1],r[0];\nch q[1],r[1];\nch q[1],r[2];'
        instruction_set = self.circuit.ch(self.q[1], self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, CHGate)
        self.assertQasm(qasm_txt)

    def test_crz_reg_reg(self):
        qasm_txt = 'crz(1) q[0],r[0];\ncrz(1) q[1],r[1];\ncrz(1) q[2],r[2];'
        instruction_set = self.circuit.crz(1, self.q, self.r)
        self.assertStmtsType(instruction_set.instructions, CrzGate)
        self.assertQasm(qasm_txt)

    def test_crz_reg_reg_inv(self):
        qasm_txt = 'crz(-1) q[0],r[0];\ncrz(-1) q[1],r[1];\ncrz(-1) q[2],r[2];'
        instruction_set = self.circuit.crz(1, self.q, self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, CrzGate)
        self.assertQasm(qasm_txt)

    def test_crz_reg_bit(self):
        qasm_txt = 'crz(1) q[0],r[1];\ncrz(1) q[1],r[1];\ncrz(1) q[2],r[1];'
        instruction_set = self.circuit.crz(1, self.q, self.r[1])
        self.assertStmtsType(instruction_set.instructions, CrzGate)
        self.assertQasm(qasm_txt)

    def test_crz_reg_bit_inv(self):
        qasm_txt = 'crz(-1) q[0],r[1];\ncrz(-1) q[1],r[1];\ncrz(-1) q[2],r[1];'
        instruction_set = self.circuit.crz(1, self.q, self.r[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, CrzGate)
        self.assertQasm(qasm_txt)

    def test_crz_bit_reg(self):
        qasm_txt = 'crz(1) q[1],r[0];\ncrz(1) q[1],r[1];\ncrz(1) q[1],r[2];'
        instruction_set = self.circuit.crz(1, self.q[1], self.r)
        self.assertStmtsType(instruction_set.instructions, CrzGate)
        self.assertQasm(qasm_txt)

    def test_crz_bit_reg_inv(self):
        qasm_txt = 'crz(-1) q[1],r[0];\ncrz(-1) q[1],r[1];\ncrz(-1) q[1],r[2];'
        instruction_set = self.circuit.crz(1, self.q[1], self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, CrzGate)
        self.assertQasm(qasm_txt)

    def test_cu1_reg_reg(self):
        qasm_txt = 'cu1(1) q[0],r[0];\ncu1(1) q[1],r[1];\ncu1(1) q[2],r[2];'
        instruction_set = self.circuit.cu1(1, self.q, self.r)
        self.assertStmtsType(instruction_set.instructions, Cu1Gate)
        self.assertQasm(qasm_txt)

    def test_cu1_reg_reg_inv(self):
        qasm_txt = 'cu1(-1) q[0],r[0];\ncu1(-1) q[1],r[1];\ncu1(-1) q[2],r[2];'
        instruction_set = self.circuit.cu1(1, self.q, self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, Cu1Gate)
        self.assertQasm(qasm_txt)

    def test_cu1_reg_bit(self):
        qasm_txt = 'cu1(1) q[0],r[1];\ncu1(1) q[1],r[1];\ncu1(1) q[2],r[1];'
        instruction_set = self.circuit.cu1(1, self.q, self.r[1])
        self.assertStmtsType(instruction_set.instructions, Cu1Gate)
        self.assertQasm(qasm_txt)

    def test_cu1_reg_bit_inv(self):
        qasm_txt = 'cu1(-1) q[0],r[1];\ncu1(-1) q[1],r[1];\ncu1(-1) q[2],r[1];'
        instruction_set = self.circuit.cu1(1, self.q, self.r[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, Cu1Gate)
        self.assertQasm(qasm_txt)

    def test_cu1_bit_reg(self):
        qasm_txt = 'cu1(1) q[1],r[0];\ncu1(1) q[1],r[1];\ncu1(1) q[1],r[2];'
        instruction_set = self.circuit.cu1(1, self.q[1], self.r)
        self.assertStmtsType(instruction_set.instructions, Cu1Gate)
        self.assertQasm(qasm_txt)

    def test_cu1_bit_reg_inv(self):
        qasm_txt = 'cu1(-1) q[1],r[0];\ncu1(-1) q[1],r[1];\ncu1(-1) q[1],r[2];'
        instruction_set = self.circuit.cu1(1, self.q[1], self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, Cu1Gate)
        self.assertQasm(qasm_txt)

    def test_cu3_reg_reg(self):
        qasm_txt = 'cu3(1,2,3) q[0],r[0];\ncu3(1,2,3) q[1],r[1];\ncu3(1,2,3) q[2],r[2];'
        instruction_set = self.circuit.cu3(1, 2, 3, self.q, self.r)
        self.assertStmtsType(instruction_set.instructions, Cu3Gate)
        self.assertQasm(qasm_txt)

    def test_cu3_reg_reg_inv(self):
        qasm_txt = 'cu3(-1,-3,-2) q[0],r[0];\ncu3(-1,-3,-2) q[1],r[1];\ncu3(-1,-3,-2) q[2],r[2];'
        instruction_set = self.circuit.cu3(1, 2, 3, self.q, self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, Cu3Gate)
        self.assertQasm(qasm_txt)

    def test_cu3_reg_bit(self):
        qasm_txt = 'cu3(1,2,3) q[0],r[1];\ncu3(1,2,3) q[1],r[1];\ncu3(1,2,3) q[2],r[1];'
        instruction_set = self.circuit.cu3(1, 2, 3, self.q, self.r[1])
        self.assertStmtsType(instruction_set.instructions, Cu3Gate)
        self.assertQasm(qasm_txt)

    def test_cu3_reg_bit_inv(self):
        qasm_txt = 'cu3(-1,-3,-2) q[0],r[1];\ncu3(-1,-3,-2) q[1],r[1];\ncu3(-1,-3,-2) q[2],r[1];'
        instruction_set = self.circuit.cu3(1, 2, 3, self.q, self.r[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, Cu3Gate)
        self.assertQasm(qasm_txt)

    def test_cu3_bit_reg(self):
        qasm_txt = 'cu3(1,2,3) q[1],r[0];\ncu3(1,2,3) q[1],r[1];\ncu3(1,2,3) q[1],r[2];'
        instruction_set = self.circuit.cu3(1, 2, 3, self.q[1], self.r)
        self.assertStmtsType(instruction_set.instructions, Cu3Gate)
        self.assertQasm(qasm_txt)

    def test_cu3_bit_reg_inv(self):
        qasm_txt = 'cu3(-1,-3,-2) q[1],r[0];\ncu3(-1,-3,-2) q[1],r[1];\ncu3(-1,-3,-2) q[1],r[2];'
        instruction_set = self.circuit.cu3(1, 2, 3, self.q[1], self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, Cu3Gate)
        self.assertQasm(qasm_txt)

    def test_cx_reg_reg(self):
        qasm_txt = 'cx q[0],r[0];\ncx q[1],r[1];\ncx q[2],r[2];'
        instruction_set = self.circuit.cx(self.q, self.r)
        self.assertStmtsType(instruction_set.instructions, CnotGate)
        self.assertQasm(qasm_txt)

    def test_cx_reg_reg_inv(self):
        qasm_txt = 'cx q[0],r[0];\ncx q[1],r[1];\ncx q[2],r[2];'
        instruction_set = self.circuit.cx(self.q, self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, CnotGate)
        self.assertQasm(qasm_txt)

    def test_cx_reg_bit(self):
        qasm_txt = 'cx q[0],r[1];\ncx q[1],r[1];\ncx q[2],r[1];'
        instruction_set = self.circuit.cx(self.q, self.r[1])
        self.assertStmtsType(instruction_set.instructions, CnotGate)
        self.assertQasm(qasm_txt)

    def test_cx_reg_bit_inv(self):
        qasm_txt = 'cx q[0],r[1];\ncx q[1],r[1];\ncx q[2],r[1];'
        instruction_set = self.circuit.cx(self.q, self.r[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, CnotGate)
        self.assertQasm(qasm_txt)

    def test_cx_bit_reg(self):
        qasm_txt = 'cx q[1],r[0];\ncx q[1],r[1];\ncx q[1],r[2];'
        instruction_set = self.circuit.cx(self.q[1], self.r)
        self.assertStmtsType(instruction_set.instructions, CnotGate)
        self.assertQasm(qasm_txt)

    def test_cx_bit_reg_inv(self):
        qasm_txt = 'cx q[1],r[0];\ncx q[1],r[1];\ncx q[1],r[2];'
        instruction_set = self.circuit.cx(self.q[1], self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, CnotGate)
        self.assertQasm(qasm_txt)

    def test_cxbase_reg_reg(self):
        qasm_txt = 'CX q[0],r[0];\nCX q[1],r[1];\nCX q[2],r[2];'
        instruction_set = self.circuit.cx_base(self.q, self.r)
        self.assertStmtsType(instruction_set.instructions, CXBase)
        self.assertQasm(qasm_txt)

    def test_cxbase_reg_reg_inv(self):
        qasm_txt = 'CX q[0],r[0];\nCX q[1],r[1];\nCX q[2],r[2];'
        instruction_set = self.circuit.cx_base(self.q, self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, CXBase)
        self.assertQasm(qasm_txt)

    def test_cxbase_reg_bit(self):
        qasm_txt = 'CX q[0],r[1];\nCX q[1],r[1];\nCX q[2],r[1];'
        instruction_set = self.circuit.cx_base(self.q, self.r[1])
        self.assertStmtsType(instruction_set.instructions, CXBase)
        self.assertQasm(qasm_txt)

    def test_cxbase_reg_bit_inv(self):
        qasm_txt = 'CX q[0],r[1];\nCX q[1],r[1];\nCX q[2],r[1];'
        instruction_set = self.circuit.cx_base(self.q, self.r[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, CXBase)
        self.assertQasm(qasm_txt)

    def test_cxbase_bit_reg(self):
        qasm_txt = 'CX q[1],r[0];\nCX q[1],r[1];\nCX q[1],r[2];'
        instruction_set = self.circuit.cx_base(self.q[1], self.r)
        self.assertStmtsType(instruction_set.instructions, CXBase)
        self.assertQasm(qasm_txt)

    def test_cxbase_bit_reg_inv(self):
        qasm_txt = 'CX q[1],r[0];\nCX q[1],r[1];\nCX q[1],r[2];'
        instruction_set = self.circuit.cx_base(self.q[1], self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, CXBase)
        self.assertQasm(qasm_txt)

    def test_cy_reg_reg(self):
        qasm_txt = 'cy q[0],r[0];\ncy q[1],r[1];\ncy q[2],r[2];'
        instruction_set = self.circuit.cy(self.q, self.r)
        self.assertStmtsType(instruction_set.instructions, CyGate)
        self.assertQasm(qasm_txt)

    def test_cy_reg_reg_inv(self):
        qasm_txt = 'cy q[0],r[0];\ncy q[1],r[1];\ncy q[2],r[2];'
        instruction_set = self.circuit.cy(self.q, self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, CyGate)
        self.assertQasm(qasm_txt)

    def test_cy_reg_bit(self):
        qasm_txt = 'cy q[0],r[1];\ncy q[1],r[1];\ncy q[2],r[1];'
        instruction_set = self.circuit.cy(self.q, self.r[1])
        self.assertStmtsType(instruction_set.instructions, CyGate)
        self.assertQasm(qasm_txt)

    def test_cy_reg_bit_inv(self):
        qasm_txt = 'cy q[0],r[1];\ncy q[1],r[1];\ncy q[2],r[1];'
        instruction_set = self.circuit.cy(self.q, self.r[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, CyGate)
        self.assertQasm(qasm_txt)

    def test_cy_bit_reg(self):
        qasm_txt = 'cy q[1],r[0];\ncy q[1],r[1];\ncy q[1],r[2];'
        instruction_set = self.circuit.cy(self.q[1], self.r)
        self.assertStmtsType(instruction_set.instructions, CyGate)
        self.assertQasm(qasm_txt)

    def test_cy_bit_reg_inv(self):
        qasm_txt = 'cy q[1],r[0];\ncy q[1],r[1];\ncy q[1],r[2];'
        instruction_set = self.circuit.cy(self.q[1], self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, CyGate)
        self.assertQasm(qasm_txt)

    def test_cz_reg_reg(self):
        qasm_txt = 'cz q[0],r[0];\ncz q[1],r[1];\ncz q[2],r[2];'
        instruction_set = self.circuit.cz(self.q, self.r)
        self.assertStmtsType(instruction_set.instructions, CzGate)
        self.assertQasm(qasm_txt)

    def test_cz_reg_reg_inv(self):
        qasm_txt = 'cz q[0],r[0];\ncz q[1],r[1];\ncz q[2],r[2];'
        instruction_set = self.circuit.cz(self.q, self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, CzGate)
        self.assertQasm(qasm_txt)

    def test_cz_reg_bit(self):
        qasm_txt = 'cz q[0],r[1];\ncz q[1],r[1];\ncz q[2],r[1];'
        instruction_set = self.circuit.cz(self.q, self.r[1])
        self.assertStmtsType(instruction_set.instructions, CzGate)
        self.assertQasm(qasm_txt)

    def test_cz_reg_bit_inv(self):
        qasm_txt = 'cz q[0],r[1];\ncz q[1],r[1];\ncz q[2],r[1];'
        instruction_set = self.circuit.cz(self.q, self.r[1]).inverse()
        self.assertStmtsType(instruction_set.instructions, CzGate)
        self.assertQasm(qasm_txt)

    def test_cz_bit_reg(self):
        qasm_txt = 'cz q[1],r[0];\ncz q[1],r[1];\ncz q[1],r[2];'
        instruction_set = self.circuit.cz(self.q[1], self.r)
        self.assertStmtsType(instruction_set.instructions, CzGate)
        self.assertQasm(qasm_txt)

    def test_cz_bit_reg_inv(self):
        qasm_txt = 'cz q[1],r[0];\ncz q[1],r[1];\ncz q[1],r[2];'
        instruction_set = self.circuit.cz(self.q[1], self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, CzGate)
        self.assertQasm(qasm_txt)

    def test_swap_reg_reg(self):
        qasm_txt = 'swap q[0],r[0];\nswap q[1],r[1];\nswap q[2],r[2];'
        instruction_set = self.circuit.swap(self.q, self.r)
        self.assertStmtsType(instruction_set.instructions, SwapGate)
        self.assertQasm(qasm_txt)

    def test_swap_reg_reg_inv(self):
        qasm_txt = 'swap q[0],r[0];\nswap q[1],r[1];\nswap q[2],r[2];'
        instruction_set = self.circuit.swap(self.q, self.r).inverse()
        self.assertStmtsType(instruction_set.instructions, SwapGate)
        self.assertQasm(qasm_txt)


class TestStandard3Q(StandardExtensionTest):
    """Standard Extension Test. Gates with three Qubits"""

    def setUp(self):
        self.q = QuantumRegister(3, "q")
        self.r = QuantumRegister(3, "r")
        self.s = QuantumRegister(3, "s")
        self.c = ClassicalRegister(3, "c")
        self.circuit = QuantumCircuit(self.q, self.r, self.s, self.c)
        self.c_header = 80  # lenght of the header

    def test_barrier_None(self):
        self.circuit.barrier()
        qasm_txt = 'barrier q[0],q[1],q[2],r[0],r[1],r[2],s[0],s[1],s[2];'
        self.assertResult(Barrier, qasm_txt, qasm_txt)

    def test_ccx_reg_reg_reg(self):
        qasm_txt = 'ccx q[0],r[0],s[0];\nccx q[1],r[1],s[1];\nccx q[2],r[2],s[2];'
        instruction_set = self.circuit.ccx(self.q, self.r, self.s)
        self.assertStmtsType(instruction_set.instructions, ToffoliGate)
        self.assertQasm(qasm_txt)

    def test_ccx_reg_reg_inv(self):
        qasm_txt = 'ccx q[0],r[0],s[0];\nccx q[1],r[1],s[1];\nccx q[2],r[2],s[2];'
        instruction_set = self.circuit.ccx(self.q, self.r, self.s).inverse()
        self.assertStmtsType(instruction_set.instructions, ToffoliGate)
        self.assertQasm(qasm_txt)

    def test_cswap_reg_reg_reg(self):
        qasm_txt = 'cswap q[0],r[0],s[0];\n' \
                   'cswap q[1],r[1],s[1];\n' \
                   'cswap q[2],r[2],s[2];'
        instruction_set = self.circuit.cswap(self.q, self.r, self.s)
        self.assertStmtsType(instruction_set.instructions, FredkinGate)
        self.assertQasm(qasm_txt)

    def test_cswap_reg_reg_inv(self):
        qasm_txt = 'cswap q[0],r[0],s[0];\n' \
                   'cswap q[1],r[1],s[1];\n' \
                   'cswap q[2],r[2],s[2];'
        instruction_set = self.circuit.cswap(self.q, self.r, self.s).inverse()
        self.assertStmtsType(instruction_set.instructions, FredkinGate)
        self.assertQasm(qasm_txt)


if __name__ == '__main__':
    unittest.main(verbosity=2)
