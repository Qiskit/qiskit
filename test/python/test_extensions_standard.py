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

import qiskit
# from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
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

from .common import QiskitTestCase


class TestStandard(QiskitTestCase):
    """Standard Extension Test."""

    def setUp(self):
        self.q = qiskit.QuantumRegister(3, "q")
        self.c = qiskit.ClassicalRegister(3, "c")
        self.circuit = qiskit.QuantumCircuit(self.q, self.c)

    def test_barrier(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.barrier, self.c[0])
        # TODO self.assertRaises(QISKitError, c.barrier, 0)
        c.barrier(self.q[1])
        qasm_txt = 'barrier q[1];'
        self.assertResult(Barrier, qasm_txt, qasm_txt)

    def test_ccx(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.ccx, self.c[0], self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.ccx, self.q[0], self.q[0], self.q[2])
        # TODO self.assertRaises(QISKitError, c.ccx, 0, self.q[0], self.q[2])
        c.ccx(self.q[0], self.q[1], self.q[2])
        qasm_txt = 'ccx q[0],q[1],q[2];'
        self.assertResult(ToffoliGate, qasm_txt, qasm_txt)

    def test_ch(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.ch, self.c[0], self.c[1])
        self.assertRaises(QISKitError, c.ch, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.ch, 0, self.q[0])
        c.ch(self.q[0], self.q[1])
        qasm_txt = 'ch q[0],q[1];'
        self.assertResult(CHGate, qasm_txt, qasm_txt)

    def test_crz(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.crz, 0, self.c[0], self.c[1])
        self.assertRaises(QISKitError, c.crz, 0, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.crz, 0, 0, self.q[0])
        c.crz(1, self.q[0], self.q[1])
        self.assertResult(CrzGate, 'crz(1) q[0],q[1];', 'crz(-1) q[0],q[1];')

    def test_cswap(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.cswap, self.c[0], self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cswap, self.q[1], self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cswap, self.q[1], 0, self.q[0])
        c.cswap(self.q[0], self.q[1], self.q[2])
        qasm_txt = 'cx q[2],q[1];\nccx q[0],q[1],q[2];\ncx q[2],q[1];'
        self.assertResult(FredkinGate, qasm_txt, qasm_txt)

    def test_cu1(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.cu1, self.c[0], self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cu1, 1, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cu1, self.q[1], 0, self.q[0])
        c.cu1(1, self.q[1], self.q[2])
        self.assertResult(Cu1Gate, 'cu1(1) q[1],q[2];', 'cu1(-1) q[1],q[2];')

    def test_cu3(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.cu3, 0, 0, self.c[0], self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cu3, 0, 0, 1, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cu3, 0, 0, 0, self.q[1], 0, self.q[0])
        c.cu3(1, 2, 3, self.q[1], self.q[2])
        self.assertResult(Cu3Gate, 'cu3(1,2,3) q[1],q[2];', 'cu3(-1,-3,-2) q[1],q[2];')

    def test_cx(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.cx, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cx, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cx, 0, self.q[0])
        c.cx(self.q[1], self.q[2])
        qasm_txt = 'cx q[1],q[2];'
        self.assertResult(CnotGate, qasm_txt, qasm_txt)

    def test_cxbase(self):
        qasm_txt = 'CX q[1],q[2];'
        c = self.circuit
        self.assertRaises(QISKitError, c.cx_base, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cx_base, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cx_base, 0, self.q[0])
        c.cx_base(self.q[1], self.q[2])
        qasm_txt = 'CX q[1],q[2];'
        self.assertResult(CXBase, qasm_txt, qasm_txt)

    def test_cy(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.cy, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cy, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cy, 0, self.q[0])
        c.cy(self.q[1], self.q[2])
        qasm_txt = 'cy q[1],q[2];'
        self.assertResult(CyGate, qasm_txt, qasm_txt)

    def test_cz(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.cz, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cz, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cy, 0, self.q[0])
        c.cz(self.q[1], self.q[2])
        qasm_txt = 'cz q[1],q[2];'
        self.assertResult(CzGate, qasm_txt, qasm_txt)

    def test_h(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.h, self.c[0])
        # TODO self.assertRaises(QISKitError, c.h, 0)
        c.h(self.q[1])
        qasm_txt = 'h q[1];'
        self.assertResult(HGate, qasm_txt, qasm_txt)

    def test_iden(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.iden, self.c[0])
        # TODO self.assertRaises(QISKitError, c.iden, 0)
        c.iden(self.q[1])
        self.assertResult(IdGate, 'id q[1];', 'id q[1];')

    def test_rx(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.rx, 1, self.c[0])
        # TODO self.assertRaises(QISKitError, c.rx, 1, 1)
        c.rx(1, self.q[1])
        self.assertResult(RXGate, 'rx(1) q[1];', 'rx(-1) q[1];')

    def test_rx_pi(self):
        c = self.circuit
        c.rx(pi/2, self.q[1])
        self.assertResult(RXGate, 'rx(pi/2) q[1];', 'rx(-pi/2) q[1];')

    def test_ry(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.ry, 1, self.c[0])
        # TODO self.assertRaises(QISKitError, c.ry, 1, 1)
        c.ry(1, self.q[1])
        self.assertResult(RYGate, 'ry(1) q[1];', 'ry(-1) q[1];')

    def test_ry_pi(self):
        c = self.circuit
        c.ry(pi/2, self.q[1])
        self.assertResult(RYGate, 'ry(pi/2) q[1];', 'ry(-pi/2) q[1];')

    def test_rz(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.rz, 1, self.c[0])
        # TODO self.assertRaises(QISKitError, c.rz, 1, 1)
        c.rz(1, self.q[1])
        self.assertResult(RZGate, 'rz(1) q[1];', 'rz(-1) q[1];')

    def test_rz_pi(self):
        c = self.circuit
        c.rz(pi/2, self.q[1])
        self.assertResult(RZGate, 'rz(pi/2) q[1];', 'rz(-pi/2) q[1];')

    def test_s(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.s, self.c[0])
        # TODO self.assertRaises(QISKitError, c.s, 1)
        c.s(self.q[1])
        self.assertResult(SGate, 's q[1];', 'sdg q[1];')

    def test_sdg(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.sdg, self.c[0])
        # TODO self.assertRaises(QISKitError, c.sdg, 1)
        c.sdg(self.q[1])
        self.assertResult(SGate, 'sdg q[1];', 's q[1];')

    def test_swap(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.swap, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.swap, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.swap, 0, self.q[0])
        c.swap(self.q[1], self.q[2])
        self.assertResult(SwapGate, 'swap q[1],q[2];', 'swap q[1],q[2];')

    def test_t(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.t, self.c[0])
        # TODO self.assertRaises(QISKitError, c.t, 1)
        c.t(self.q[1])
        self.assertResult(TGate, 't q[1];', 'tdg q[1];')

    def test_tdg(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.tdg, self.c[0])
        # TODO self.assertRaises(QISKitError, c.tdg, 1)
        c.tdg(self.q[1])
        self.assertResult(TGate, 'tdg q[1];', 't q[1];')

    def test_u1(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.u1, self.c[0], self.c[0])
        # TODO self.assertRaises(QISKitError, c.u1, self.q[0], 1)
        # CHECKME? self.assertRaises(QISKitError, c.u1, self.c[0], self.q[0])
        c.u1(1, self.q[1])
        self.assertResult(U1Gate, 'u1(1) q[1];', 'u1(-1) q[1];')

    def test_u1_pi(self):
        c = self.circuit
        c.u1(pi/2, self.q[1])
        self.assertResult(U1Gate, 'u1(pi/2) q[1];', 'u1(-pi/2) q[1];')

    def test_u2(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.u2, 0, self.c[0], self.c[1])
        # TODO self.assertRaises(QISKitError, c.u3, 0, self.q[1],, 0,)
        c.u2(1, 2, self.q[1])
        self.assertResult(U2Gate, 'u2(1,2) q[1];', 'u2(-pi - 2,-1 + pi) q[1];')

    def test_u2_pi(self):
        c = self.circuit
        c.u2(pi/2, 0.3*pi, self.q[1])
        self.assertResult(U2Gate, 'u2(pi/2,0.3*pi) q[1];', 'u2(-1.3*pi,pi/2) q[1];')

    def test_u3(self):
        c = self.circuit
        # CHECKME? self.assertRaises(QISKitError, c.u3, 0, 0, self.c[0], self.q[1])
        self.assertRaises(QISKitError, c.u3, 0, 0, 1, self.c[0])
        # TODO self.assertRaises(QISKitError, c.cu3, 0, 0, 0, self.q[1], 0, self.q[0])
        c.u3(1, 2, 3, self.q[1])
        self.assertResult(U3Gate, 'u3(1,2,3) q[1];', 'u3(-1,-3,-2) q[1];')

    def test_u3_pi(self):
        c = self.circuit
        c.u3(pi, pi/2, 0.3*pi, self.q[1])
        self.assertResult(U3Gate, 'u3(pi,pi/2,0.3*pi) q[1];', 'u3(-pi,-0.3*pi,-pi/2) q[1];')

    def test_ubase(self):
        c = self.circuit
        # self.assertRaises(QISKitError, c.ubase, self.c[1], self.c[2])
        # self.assertRaises(QISKitError, c.ubase, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cx_base, 0, self.q[0])
        c.u_base(1, 2, 3, self.q[1])
        self.assertResult(UBase, 'U(1,2,3) q[1];', 'U(-1,-3,-2) q[1];')

    def test_ubase_pi(self):
        c = self.circuit
        c.u_base(pi, pi / 2, 0.3 * pi, self.q[1])
        self.assertResult(UBase, 'U(pi,pi/2,0.3*pi) q[1];', 'U(-pi,-0.3*pi,-pi/2) q[1];')

    def test_x(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.x, self.c[0])
        # TODO self.assertRaises(QISKitError, c.x, 0)
        c.x(self.q[1])
        self.assertResult(XGate, 'x q[1];', 'x q[1];')

    def test_y(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.y, self.c[0])
        # TODO self.assertRaises(QISKitError, c.y, 0)
        c.y(self.q[1])
        self.assertResult(YGate, 'y q[1];', 'y q[1];')

    def test_z(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.z, self.c[0])
        # TODO self.assertRaises(QISKitError, c.z, 0)
        c.z(self.q[1])
        self.assertResult(ZGate, 'z q[1];', 'z q[1];')

    def assertResult(self, t, qasm_txt, qasm_txt_):
        """
        t: type
        qasm_txt: qasm representation
        qasm_txt_: qasm representation of inverse
        """
        c = self.circuit
        c_header = 58
        c_txt = len(qasm_txt)
        c_txt_ = len(qasm_txt_)
        self.assertEqual(type(c[0]), t)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(c_header + c_txt + 1, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt_ + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c_header + c_txt + c_txt_ + 2, len(c.qasm()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
