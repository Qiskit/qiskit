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
        self.q = QuantumRegister("q", 3)
        self.c = ClassicalRegister("c", 3)
        self.circuit = QuantumCircuit(self.q, self.c)

    def test_barrier(self):
        qasm_txt = 'barrier q[1];'
        c = self.circuit
        self.assertRaises(QISKitError, c.barrier, self.c[0])
        # TODO self.assertRaises(QISKitError, c.barrier, 0)
        c.barrier(self.q[1])
        self.assertEqual(type(c[0]), Barrier)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(72, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(86, len(c.qasm()))

    def test_ccx(self):
        qasm_txt = 'ccx q[0],q[1],q[2];'
        c = self.circuit
        self.assertRaises(QISKitError, c.ccx, self.c[0], self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.ccx, self.q[0], self.q[0], self.q[2])
        # TODO self.assertRaises(QISKitError, c.ccx, 0, self.q[0], self.q[2])
        c.ccx(self.q[0], self.q[1], self.q[2])
        self.assertEqual(type(c[0]), ToffoliGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(78, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(98, len(c.qasm()))

    def test_ch(self):
        qasm_txt = 'ch q[0],q[1];'
        c = self.circuit
        self.assertRaises(QISKitError, c.ch, self.c[0], self.c[1])
        self.assertRaises(QISKitError, c.ch, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.ch, 0, self.q[0])
        c.ch(self.q[0], self.q[1])
        self.assertEqual(type(c[0]), CHGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(72, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(86, len(c.qasm()))

    def test_crz(self):
        qasm_txt = 'crz(1.000000000000000) q[0],q[1];'
        c = self.circuit
        self.assertRaises(QISKitError, c.crz, 0, self.c[0], self.c[1])
        self.assertRaises(QISKitError, c.crz, 0, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.crz, 0, 0, self.q[0])
        c.crz(1, self.q[0], self.q[1])
        self.assertEqual(type(c[0]), CrzGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(92, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn('crz(-1.000000000000000) q[0],q[1];' + '\n' + qasm_txt, c.qasm())
        self.assertEqual(127, len(c.qasm()))

    def test_cswap(self):
        qasm_txt = 'cx q[2],q[1];\nccx q[0],q[1],q[2];\ncx q[2],q[1];'
        c = self.circuit
        self.assertRaises(QISKitError, c.cswap, self.c[0], self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cswap, self.q[1], self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cswap, self.q[1], 0, self.q[0])
        c.cswap(self.q[0], self.q[1], self.q[2])
        self.assertEqual(type(c[0]), FredkinGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(106, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(154, len(c.qasm()))

    def test_cu1(self):
        qasm_txt = 'cu1(1.000000000000000) q[1],q[2];'
        c = self.circuit
        self.assertRaises(QISKitError, c.cu1, self.c[0], self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cu1, 1, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cu1, self.q[1], 0, self.q[0])
        c.cu1(1, self.q[1], self.q[2])
        self.assertEqual(type(c[0]), Cu1Gate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(92, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn('cu1(-1.000000000000000) q[1],q[2];' + '\n' + qasm_txt, c.qasm())
        self.assertEqual(127, len(c.qasm()))

    def test_cu3(self):
        qasm_txt = 'cu3(1.000000000000000,2.000000000000000,3.000000000000000) q[1],q[2];'
        c = self.circuit
        self.assertRaises(QISKitError, c.cu3, 0, 0, self.c[0], self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cu3, 0, 0, 1, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cu3, 0, 0, 0, self.q[1], 0, self.q[0])
        c.cu3(1, 2, 3, self.q[1], self.q[2])
        self.assertEqual(type(c[0]), Cu3Gate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(128, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn('cu3(-1.000000000000000,-3.000000000000000,-2.000000000000000) q[1],q[2];' + '\n' + qasm_txt,
                      c.qasm())
        self.assertEqual(201, len(c.qasm()))

    def test_cx(self):
        qasm_txt = 'cx q[1],q[2];'
        c = self.circuit
        self.assertRaises(QISKitError, c.cx, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cx, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cx, 0, self.q[0])
        c.cx(self.q[1], self.q[2])
        self.assertEqual(type(c[0]), CnotGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(72, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(86, len(c.qasm()))

    def test_cxbase(self):
        qasm_txt = 'CX q[1],q[2];'
        c = self.circuit
        self.assertRaises(QISKitError, c.cx_base, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cx_base, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cx_base, 0, self.q[0])
        c.cx_base(self.q[1], self.q[2])
        self.assertEqual(type(c[0]), CXBase)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(72, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(86, len(c.qasm()))

    def test_cy(self):
        qasm_txt = 'cy q[1],q[2];'
        c = self.circuit
        self.assertRaises(QISKitError, c.cy, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cy, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cy, 0, self.q[0])
        c.cy(self.q[1], self.q[2])
        self.assertEqual(type(c[0]), CyGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(72, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(86, len(c.qasm()))

    def test_cz(self):
        qasm_txt = 'cz q[1],q[2];'
        c = self.circuit
        self.assertRaises(QISKitError, c.cz, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.cz, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cy, 0, self.q[0])
        c.cz(self.q[1], self.q[2])
        self.assertEqual(type(c[0]), CzGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(72, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(86, len(c.qasm()))

    def test_h(self):
        c = self.circuit
        qasm_txt = 'h q[1];'
        self.assertRaises(QISKitError, c.h, self.c[0])
        # TODO self.assertRaises(QISKitError, c.h, 0)
        c.h(self.q[1])
        self.assertEqual(type(c[0]), HGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(66, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(74, len(c.qasm()))

    def test_iden(self):
        c = self.circuit
        qasm_txt = 'id q[1];'
        self.assertRaises(QISKitError, c.iden, self.c[0])
        # TODO self.assertRaises(QISKitError, c.iden, 0)
        c.iden(self.q[1])
        self.assertEqual(type(c[0]), IdGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(67, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(76, len(c.qasm()))

    def test_rx(self):
        c = self.circuit
        qasm_txt = 'rx(1.000000000000000) q[1];'
        self.assertRaises(QISKitError, c.rx, 1, self.c[0])
        # TODO self.assertRaises(QISKitError, c.rx, 1, 1)
        c.rx(1, self.q[1])
        self.assertEqual(type(c[0]), RXGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(86, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn('rx(-1.000000000000000) q[1];' + '\n' + qasm_txt, c.qasm())
        self.assertEqual(115, len(c.qasm()))

    def test_ry(self):
        c = self.circuit
        qasm_txt = 'ry(1.000000000000000) q[1];'
        self.assertRaises(QISKitError, c.ry, 1, self.c[0])
        # TODO self.assertRaises(QISKitError, c.ry, 1, 1)
        c.ry(1, self.q[1])
        self.assertEqual(type(c[0]), RYGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(86, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn('ry(-1.000000000000000) q[1];' + '\n' + qasm_txt, c.qasm())
        self.assertEqual(115, len(c.qasm()))

    def test_rz(self):
        c = self.circuit
        qasm_txt = 'rz(1.000000000000000) q[1];'
        self.assertRaises(QISKitError, c.rz, 1, self.c[0])
        # TODO self.assertRaises(QISKitError, c.rz, 1, 1)
        c.rz(1, self.q[1])
        self.assertEqual(type(c[0]), RZGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(86, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn('rz(-1.000000000000000) q[1];' + '\n' + qasm_txt, c.qasm())
        self.assertEqual(115, len(c.qasm()))

    def test_s(self):
        c = self.circuit
        qasm_txt = 's q[1];'
        self.assertRaises(QISKitError, c.s, self.c[0])
        # TODO self.assertRaises(QISKitError, c.s, 1)
        c.s(self.q[1])
        self.assertEqual(type(c[0]), SGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(66, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn('sdg q[1];' + '\n' + qasm_txt, c.qasm())
        self.assertEqual(76, len(c.qasm()))

    def test_sdg(self):
        c = self.circuit
        qasm_txt = 'sdg q[1];'
        self.assertRaises(QISKitError, c.sdg, self.c[0])
        # TODO self.assertRaises(QISKitError, c.sdg, 1)
        c.sdg(self.q[1])
        self.assertEqual(type(c[0]), SGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(68, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn('s q[1];' + '\n' + qasm_txt, c.qasm())
        self.assertEqual(76, len(c.qasm()))

    def test_swap(self):
        qasm_txt = 'swap q[1],q[2];'
        c = self.circuit
        self.assertRaises(QISKitError, c.swap, self.c[1], self.c[2])
        self.assertRaises(QISKitError, c.swap, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.swap, 0, self.q[0])
        c.swap(self.q[1], self.q[2])
        self.assertEqual(type(c[0]), SwapGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(74, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(90, len(c.qasm()))

    def test_t(self):
        c = self.circuit
        qasm_txt = 't q[1];'
        self.assertRaises(QISKitError, c.t, self.c[0])
        # TODO self.assertRaises(QISKitError, c.t, 1)
        c.t(self.q[1])
        self.assertEqual(type(c[0]), TGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(66, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn('tdg q[1];' + '\n' + qasm_txt, c.qasm())
        self.assertEqual(76, len(c.qasm()))

    def test_u1(self):
        qasm_txt = 'u1(1.000000000000000) q[1];'
        c = self.circuit
        self.assertRaises(QISKitError, c.u1, self.c[0], self.c[0])
        # TODO self.assertRaises(QISKitError, c.u1, self.q[0], 1)
        # CHECKME? self.assertRaises(QISKitError, c.u1, self.c[0], self.q[0])
        c.u1(1, self.q[1])
        self.assertEqual(type(c[0]), U1Gate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(86, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn('u1(-1.000000000000000) q[1];' + '\n' + qasm_txt, c.qasm())
        self.assertEqual(115, len(c.qasm()))

    def test_u2(self):
        qasm_txt = 'u2(1.000000000000000,2.000000000000000) q[1];'
        c = self.circuit
        self.assertRaises(QISKitError, c.u2, 0, self.c[0], self.c[1])
        # TODO self.assertRaises(QISKitError, c.u3, 0, self.q[1],, 0,)
        c.u2(1, 2, self.q[1])
        self.assertEqual(type(c[0]), U2Gate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(104, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn('u2(-5.141592653589793,2.141592653589793) q[1];' + '\n' + qasm_txt, c.qasm())  # CHECKME!
        self.assertEqual(151, len(c.qasm()))

    def test_u3(self):
        qasm_txt = 'u3(1.000000000000000,2.000000000000000,3.000000000000000) q[1];'
        c = self.circuit
        # CHECKME? self.assertRaises(QISKitError, c.u3, 0, 0, self.c[0], self.q[1])
        self.assertRaises(QISKitError, c.u3, 0, 0, 1, self.c[0])
        # TODO self.assertRaises(QISKitError, c.cu3, 0, 0, 0, self.q[1], 0, self.q[0])
        c.u3(1, 2, 3, self.q[1])
        self.assertEqual(type(c[0]), U3Gate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(122, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn('u3(-1.000000000000000,-3.000000000000000,-2.000000000000000) q[1];' + '\n' + qasm_txt, c.qasm())
        self.assertEqual(189, len(c.qasm()))

    def test_ubase(self):
        qasm_txt = 'U(1.000000000000000,2.000000000000000,3.000000000000000) q[1];'
        c = self.circuit
        # self.assertRaises(QISKitError, c.ubase, self.c[1], self.c[2])
        # self.assertRaises(QISKitError, c.ubase, self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cx_base, 0, self.q[0])
        c.u_base([1, 2, 3], self.q[1])  # <- is that use intented?! CHECKME
        self.assertEqual(type(c[0]), UBase)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(121, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(
            'U(-1.000000000000000,-3.000000000000000,-2.000000000000000) q[1];\nU(-1.000000000000000,-3.000000000000000,-2.000000000000000) q[1];',
            c.qasm())  # CHECKME!
        self.assertEqual(190, len(c.qasm()))

    def test_tdg(self):
        c = self.circuit
        qasm_txt = 'tdg q[1];'
        self.assertRaises(QISKitError, c.tdg, self.c[0])
        # TODO self.assertRaises(QISKitError, c.tdg, 1)
        c.tdg(self.q[1])
        self.assertEqual(type(c[0]), TGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(68, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn('t q[1];' + '\n' + qasm_txt, c.qasm())
        self.assertEqual(76, len(c.qasm()))

    def test_x(self):
        c = self.circuit
        qasm_txt = 'x q[1];'
        self.assertRaises(QISKitError, c.x, self.c[0])
        # TODO self.assertRaises(QISKitError, c.x, 0)
        c.x(self.q[1])
        self.assertEqual(type(c[0]), XGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(66, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(74, len(c.qasm()))

    def test_y(self):
        c = self.circuit
        qasm_txt = 'y q[1];'
        self.assertRaises(QISKitError, c.y, self.c[0])
        # TODO self.assertRaises(QISKitError, c.y, 0)
        c.y(self.q[1])
        self.assertEqual(type(c[0]), YGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(66, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(74, len(c.qasm()))

    def test_z(self):
        c = self.circuit
        qasm_txt = 'z q[1];'
        self.assertRaises(QISKitError, c.z, self.c[0])
        # TODO self.assertRaises(QISKitError, c.z, 0)
        c.z(self.q[1])
        self.assertEqual(type(c[0]), ZGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(66, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt + '\n' + qasm_txt, c.qasm())
        self.assertEqual(74, len(c.qasm()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
