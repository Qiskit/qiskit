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
from qiskit.extensions.standard.ccx import ToffoliGate
from qiskit.extensions.standard.ch import CHGate
from qiskit.extensions.standard.crz import CrzGate
from qiskit.extensions.standard.cswap import FredkinGate
from qiskit.extensions.standard.barrier import Barrier
from qiskit.extensions.standard.h import HGate
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
        c = self.circuit
        qasm_txt = 'barrier q[1];'
        self.assertRaises(QISKitError, c.barrier, self.c[0])
        #TODO self.assertRaises(QISKitError, c.barrier, 0)
        c.barrier(self.q[1])
        self.assertEqual(type(c[0]), Barrier)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(72, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt+'\n'+qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])  
        self.assertIn(qasm_txt+'\n'+qasm_txt, c.qasm())


    def test_ccx(self):
        c = self.circuit
        qasm_txt = 'ccx q[0],q[1],q[2];'
        self.assertRaises(QISKitError, c.ccx, self.c[0],self.c[1],self.c[2])
        self.assertRaises(QISKitError, c.ccx, self.q[0], self.q[0], self.q[2])
        #TODO self.assertRaises(QISKitError, c.ccx, 0, self.q[0], self.q[2])
        c.ccx(self.q[0],self.q[1],self.q[2])
        self.assertEqual(type(c[0]), ToffoliGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(78, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt+'\n'+qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])  
        self.assertIn(qasm_txt+'\n'+qasm_txt, c.qasm())


    def test_ch(self):
        c = self.circuit
        qasm_txt = 'ch q[0],q[1];'
        self.assertRaises(QISKitError, c.ch, self.c[0],self.c[1])
        self.assertRaises(QISKitError, c.ch, self.q[0], self.q[0])
        #TODO self.assertRaises(QISKitError, c.ch, 0, self.q[0])
        c.ch(self.q[0],self.q[1])
        self.assertEqual(type(c[0]), CHGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(72, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt+'\n'+qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])  
        self.assertIn(qasm_txt+'\n'+qasm_txt, c.qasm())

    def test_crz(self):
        c = self.circuit
        qasm_txt = 'crz(1.000000000000000) q[0],q[1];'
        self.assertRaises(QISKitError, c.crz, 0, self.c[0],self.c[1])
        self.assertRaises(QISKitError, c.crz, 0, self.q[0], self.q[0])
        #TODO self.assertRaises(QISKitError, c.crz, 0, 0, self.q[0])
        c.crz(1,self.q[0],self.q[1])
        self.assertEqual(type(c[0]), CrzGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(92, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt+'\n'+qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])  
        self.assertIn('crz(-1.000000000000000) q[0],q[1];'+'\n'+qasm_txt, c.qasm())

    def test_cswap(self):
        qasm_txt = 'cx q[2],q[1];\nccx q[0],q[1],q[2];\ncx q[2],q[1];'
        c = self.circuit
        self.assertRaises(QISKitError, c.cswap, self.c[0], self.c[1],self.c[2])
        self.assertRaises(QISKitError, c.cswap, self.q[1], self.q[0], self.q[0])
        # TODO self.assertRaises(QISKitError, c.cswap, self.q[1], 0, self.q[0])
        c.cswap(self.q[0],self.q[1],self.q[2])
        self.assertEqual(type(c[0]), FredkinGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(106, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn(qasm_txt+'\n'+qasm_txt, c.qasm())
        self.assertEqual(c[0].inverse(), c[0])
        self.assertIn(qasm_txt+'\n'+qasm_txt,c.qasm())


    def test_h(self):
        c = self.circuit
        qasm_txt = 'h q[1];'
        self.assertRaises(QISKitError, c.h, self.c[0])
        #TODO self.assertRaises(QISKitError, c.h, 0)
        c.h(self.q[1])
        self.assertEqual(type(c[0]), HGate)
        self.assertIn(qasm_txt, c.qasm())
        self.assertEqual(66, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn('h q[1];\nh q[1];', c.qasm())
        self.assertEqual(c[0].inverse(), c[0])  
        self.assertIn('h q[1];\nh q[1];', c.qasm())

    def test_x(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.x, self.c[0])
        # TODO self.assertRaises(QISKitError, c.x, 0)
        c.x(self.q[1])
        self.assertEqual(type(c[0]), XGate)
        self.assertIn('x q[1];', c.qasm())
        self.assertEqual(66, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn('x q[1];\nx q[1];', c.qasm())
        self.assertEqual(c[0].inverse(), c[0])  
        self.assertIn('x q[1];\nx q[1];', c.qasm())

    def test_y(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.y, self.c[0])
        # TODO self.assertRaises(QISKitError, c.y, 0)
        c.y(self.q[1])
        self.assertEqual(type(c[0]), YGate)
        self.assertIn('y q[1];', c.qasm())
        self.assertEqual(66, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn('y q[1];\ny q[1];', c.qasm())
        self.assertEqual(c[0].inverse(), c[0])  
        self.assertIn('y q[1];\ny q[1];', c.qasm())


    def test_z(self):
        c = self.circuit
        self.assertRaises(QISKitError, c.z, self.c[0])
        # TODO self.assertRaises(QISKitError, c.z, 0)
        c.z(self.q[1])
        self.assertEqual(type(c[0]), ZGate)
        self.assertIn('z q[1];', c.qasm())
        self.assertEqual(66, len(c.qasm()))
        c[0].reapply(c)
        self.assertIn('z q[1];\nz q[1];', c.qasm())
        self.assertEqual(c[0].inverse(), c[0])  
        self.assertIn('z q[1];\nz q[1];', c.qasm())

if __name__ == '__main__':
    unittest.main(verbosity=2)
