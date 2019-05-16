# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Gate cancellation pass testing"""

import unittest
import sympy
from qiskit.test import QiskitTestCase

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PassManager, PropertySet
from qiskit.compiler import transpile
from qiskit.transpiler.passes import CommutationAnalysis, CommutativeCancellation


class TestCommutativeCancellation(QiskitTestCase):

    """Test the CommutativeCancellation pass."""

    def setUp(self):

        self.com_pass_ = CommutationAnalysis()
        self.pass_ = CommutativeCancellation()
        self.pset = self.pass_.property_set = PropertySet()

    def test_all_gates(self):
        """Test all gates on 1 and 2 qubits

        q0:-[H]-[H]--[x]-[x]--[y]-[y]--[rz]-[rz]--[u1]-[u1]---------.--.--.--.--.--.-
                                                                    |  |  |  |  |  |
        q1:---------------------------------------------------------X--X--Y--Y--.--.-

        =

        qr0:---[u1]---

        qr1:----------
        """
        qr = QuantumRegister(2, 'q')
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.x(qr[0])
        circuit.x(qr[0])
        circuit.y(qr[0])
        circuit.y(qr[0])
        circuit.rz(0.5, qr[0])
        circuit.rz(0.5, qr[0])
        circuit.u1(0.5, qr[0])
        circuit.u1(0.5, qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cy(qr[0], qr[1])
        circuit.cy(qr[0], qr[1])
        circuit.cz(qr[0], qr[1])
        circuit.cz(qr[0], qr[1])

        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        new_circuit = transpile(circuit, pass_manager=passmanager)

        expected = QuantumCircuit(qr)
        expected.u1(2.0, qr[0])

        self.assertEqual(expected, new_circuit)

    def test_commutative_circuit1(self):
        """A simple circuit where three CNOTs commute, the first and the last cancel.

        qr0:----.---------------.--       qr0:------------
                |               |
        qr1:---(+)-----(+)-----(+)-   =   qr1:-------(+)--
                        |                             |
        qr2:---[H]------.----------       qr2:---[H]--.---
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.h(qr[2])
        circuit.cx(qr[2], qr[1])
        circuit.cx(qr[0], qr[1])

        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        new_circuit = transpile(circuit, pass_manager=passmanager)

        expected = QuantumCircuit(qr)
        expected.h(qr[2])
        expected.cx(qr[2], qr[1])

        self.assertEqual(expected, new_circuit)

    def test_commutative_circuit2(self):
        """
        A simple circuit where three CNOTs commute, the first and the last cancel,
        also two X gates cancel and two Rz gates combine.

        qr0:----.---------------.--------     qr0:-------------
                |               |
        qr1:---(+)---(+)--[X]--(+)--[X]--  =  qr1:--------(+)--
                      |                                    |
        qr2:---[Rz]---.---[Rz]-[T]--[S]--     qr2:--[U1]---.---
        """
        qr = QuantumRegister(3, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.rz(sympy.pi / 3, qr[2])
        circuit.cx(qr[2], qr[1])
        circuit.rz(sympy.pi / 3, qr[2])
        circuit.t(qr[2])
        circuit.s(qr[2])
        circuit.x(qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.x(qr[1])

        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        new_circuit = transpile(circuit, pass_manager=passmanager)
        expected = QuantumCircuit(qr)
        expected.u1(sympy.pi * 17 / 12, qr[2])
        expected.cx(qr[2], qr[1])

        self.assertEqual(expected, new_circuit)


if __name__ == '__main__':
    unittest.main()
