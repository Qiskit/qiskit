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
import numpy as np
from qiskit.test import QiskitTestCase

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PassManager, PropertySet
from qiskit.transpiler.passes import CommutationAnalysis, CommutativeCancellation, FixedPoint, Size


class TestCommutativeCancellation(QiskitTestCase):

    """Test the CommutativeCancellation pass."""

    def setUp(self):

        self.com_pass_ = CommutationAnalysis()
        self.pass_ = CommutativeCancellation()
        self.pset = self.pass_.property_set = PropertySet()

    def test_all_gates(self):
        """Test all gates on 1 and 2 qubits

        q0:-[H]-[H]--[x]-[x]--[y]-[y]--[rz]-[rz]--[u1]-[u1]-[rx]-[rx]---.--.--.--.--.--.-
                                                                        |  |  |  |  |  |
        q1:-------------------------------------------------------------X--X--Y--Y--.--.-

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
        circuit.rx(0.5, qr[0])
        circuit.rx(0.5, qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.cy(qr[0], qr[1])
        circuit.cy(qr[0], qr[1])
        circuit.cz(qr[0], qr[1])
        circuit.cz(qr[0], qr[1])

        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(qr)
        expected.u1(2.0, qr[0])
        expected.rx(1.0, qr[0])

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
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(qr)
        expected.h(qr[2])
        expected.cx(qr[2], qr[1])

        self.assertEqual(expected, new_circuit)

    def test_consecutive_cnots(self):
        """A simple circuit equals identity

        qr0:----.- ----.--       qr0:------------
                |      |
        qr1:---(+)----(+)-   =   qr1:------------
        """

        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)

        self.assertEqual(expected, new_circuit)

    def test_consecutive_cnots2(self):
        """
        Two CNOTs that equals identity, with rotation gates inserted.
        """
        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.rx(np.pi, qr[0])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.rx(np.pi, qr[0])

        passmanager = PassManager()
        passmanager.append([CommutationAnalysis(),
                            CommutativeCancellation(),
                            Size(), FixedPoint('size')],
                           do_while=lambda property_set: not property_set['size_fixed_point'])
        new_circuit = passmanager.run(circuit)
        expected = QuantumCircuit(qr)

        self.assertEqual(expected, new_circuit)

    def test_2_alternating_cnots(self):
        """A simple circuit where nothing should be cancelled.

        qr0:----.- ---(+)-       qr0:----.----(+)-
                |      |                 |     |
        qr1:---(+)-----.--   =   qr1:---(+)----.--

        """

        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[0])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.cx(qr[0], qr[1])
        expected.cx(qr[1], qr[0])

        self.assertEqual(expected, new_circuit)

    def test_control_bit_of_cnot(self):
        """A simple circuit where nothing should be cancelled.

        qr0:----.------[X]------.--       qr0:----.------[X]------.--
                |               |                 |               |
        qr1:---(+)-------------(+)-   =   qr1:---(+)-------------(+)-
        """

        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.x(qr[0])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.cx(qr[0], qr[1])
        expected.x(qr[0])
        expected.cx(qr[0], qr[1])

        self.assertEqual(expected, new_circuit)

    def test_control_bit_of_cnot1(self):
        """A simple circuit where the two cnots shoule be cancelled.

        qr0:----.------[Z]------.--       qr0:---[Z]---
                |               |
        qr1:---(+)-------------(+)-   =   qr1:---------
        """

        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.z(qr[0])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.z(qr[0])

        self.assertEqual(expected, new_circuit)

    def test_control_bit_of_cnot2(self):
        """A simple circuit where the two cnots shoule be cancelled.

        qr0:----.------[T]------.--       qr0:---[T]---
                |               |
        qr1:---(+)-------------(+)-   =   qr1:---------
        """

        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.t(qr[0])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.t(qr[0])

        self.assertEqual(expected, new_circuit)

    def test_control_bit_of_cnot3(self):
        """A simple circuit where the two cnots shoule be cancelled.

        qr0:----.------[Rz]------.--       qr0:---[Rz]---
                |                |
        qr1:---(+)-------- -----(+)-   =   qr1:----------
        """

        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.rz(np.pi / 3, qr[0])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.rz(np.pi / 3, qr[0])

        self.assertEqual(expected, new_circuit)

    def test_control_bit_of_cnot4(self):
        """A simple circuit where the two cnots shoule be cancelled.

        qr0:----.------[T]------.--       qr0:---[T]---
                |               |
        qr1:---(+)-------------(+)-   =   qr1:---------
        """

        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.t(qr[0])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.t(qr[0])

        self.assertEqual(expected, new_circuit)

    def test_target_bit_of_cnot(self):

        """A simple circuit where nothing should be cancelled.

        qr0:----.---------------.--       qr0:----.---------------.--
                |               |                 |               |
        qr1:---(+)-----[Z]-----(+)-   =   qr1:---(+)----[Z]------(+)-
        """

        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.z(qr[1])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.cx(qr[0], qr[1])
        expected.z(qr[1])
        expected.cx(qr[0], qr[1])

        self.assertEqual(expected, new_circuit)

    def test_target_bit_of_cnot1(self):
        """A simple circuit where nothing should be cancelled.

        qr0:----.---------------.--       qr0:----.---------------.--
                |               |                 |               |
        qr1:---(+)-----[T]-----(+)-   =   qr1:---(+)----[T]------(+)-
        """

        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.t(qr[1])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.cx(qr[0], qr[1])
        expected.t(qr[1])
        expected.cx(qr[0], qr[1])

        self.assertEqual(expected, new_circuit)

    def test_target_bit_of_cnot2(self):
        """A simple circuit where nothing should be cancelled.

        qr0:----.---------------.--       qr0:----.---------------.--
                |               |                 |               |
        qr1:---(+)-----[Rz]----(+)-   =   qr1:---(+)----[Rz]-----(+)-
        """

        qr = QuantumRegister(2, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.rz(np.pi / 3, qr[1])
        circuit.cx(qr[0], qr[1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.cx(qr[0], qr[1])
        expected.rz(np.pi / 3, qr[1])
        expected.cx(qr[0], qr[1])

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
        circuit.rz(np.pi / 3, qr[2])
        circuit.cx(qr[2], qr[1])
        circuit.rz(np.pi / 3, qr[2])
        circuit.t(qr[2])
        circuit.s(qr[2])
        circuit.x(qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.x(qr[1])

        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        new_circuit = passmanager.run(circuit)
        expected = QuantumCircuit(qr)
        expected.u1(np.pi * 17 / 12, qr[2])
        expected.cx(qr[2], qr[1])

        self.assertEqual(expected, new_circuit)

    def test_commutative_circuit3(self):
        """
        A simple circuit where three CNOTs commute, the first and the last cancel,
        also two X gates cancel and two Rz gates combine.

        qr0:-------.------------------.-------------     qr0:-------------
                   |                  |
        qr1:------(+)------(+)--[X]--(+)-------[X]--  =  qr1:--------(+)--
                            |                                         |
        qr2:------[Rz]--.---.----.---[Rz]-[T]--[S]--     qr2:--[U1]---.---
                        |        |
        qr3:-[Rz]--[X]-(+)------(+)--[X]-[Rz]-------     qr3:--[Rz]-------
        """

        qr = QuantumRegister(4, 'qr')
        circuit = QuantumCircuit(qr)

        circuit.cx(qr[0], qr[1])
        circuit.rz(np.pi / 3, qr[2])
        circuit.rz(np.pi / 3, qr[3])
        circuit.x(qr[3])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[2], qr[1])
        circuit.cx(qr[2], qr[3])
        circuit.rz(np.pi / 3, qr[2])
        circuit.t(qr[2])
        circuit.x(qr[3])
        circuit.rz(np.pi / 3, qr[3])
        circuit.s(qr[2])
        circuit.x(qr[1])
        circuit.cx(qr[0], qr[1])
        circuit.x(qr[1])

        passmanager = PassManager()
        passmanager.append([CommutationAnalysis(),
                            CommutativeCancellation(), Size(), FixedPoint('size')],
                           do_while=lambda property_set: not property_set['size_fixed_point'])
        new_circuit = passmanager.run(circuit)
        expected = QuantumCircuit(qr)
        expected.u1(np.pi * 17 / 12, qr[2])
        expected.u1(np.pi * 2 / 3, qr[3])
        expected.cx(qr[2], qr[1])

        self.assertEqual(expected, new_circuit)

    def test_cnot_cascade(self):
        """
        A cascade of CNOTs that equals identity.
        """

        qr = QuantumRegister(10, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[4], qr[5])
        circuit.cx(qr[5], qr[6])
        circuit.cx(qr[6], qr[7])
        circuit.cx(qr[7], qr[8])
        circuit.cx(qr[8], qr[9])

        circuit.cx(qr[8], qr[9])
        circuit.cx(qr[7], qr[8])
        circuit.cx(qr[6], qr[7])
        circuit.cx(qr[5], qr[6])
        circuit.cx(qr[4], qr[5])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[0], qr[1])

        passmanager = PassManager()
        # passmanager.append(CommutativeCancellation())
        passmanager.append([CommutationAnalysis(),
                            CommutativeCancellation(), Size(), FixedPoint('size')],
                           do_while=lambda property_set: not property_set['size_fixed_point'])
        new_circuit = passmanager.run(circuit)
        expected = QuantumCircuit(qr)

        self.assertEqual(expected, new_circuit)

    def test_cnot_cascade1(self):
        """
        A cascade of CNOTs that equals identity, with rotation gates inserted.
        """

        qr = QuantumRegister(10, 'qr')
        circuit = QuantumCircuit(qr)
        circuit.rx(np.pi, qr[0])
        circuit.rx(np.pi, qr[1])
        circuit.rx(np.pi, qr[2])
        circuit.rx(np.pi, qr[3])
        circuit.rx(np.pi, qr[4])
        circuit.rx(np.pi, qr[5])
        circuit.rx(np.pi, qr[6])
        circuit.rx(np.pi, qr[7])
        circuit.rx(np.pi, qr[8])
        circuit.rx(np.pi, qr[9])
        circuit.cx(qr[0], qr[1])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[4], qr[5])
        circuit.cx(qr[5], qr[6])
        circuit.cx(qr[6], qr[7])
        circuit.cx(qr[7], qr[8])
        circuit.cx(qr[8], qr[9])
        circuit.cx(qr[8], qr[9])
        circuit.cx(qr[7], qr[8])
        circuit.cx(qr[6], qr[7])
        circuit.cx(qr[5], qr[6])
        circuit.cx(qr[4], qr[5])
        circuit.cx(qr[3], qr[4])
        circuit.cx(qr[2], qr[3])
        circuit.cx(qr[1], qr[2])
        circuit.cx(qr[0], qr[1])
        circuit.rx(np.pi, qr[0])
        circuit.rx(np.pi, qr[1])
        circuit.rx(np.pi, qr[2])
        circuit.rx(np.pi, qr[3])
        circuit.rx(np.pi, qr[4])
        circuit.rx(np.pi, qr[5])
        circuit.rx(np.pi, qr[6])
        circuit.rx(np.pi, qr[7])
        circuit.rx(np.pi, qr[8])
        circuit.rx(np.pi, qr[9])
        passmanager = PassManager()
        # passmanager.append(CommutativeCancellation())
        passmanager.append([CommutationAnalysis(),
                            CommutativeCancellation(), Size(), FixedPoint('size')],
                           do_while=lambda property_set: not property_set['size_fixed_point'])
        new_circuit = passmanager.run(circuit)
        expected = QuantumCircuit(qr)

        self.assertEqual(expected, new_circuit)

    def test_conditional_gates_dont_commute(self):
        """Conditional gates do not commute and do not cancel"""
        circuit = QuantumCircuit(3, 2)
        circuit.h(0)
        circuit.measure(0, 0)
        circuit.cx(1, 2)
        circuit.cx(1, 2).c_if(circuit.cregs[0], 0)
        circuit.measure([1, 2], [0, 1])

        new_pm = PassManager(CommutativeCancellation())
        new_circuit = new_pm.run(circuit)

        self.assertEqual(circuit, new_circuit)


if __name__ == '__main__':
    unittest.main()
