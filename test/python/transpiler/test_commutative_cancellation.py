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
from qiskit.circuit.library import U1Gate, RZGate, RXGate
from qiskit.transpiler import PassManager, PropertySet
from qiskit.transpiler.passes import CommutationAnalysis, CommutativeCancellation, FixedPoint, Size
from qiskit.quantum_info import Operator
from qiskit.transpiler.passes.optimization.axis_angle_analysis import _su2_axis_angle as su2
from qiskit.transpiler.passes.optimization.axis_angle_analysis import AxisAngleAnalysis
from qiskit.transpiler.passes.optimization.axis_angle_reduction import AxisAngleReduction

class TestCommutativeCancellation(QiskitTestCase):

    """Test the CommutativeCancellation pass."""

    def setUp(self):
        super().setUp()
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
        circuit.append(U1Gate(0.5), [qr[0]])  # TODO this should work with Phase gates too
        circuit.append(U1Gate(0.5), [qr[0]])
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
        expected.append(RZGate(2.0), [qr[0]])
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
        expected.append(RZGate(np.pi * 17 / 12), [qr[2]])
        expected.cx(qr[2], qr[1])
        expected.global_phase = (np.pi * 17 / 12 - (2 * np.pi / 3)) / 2
        print(Operator(expected) == Operator(circuit))
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
        expected.append(RZGate(np.pi * 17 / 12), [qr[2]])
        expected.append(RZGate(np.pi * 2 / 3), [qr[3]])
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

    def test_basis_01(self):
        """Test basis priority change, phase gate"""
        circuit = QuantumCircuit(1)
        circuit.s(0)
        circuit.z(0)
        circuit.t(0)
        circuit.rz(np.pi, 0)
        passmanager = PassManager()
        passmanager.append(CommutativeCancellation(basis_gates=['cx', 'p', 'sx']))
        new_circuit = passmanager.run(circuit)
        expected = QuantumCircuit(1)
        expected.rz(11 * np.pi / 4, 0)
        expected.global_phase = 11 * np.pi / 4 / 2 - np.pi / 2

        self.assertEqual(new_circuit, expected)

    def test_basis_02(self):
        """Test basis priority change, Rz gate"""
        circuit = QuantumCircuit(1)
        circuit.s(0)
        circuit.z(0)
        circuit.t(0)
        passmanager = PassManager()
        passmanager.append(CommutativeCancellation(basis_gates=['cx', 'rz', 'sx']))
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(1)
        expected.rz(7 * np.pi / 4, 0)
        expected.global_phase = 7 * np.pi / 4 / 2
        self.assertEqual(new_circuit, expected)

    def test_basis_03(self):
        """Test no specified basis"""
        circuit = QuantumCircuit(1)
        circuit.s(0)
        circuit.z(0)
        circuit.t(0)
        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        new_circuit = passmanager.run(circuit)

        expected = QuantumCircuit(1)
        expected.s(0)
        expected.z(0)
        expected.t(0)
        self.assertEqual(new_circuit, expected)

    def test_basis_global_phase_01(self):
        """Test no specified basis, rz"""
        circ = QuantumCircuit(1)
        circ.rz(np.pi/2, 0)
        circ.p(np.pi/2, 0)
        circ.p(np.pi/2, 0)
        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        ccirc = passmanager.run(circ)
        self.assertEqual(Operator(circ), Operator(ccirc))

    def test_basis_global_phase_02(self):
        """Test no specified basis, p"""
        circ = QuantumCircuit(1)
        circ.p(np.pi/2, 0)
        circ.rz(np.pi/2, 0)
        circ.p(np.pi/2, 0)
        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        ccirc = passmanager.run(circ)
        self.assertEqual(Operator(circ), Operator(ccirc))

    def test_basis_global_phase_03(self):
        """Test global phase preservation if cummulative z-rotation is 0"""
        circ = QuantumCircuit(1)
        circ.rz(np.pi/2, 0)
        circ.p(np.pi/2, 0)
        circ.z(0)
        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        ccirc = passmanager.run(circ)
        self.assertEqual(Operator(circ), Operator(ccirc))

    def test_basis_global_phase_04(self):
        """Test global phase preservation if cummulative z-rotation is 0"""
        altp = QuantumCircuit(1, global_phase=np.pi, name='altp')
        altp.rz(np.pi/2, 0)
        altpgate = altp.to_gate()
        circ = QuantumCircuit(1)
        circ.rz(np.pi/2, 0)
        circ.x(0)
        circ.p(np.pi/2, 0)
        circ.append(altpgate, [0])
        circ.rx(np.pi/2, 0)
        circ.x(0)
        
        np.set_printoptions(precision=3, linewidth=250, suppress=True)
        passmanager = PassManager()
        passmanager.append(CommutativeCancellation())
        ccirc = passmanager.run(circ)
        print(circ)
        print(ccirc)
        self.assertEqual(Operator(circ), Operator(ccirc))

    def test_axis_angle(self):
        from qiskit.circuit.library.standard_gates import SGate, TGate, RXGate, XGate, ZGate, RYGate, YGate, IGate

        x = XGate()
        t = TGate()
        u1 = U1Gate(np.pi/3)
        rz = RZGate(np.pi/3)
        rx = RXGate(np.pi/2)
        ry = RYGate(np.pi/3)
        y = YGate()
        z = ZGate()
        iden = IGate()
        np.set_printoptions(precision=3, linewidth=250, suppress=True)
        print('')
        ccirc = passmanager.run(circ)

        axis, angle, phase = su2(np.exp(1j*np.pi/4) * np.array([[1, 0], [0, 1]]))
        print(f'axis={axis}, angle={angle}, phase={phase}')
        breakpoint()

    def test_axis_angle_pass(self):
        altp = QuantumCircuit(1, global_phase=np.pi, name='altp')
        altp.rz(np.pi/2, 0)
        altpgate = altp.to_gate()
        circ = QuantumCircuit(1)
        circ.rz(np.pi/2, 0)
        circ.p(np.pi/2, 0)
        circ.append(altpgate, [0])
        circ.rx(np.pi/2, 0)
        circ.x(0)

        passmanager = PassManager()
        passmanager.append(AxisAngleAnalysis())
        ccirc = passmanager.run(circ)

    def test_axis_angle_pass_2q_noninteracting(self):
        altp = QuantumCircuit(1, global_phase=np.pi, name='altp')
        altp.rz(np.pi/2, 0)
        altpgate = altp.to_gate()
        circ = QuantumCircuit(2)
        # z-axis, qubit 0
        circ.rz(np.pi/2, 0)
        circ.p(np.pi/2, 0)
        circ.append(altpgate, [0])  # test custom gate
        # x-axis, qubit 0
        circ.rx(np.pi/2, 0)
        circ.x(0)
        # zx-axis, qubit 0
        circ.h(0)
        circ.h(0)
        circ.h(0)
        # -zx-axis, qubit 0
        circ.rv(-1, 0, -1, 0)
        # z-axis, qubit 1
        circ.rz(np.pi/2, 1)
        circ.t(1)
        # y-axis, qubit 1
        circ.ry(np.pi/3, 1)
        circ.y(1)
        circ.r(np.pi/3, np.pi/2, 1)
        # cos(pi/3)x + sin(pi/3)y axis, qubit 1, 2 parameters
        circ.r(np.pi/2, np.pi/3, 1)
        circ.r(np.pi/3, np.pi/3, 1)
        circ.cx(0, 1) # check interruption by cx
        circ.r(np.pi/3, np.pi/3, 1)
        passmanager = PassManager()
        passmanager.append(AxisAngleReduction())
        ccirc = passmanager.run(circ)
        # TODO: set equal when fixing global phase
        self.assertTrue(Operator(circ).equiv(ccirc))
        
        passmanager2 = PassManager()
        passmanager2.append(CommutativeCancellation())
        ccirc2 = passmanager2.run(circ)
        breakpoint()


    def test_symmetric_cancellation(self):
        circ = QuantumCircuit(2)
        rot3 = RXGate(2 * np.pi / 3)
        circ.z(0)
        circ.z(0)
        circ.z(0)
        circ.s(0)
        circ.s(0)
        circ.s(0)
        circ.s(0)
        circ.z(0)        
        passmanager = PassManager()
        passmanager.append(AxisAngleReduction())
        ccirc = passmanager.run(circ)
        self.assertEqual(len(ccirc), 0)

        passmanager2 = PassManager()
        passmanager2.append(CommutativeCancellation())
        ccirc2 = passmanager2.run(circ)

        passmanager2 = PassManager()
        passmanager2.append(CommutativeCancellation())
        ccirc2 = passmanager2.run(circ)
        breakpoint()

if __name__ == '__main__':
    unittest.main()
