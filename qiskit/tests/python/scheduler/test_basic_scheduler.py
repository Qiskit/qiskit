# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test cases for the pulse scheduler passes."""

from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, schedule
from qiskit.circuit import Gate
from qiskit.exceptions import QiskitError
from qiskit.pulse import (Schedule, DriveChannel, AcquireChannel, Acquire,
                          MeasureChannel, MemorySlot, Gaussian, Play)

from qiskit.test.mock import FakeBackend, FakeOpenPulse2Q, FakeOpenPulse3Q
from qiskit.test import QiskitTestCase


class TestBasicSchedule(QiskitTestCase):
    """Scheduling tests."""

    def setUp(self):
        self.backend = FakeOpenPulse2Q()
        self.inst_map = self.backend.defaults().instruction_schedule_map

    def test_unavailable_defaults(self):
        """Test backend with unavailable defaults."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        backend = FakeBackend(None)
        backend.defaults = backend.configuration
        self.assertRaises(QiskitError, lambda: schedule(qc, backend))

    def test_alap_pass(self):
        """Test ALAP scheduling."""
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q, c)
        qc.u2(3.14, 1.57, q[0])
        qc.u2(0.5, 0.25, q[1])
        qc.barrier(q[1])
        qc.u2(0.5, 0.25, q[1])
        qc.barrier(q[0], q[1])
        qc.cx(q[0], q[1])
        qc.measure(q, c)
        sched = schedule(qc, self.backend)
        # X pulse on q0 should end at the start of the CNOT
        expected = Schedule(
            (28, self.inst_map.get('u2', [0], 3.14, 1.57)),
            self.inst_map.get('u2', [1], 0.5, 0.25),
            (28, self.inst_map.get('u2', [1], 0.5, 0.25)),
            (56, self.inst_map.get('cx', [0, 1])),
            (78, self.inst_map.get('measure', [0, 1])))
        for actual, expected in zip(sched.instructions, expected.instructions):
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])

    def test_alap_with_barriers(self):
        """Test that ALAP respects barriers on new qubits."""
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q, c)
        qc.u2(0, 0, q[0])
        qc.barrier(q[0], q[1])
        qc.u2(0, 0, q[1])
        sched = schedule(qc, self.backend, method='alap')
        expected = Schedule(
            self.inst_map.get('u2', [0], 0, 0),
            (28, self.inst_map.get('u2', [1], 0, 0)))
        for actual, expected in zip(sched.instructions, expected.instructions):
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])

    def test_empty_circuit_schedule(self):
        """Test empty circuit being scheduled."""
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q, c)
        sched = schedule(qc, self.backend, method='alap')
        expected = Schedule()
        self.assertEqual(sched.instructions, expected.instructions)

    def test_alap_aligns_end(self):
        """Test that ALAP always acts as though there is a final global barrier."""
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q, c)
        qc.u3(0, 0, 0, q[0])
        qc.u2(0, 0, q[1])
        sched = schedule(qc, self.backend, method='alap')
        expected_sched = Schedule(
            self.inst_map.get('u2', [1], 0, 0),
            (26, self.inst_map.get('u3', [0], 0, 0, 0)))
        for actual, expected in zip(sched.instructions, expected_sched.instructions):
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])
        self.assertEqual(sched.ch_duration(DriveChannel(0)),
                         expected_sched.ch_duration(DriveChannel(1)))

    def test_asap_pass(self):
        """Test ASAP scheduling."""
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q, c)
        qc.u2(3.14, 1.57, q[0])
        qc.u2(0.5, 0.25, q[1])
        qc.barrier(q[1])
        qc.u2(0.5, 0.25, q[1])
        qc.barrier(q[0], q[1])
        qc.cx(q[0], q[1])
        qc.measure(q, c)
        sched = schedule(qc, self.backend, method="as_soon_as_possible")
        # X pulse on q0 should start at t=0
        expected = Schedule(
            self.inst_map.get('u2', [0], 3.14, 1.57),
            self.inst_map.get('u2', [1], 0.5, 0.25),
            (28, self.inst_map.get('u2', [1], 0.5, 0.25)),
            (56, self.inst_map.get('cx', [0, 1])),
            (78, self.inst_map.get('measure', [0, 1])))
        for actual, expected in zip(sched.instructions, expected.instructions):
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])

    def test_alap_resource_respecting(self):
        """Test that the ALAP pass properly respects busy resources when backwards scheduling.
        For instance, a CX on 0 and 1 followed by an X on only 1 must respect both qubits'
        timeline."""
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q, c)
        qc.cx(q[0], q[1])
        qc.u2(0.5, 0.25, q[1])
        sched = schedule(qc, self.backend, method="as_late_as_possible")
        insts = sched.instructions
        self.assertEqual(insts[0][0], 0)
        self.assertEqual(insts[4][0], 22)

        qc = QuantumCircuit(q, c)
        qc.cx(q[0], q[1])
        qc.u2(0.5, 0.25, q[1])
        qc.measure(q, c)
        sched = schedule(qc, self.backend, method="as_late_as_possible")
        self.assertEqual(sched.instructions[-1][0], 50)

    def test_inst_map_schedules_unaltered(self):
        """Test that forward scheduling doesn't change relative timing with a command."""
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q, c)
        qc.cx(q[0], q[1])
        sched1 = schedule(qc, self.backend, method="as_soon_as_possible")
        sched2 = schedule(qc, self.backend, method="as_late_as_possible")
        for asap, alap in zip(sched1.instructions, sched2.instructions):
            self.assertEqual(asap[0], alap[0])
            self.assertEqual(asap[1], alap[1])
        insts = sched1.instructions
        self.assertEqual(insts[0][0], 0)
        self.assertEqual(insts[1][0], 10)
        self.assertEqual(insts[2][0], 20)
        self.assertEqual(insts[3][0], 20)

    def test_measure_combined(self):
        """
        Test to check for measure on the same qubit which generated another measure schedule.

        The measures on different qubits are combined, but measures on the same qubit
        adds another measure to the schedule.
        """
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q, c)
        qc.u2(3.14, 1.57, q[0])
        qc.cx(q[0], q[1])
        qc.measure(q[0], c[0])
        qc.measure(q[1], c[1])
        qc.measure(q[1], c[1])
        sched = schedule(qc, self.backend, method="as_soon_as_possible")
        expected = Schedule(
            self.inst_map.get('u2', [0], 3.14, 1.57),
            (28, self.inst_map.get('cx', [0, 1])),
            (50, self.inst_map.get('measure', [0, 1])),
            (60, self.inst_map.get('measure', [0, 1]).filter(channels=[MeasureChannel(1)])),
            (60, Acquire(10, AcquireChannel(0), MemorySlot(0))),
            (60, Acquire(10, AcquireChannel(1), MemorySlot(1))))
        self.assertEqual(sched.instructions, expected.instructions)

    def test_3q_schedule(self):
        """Test a schedule that was recommended by David McKay :D """
        backend = FakeOpenPulse3Q()
        inst_map = backend.defaults().instruction_schedule_map
        q = QuantumRegister(3)
        c = ClassicalRegister(3)
        qc = QuantumCircuit(q, c)
        qc.cx(q[0], q[1])
        qc.u2(0.778, 0.122, q[2])
        qc.u3(3.14, 1.57, 0., q[0])
        qc.u2(3.14, 1.57, q[1])
        qc.cx(q[1], q[2])
        qc.u2(0.778, 0.122, q[2])
        sched = schedule(qc, backend)
        expected = Schedule(
            inst_map.get('cx', [0, 1]),
            (22, inst_map.get('u2', [1], 3.14, 1.57)),
            (46, inst_map.get('u2', [2], 0.778, 0.122)),
            (50, inst_map.get('cx', [1, 2])),
            (72, inst_map.get('u2', [2], 0.778, 0.122)),
            (74, inst_map.get('u3', [0], 3.14, 1.57)))
        for actual, expected in zip(sched.instructions, expected.instructions):
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])

    def test_schedule_multi(self):
        """Test scheduling multiple circuits at once."""
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc0 = QuantumCircuit(q, c)
        qc0.cx(q[0], q[1])
        qc1 = QuantumCircuit(q, c)
        qc1.cx(q[0], q[1])
        schedules = schedule([qc0, qc1], self.backend)
        expected_insts = schedule(qc0, self.backend).instructions
        for actual, expected in zip(schedules[0].instructions, expected_insts):
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])

    def test_circuit_name_kept(self):
        """Test that the new schedule gets its name from the circuit."""
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        qc = QuantumCircuit(q, c, name='CIRCNAME')
        qc.cx(q[0], q[1])
        sched = schedule(qc, self.backend, method="asap")
        self.assertEqual(sched.name, qc.name)
        sched = schedule(qc, self.backend, method="alap")
        self.assertEqual(sched.name, qc.name)

    def test_can_add_gates_into_free_space(self):
        """The scheduler does some time bookkeeping to know when qubits are free to be
        scheduled. Make sure this works for qubits that are used in the future. This was
        a bug, uncovered by this example:

           q0 =  - - - - |X|
           q1 = |X| |u2| |X|

        In ALAP scheduling, the next operation on qubit 0 would be added at t=0 rather
        than immediately before the X gate.
        """
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        for i in range(2):
            qc.u2(0, 0, [qr[i]])
            qc.u1(3.14, [qr[i]])
            qc.u2(0, 0, [qr[i]])
        sched = schedule(qc, self.backend, method="alap")
        expected = Schedule(
            self.inst_map.get('u2', [0], 0, 0),
            self.inst_map.get('u2', [1], 0, 0),
            (28, self.inst_map.get('u1', [0], 3.14)),
            (28, self.inst_map.get('u1', [1], 3.14)),
            (28, self.inst_map.get('u2', [0], 0, 0)),
            (28, self.inst_map.get('u2', [1], 0, 0)))
        for actual, expected in zip(sched.instructions, expected.instructions):
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])

    def test_barriers_in_middle(self):
        """As a follow on to `test_can_add_gates_into_free_space`, similar issues
        arose for barriers, specifically.
        """
        qr = QuantumRegister(2)
        qc = QuantumCircuit(qr)
        for i in range(2):
            qc.u2(0, 0, [qr[i]])
            qc.barrier(qr[i])
            qc.u1(3.14, [qr[i]])
            qc.barrier(qr[i])
            qc.u2(0, 0, [qr[i]])
        sched = schedule(qc, self.backend, method="alap")
        expected = Schedule(
            self.inst_map.get('u2', [0], 0, 0),
            self.inst_map.get('u2', [1], 0, 0),
            (28, self.inst_map.get('u1', [0], 3.14)),
            (28, self.inst_map.get('u1', [1], 3.14)),
            (28, self.inst_map.get('u2', [0], 0, 0)),
            (28, self.inst_map.get('u2', [1], 0, 0)))
        for actual, expected in zip(sched.instructions, expected.instructions):
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])

    def test_parametric_input(self):
        """Test that scheduling works with parametric pulses as input."""
        qr = QuantumRegister(1)
        qc = QuantumCircuit(qr)
        qc.append(Gate('gauss', 1, []), qargs=[qr[0]])
        custom_gauss = Schedule(Play(Gaussian(duration=25, sigma=4, amp=0.5j), DriveChannel(0)))
        self.inst_map.add('gauss', [0], custom_gauss)
        sched = schedule(qc, self.backend, inst_map=self.inst_map)
        self.assertEqual(sched.instructions[0],
                         custom_gauss.instructions[0])
