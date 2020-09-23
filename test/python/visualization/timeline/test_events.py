# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for event of timeline drawer."""

import qiskit
from qiskit.circuit import library
from qiskit.test import QiskitTestCase
from qiskit.visualization.timeline import events, types
from qiskit import QuantumCircuit, QuantumRegister, transpile


class TestLoadScheduledCircuit(QiskitTestCase):
    """Test for loading program."""

    def setUp(self) -> None:
        """Setup."""
        super().setUp()

        self.qr = QuantumRegister(3)

        circ = QuantumCircuit(self.qr)
        circ.delay(duration=100, qarg=self.qr[2])
        circ.barrier(self.qr[0], self.qr[1], self.qr[2])
        circ.h(self.qr[0])
        circ.cx(self.qr[0], self.qr[1])

        self.circ = transpile(circ,
                              scheduling_method='alap',
                              basis_gates=['h', 'cx'],
                              instruction_durations=[('h', 0, 200), ('cx', [0, 1], 1000)],
                              optimization_level=0)

    def test_create_from_program(self):
        """Test factory method."""
        bit_event_q0 = events.BitEvents.load_program(self.circ, self.qr[0])
        bit_event_q1 = events.BitEvents.load_program(self.circ, self.qr[1])
        bit_event_q2 = events.BitEvents.load_program(self.circ, self.qr[2])

        gates_q0 = list(bit_event_q0.get_gates())
        links_q0 = list(bit_event_q0.get_bit_links())
        barriers_q0 = list(bit_event_q0.get_barriers())

        self.assertEqual(len(gates_q0), 2)
        self.assertEqual(len(links_q0), 1)
        self.assertEqual(len(barriers_q0), 1)

        # h gate
        self.assertEqual(gates_q0[0].t0, 100)

        # cx gate
        self.assertEqual(gates_q0[1].t0, 300)

        # link
        self.assertEqual(links_q0[1].t0, 800)

        # barrier
        self.assertEqual(barriers_q0[1].t0, 100)

        gates_q1 = list(bit_event_q1.get_gates())
        links_q1 = list(bit_event_q1.get_bit_links())
        barriers_q1 = list(bit_event_q1.get_barriers())

        self.assertEqual(len(gates_q1), 1)
        self.assertEqual(len(links_q1), 0)
        self.assertEqual(len(barriers_q1), 1)

        # cx gate
        self.assertEqual(gates_q0[0].t0, 300)

        # barrier
        self.assertEqual(barriers_q1[1].t0, 100)

        gates_q2 = list(bit_event_q2.get_gates())
        links_q2 = list(bit_event_q2.get_bit_links())
        barriers_q2 = list(bit_event_q2.get_barriers())

        self.assertEqual(len(gates_q2), 0)
        self.assertEqual(len(links_q2), 0)
        self.assertEqual(len(barriers_q2), 1)

        # barrier
        self.assertEqual(barriers_q2[0].t0, 100)


class TestBitEvents(QiskitTestCase):
    """Tests for bit events."""

    def setUp(self) -> None:
        """Setup."""
        super().setUp()

        self.qubits = list(qiskit.QuantumRegister(2))
        self.clbits = list(qiskit.ClassicalRegister(2))

        self.instructions = [
            types.ScheduledGate(t0=0, operand=library.U1Gate(0),
                                duration=0, bits=[self.qubits[0]]),
            types.ScheduledGate(t0=0, operand=library.U2Gate(0, 0),
                                duration=10, bits=[self.qubits[0]]),
            types.ScheduledGate(t0=10, operand=library.CXGate(),
                                duration=50, bits=[self.qubits[0], self.qubits[1]]),
            types.ScheduledGate(t0=100, operand=library.U3Gate(0, 0, 0),
                                duration=20, bits=[self.qubits[0]]),
            types.ScheduledGate(t0=120, operand=library.Barrier(2),
                                duration=0, bits=[self.qubits[0], self.qubits[1]]),
            types.ScheduledGate(t0=120, operand=library.CXGate(),
                                duration=50, bits=[self.qubits[1], self.qubits[0]]),
            types.ScheduledGate(t0=200, operand=library.Barrier(1),
                                duration=0, bits=[self.qubits[0]]),
            types.ScheduledGate(t0=200, operand=library.Measure(),
                                duration=100, bits=[self.qubits[0], self.clbits[0]]),
        ]

    def test_gate_output(self):
        """Test gate output."""
        bit_event = events.BitEvents(self.qubits[0], self.instructions)

        gates = list(bit_event.get_gates())
        ref_list = [
            types.ScheduledGate(t0=0, operand=library.U1Gate(0),
                                duration=0, bits=[self.qubits[0]]),
            types.ScheduledGate(t0=0, operand=library.U2Gate(0, 0),
                                duration=10, bits=[self.qubits[0]]),
            types.ScheduledGate(t0=10, operand=library.CXGate(),
                                duration=50, bits=[self.qubits[0], self.qubits[1]]),
            types.ScheduledGate(t0=100, operand=library.U3Gate(0, 0, 0),
                                duration=20, bits=[self.qubits[0]]),
            types.ScheduledGate(t0=120, operand=library.CXGate(),
                                duration=50, bits=[self.qubits[1], self.qubits[0]]),
            types.ScheduledGate(t0=200, operand=library.Measure(),
                                duration=100, bits=[self.qubits[0], self.clbits[0]])
        ]

        self.assertListEqual(gates, ref_list)

    def test_barrier_output(self):
        """Test barrier output."""
        bit_event = events.BitEvents(self.qubits[0], self.instructions)

        barriers = list(bit_event.get_barriers())
        ref_list = [
            types.Barrier(t0=120, bits=[self.qubits[0], self.qubits[1]]),
            types.Barrier(t0=200, bits=[self.qubits[0]])
        ]

        self.assertListEqual(barriers, ref_list)

    def test_bit_link_output(self):
        """Test link output."""
        bit_event = events.BitEvents(self.qubits[0], self.instructions)

        links = list(bit_event.get_bit_links())
        ref_list = [
            types.GateLink(t0=35.0, operand=library.CXGate(),
                           bits=[self.qubits[0], self.qubits[1]]),
            types.GateLink(t0=250.0, operand=library.Measure(),
                           bits=[self.qubits[0], self.clbits[0]])
        ]

        self.assertListEqual(links, ref_list)
