# -*- coding: utf-8 -*-

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


class TestBitEvents(QiskitTestCase):
    """Tests for bit events."""

    # TODO: Add test for load program method when scheduled circuit is ready.

    def setUp(self) -> None:
        """Setup."""
        self.qubits = list(qiskit.QuantumRegister(2))
        self.clbits = list(qiskit.ClassicalRegister(2))

        self.instructions = [
            types.ScheduledGate(t0=0, operand=library.U1Gate,
                                duration=0, bits=[self.qubits[0]]),
            types.ScheduledGate(t0=0, operand=library.U2Gate,
                                duration=10, bits=[self.qubits[0]]),
            types.ScheduledGate(t0=10, operand=library.CXGate,
                                duration=50, bits=[self.qubits[0], self.qubits[1]]),
            types.ScheduledGate(t0=100, operand=library.U3Gate,
                                duration=20, bits=[self.qubits[0]]),
            types.ScheduledGate(t0=120, operand=library.Barrier,
                                duration=0, bits=[self.qubits[0], self.qubits[1]]),
            types.ScheduledGate(t0=120, operand=library.CXGate,
                                duration=50, bits=[self.qubits[1], self.qubits[0]]),
            types.ScheduledGate(t0=200, operand=library.Barrier,
                                duration=0, bits=[self.qubits[0]]),
            types.ScheduledGate(t0=200, operand=library.Measure,
                                duration=100, bits=[self.qubits[1], self.clbits[0]]),
        ]

    def test_gate_output(self):
        """Test gate output."""
        bit_event = events.BitEvents(self.qubits[0], self.instructions)

        gates = bit_event.gates()
        ref_list = [
            types.ScheduledGate(t0=0, operand=library.U1Gate,
                                duration=0, bits=[self.qubits[0]]),
            types.ScheduledGate(t0=0, operand=library.U2Gate,
                                duration=10, bits=[self.qubits[0]]),
            types.ScheduledGate(t0=10, operand=library.CXGate,
                                duration=50, bits=[self.qubits[0], self.qubits[1]]),
            types.ScheduledGate(t0=100, operand=library.U3Gate,
                                duration=20, bits=[self.qubits[0]]),
            types.ScheduledGate(t0=120, operand=library.CXGate,
                                duration=50, bits=[self.qubits[1], self.qubits[0]]),
            types.ScheduledGate(t0=200, operand=library.Measure,
                                duration=100, bits=[self.qubits[1], self.clbits[0]])
        ]

        self.assertListEqual(gates, ref_list)

    def test_barrier_output(self):
        """Test barrier output."""
        bit_event = events.BitEvents(self.qubits[0], self.instructions)

        barriers = bit_event.barriers()
        ref_list = [
            types.Barrier(t0=120, bits=[self.qubits[0], self.qubits[1]]),
            types.Barrier(t0=200, bits=[self.qubits[0]])
        ]

        self.assertListEqual(barriers, ref_list)

    def test_bit_link_output(self):
        """Test link output."""
        bit_event = events.BitEvents(self.qubits[0], self.instructions)

        links = bit_event.bit_links()
        ref_list = [
            types.GateLink(t0=10, operand=library.CXGate, bits=[self.qubits[0], self.qubits[1]])
        ]

        self.assertListEqual(links, ref_list)
