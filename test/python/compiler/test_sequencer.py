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

# pylint: disable=missing-function-docstring

"""Tests basic functionality of the sequence function"""
import unittest

from qiskit import QuantumCircuit, pulse
from qiskit.compiler import sequence, transpile, schedule
from qiskit.pulse.transforms import pad
from qiskit.test.mock import FakeParis

from qiskit.test import QiskitTestCase


class TestSequence(QiskitTestCase):
    """Test sequence function."""

    def setUp(self):
        super().setUp()
        self.backend = FakeParis()

    def test_sequence_empty(self):
        self.assertEqual(sequence([], self.backend), [])

    def test_transpile_and_sequence_agree_with_schedule(self):
        qc = QuantumCircuit(2, name="bell")
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        sc = transpile(qc, self.backend, scheduling_method="alap")
        actual = sequence(sc, self.backend)
        expected = schedule(transpile(qc, self.backend), self.backend)
        self.assertEqual(actual, pad(expected))

    def test_transpile_and_sequence_agree_with_schedule_for_circuit_with_delay(self):
        qc = QuantumCircuit(1, 1, name="t2")
        qc.h(0)
        qc.delay(500, 0, unit="ns")
        qc.h(0)
        qc.measure(0, 0)
        sc = transpile(qc, self.backend, scheduling_method="alap")
        actual = sequence(sc, self.backend)
        expected = schedule(transpile(qc, self.backend), self.backend)
        self.assertEqual(
            actual.exclude(instruction_types=[pulse.Delay]),
            expected.exclude(instruction_types=[pulse.Delay]),
        )

    @unittest.skip("not yet determined if delays on ancilla should be removed or not")
    def test_transpile_and_sequence_agree_with_schedule_for_circuits_without_measures(self):
        qc = QuantumCircuit(2, name="bell_without_measurement")
        qc.h(0)
        qc.cx(0, 1)
        sc = transpile(qc, self.backend, scheduling_method="alap")
        actual = sequence(sc, self.backend)
        expected = schedule(transpile(qc, self.backend), self.backend)
        self.assertEqual(actual, pad(expected))
