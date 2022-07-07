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

"""Test delay instruction for quantum circuits."""

import numpy as np
from qiskit.circuit import Delay
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit import QuantumCircuit, CircuitInstruction
from qiskit.circuit.exceptions import CircuitError
from qiskit.test.base import QiskitTestCase


class TestDelayClass(QiskitTestCase):
    """Test delay instruction for quantum circuits."""

    def test_keep_units_after_adding_delays_to_circuit(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.delay(100, 0)
        qc.delay(200, 0, unit="s")
        qc.delay(300, 0, unit="ns")
        qc.delay(400, 0, unit="dt")
        self.assertEqual(qc.data[1].operation.unit, "dt")
        self.assertEqual(qc.data[2].operation.unit, "s")
        self.assertEqual(qc.data[3].operation.unit, "ns")
        self.assertEqual(qc.data[4].operation.unit, "dt")

    def test_fail_if_non_integer_duration_with_dt_unit_is_supplied(self):
        qc = QuantumCircuit(1)
        with self.assertRaises(CircuitError):
            qc.delay(0.5, 0, unit="dt")

    def test_fail_if_unknown_unit_is_supplied(self):
        qc = QuantumCircuit(1)
        with self.assertRaises(CircuitError):
            qc.delay(100, 0, unit="my_unit")

    def test_fail_if_negative_duration_is_supplied(self):
        qc = QuantumCircuit(1)
        with self.assertRaises(CircuitError):
            qc.delay(-1, 0, unit="dt")
        with self.assertRaises(CircuitError):
            qc.delay(-1.5, 0, unit="s")

    def test_add_delay_on_single_qubit_to_circuit(self):
        qc = QuantumCircuit(1)
        qc.h(0)
        qc.delay(100, 0)
        qc.delay(200, [0])
        qc.delay(300, qc.qubits[0])
        self.assertEqual(qc.data[1], CircuitInstruction(Delay(duration=100), qc.qubits, []))
        self.assertEqual(qc.data[2], CircuitInstruction(Delay(duration=200), qc.qubits, []))
        self.assertEqual(qc.data[3], CircuitInstruction(Delay(duration=300), qc.qubits, []))

    def test_add_delay_on_multiple_qubits_to_circuit(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.delay(100)
        qc.delay(200, range(2))
        qc.delay(300, qc.qubits[1:])

        expected = QuantumCircuit(3)
        expected.h(0)
        expected.delay(100, 0)
        expected.delay(100, 1)
        expected.delay(100, 2)
        expected.delay(200, 0)
        expected.delay(200, 1)
        expected.delay(300, 1)
        expected.delay(300, 2)
        self.assertEqual(qc, expected)

    def test_to_matrix_return_identity_matrix(self):
        actual = np.array(Delay(100))
        expected = np.array([[1, 0], [0, 1]], dtype=complex)
        self.assertTrue(np.array_equal(actual, expected))


class TestParameterizedDelay(QiskitTestCase):
    """Test delay instruction with parameterized duration."""

    def test_can_accept_parameterized_duration(self):
        dur = Parameter("t")
        self.assertEqual(Delay(dur).duration, dur)

    def test_can_append_parameterized_delay(self):
        dur = Parameter("t")
        qc = QuantumCircuit(1)
        qc.delay(dur)
        self.assertEqual(qc.data[0].operation, Delay(dur))

    def test_can_assign_duration_parameter(self):
        dur = Parameter("t")
        qc = QuantumCircuit(1)
        qc.delay(dur)
        qc.assign_parameters({dur: 100}, inplace=True)
        self.assertEqual(qc.data[0].operation.duration, 100)

    def test_fail_if_assign_invalid_duration_parameters(self):
        dur = Parameter("t")
        qc = QuantumCircuit(1)
        qc.delay(dur, unit="dt")
        with self.assertRaises(CircuitError):
            qc.assign_parameters({dur: 1 + 1j}, inplace=True)
        with self.assertRaises(CircuitError):
            qc.assign_parameters({dur: 0.5}, inplace=True)
        with self.assertRaises(CircuitError):
            qc.assign_parameters({dur: -1}, inplace=True)

    def test_can_assign_multiple_duration_parameters(self):
        durs = ParameterVector("dur", 2)
        qc = QuantumCircuit(2)
        qc.delay(durs[0])
        qc.delay(durs[1])
        qc = qc.assign_parameters([0, 1])
        durations = [inst.operation.duration for inst in qc.data]
        self.assertEqual(durations, [0, 0, 1, 1])
