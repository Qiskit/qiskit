# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for BasePrimitive."""

import json

from qiskit import QuantumCircuit, pulse, transpile
from qiskit.primitives.utils import _circuit_key
from qiskit.providers.fake_provider import FakeAlmaden
from qiskit.test import QiskitTestCase


class TestCircuitKey(QiskitTestCase):
    """Tests for _circuit_key function"""

    def test_different_circuits(self):
        """Test collision of quantum circuits."""

        with self.subTest("Ry circuit"):

            def test_func(n):
                qc = QuantumCircuit(1, 1, name="foo")
                qc.ry(n, 0)
                return qc

            keys = [_circuit_key(test_func(i)) for i in range(5)]
            self.assertEqual(len(keys), len(set(keys)))

        with self.subTest("pulse circuit"):

            def test_with_scheduling(n):
                custom_gate = pulse.Schedule(name="custom_x_gate")
                custom_gate.insert(
                    0, pulse.Play(pulse.Constant(160 * n, 0.1), pulse.DriveChannel(0)), inplace=True
                )
                qc = QuantumCircuit(1)
                qc.x(0)
                qc.add_calibration("x", qubits=(0,), schedule=custom_gate)
                return transpile(qc, FakeAlmaden(), scheduling_method="alap")

            keys = [_circuit_key(test_with_scheduling(i)) for i in range(1, 5)]
            self.assertEqual(len(keys), len(set(keys)))

    def test_circuit_key_controlflow(self):
        """Test for a circuit with control flow."""
        qc = QuantumCircuit(2, 1)

        with qc.for_loop(range(5)):
            qc.h(0)
            qc.cx(0, 1)
            qc.measure(0, 0)
            qc.break_loop().c_if(0, True)

        self.assertIsInstance(hash(_circuit_key(qc)), int)
        self.assertIsInstance(json.dumps(_circuit_key(qc)), str)
