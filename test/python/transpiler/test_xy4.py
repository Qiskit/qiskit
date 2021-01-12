# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests XY4 dynamical decoupling pass."""
import numpy as np

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passes import XY4
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeAlmaden


class TestXY4(QiskitTestCase):
    """Test the XY4 DD pass."""

    def setUp(self):
        super().setUp()
        self.backend = FakeAlmaden()
        self.dt_in_sec = self.backend.configuration().dt
        self.backend_prop = self.backend.properties()
        self.gate_length = self.backend_prop.gate_length('u3', 0)

    def test_xy4_simple(self):
        """Test that the pass replaces large enough delay blocks with XY4 DD sequences."""
        delay_len = 400e-9
        inter_gate_delay = 10e-9

        qc = QuantumCircuit(1)
        qc.h(0)
        qc.delay(delay_len, unit='s')
        qc.h(0)

        leftover_delay = (delay_len - 4 * self.gate_length - 3 * inter_gate_delay)

        pass_manager = PassManager()
        pass_manager.append(XY4(self.backend_prop))
        actual = pass_manager.run(qc)

        expected = QuantumCircuit(1)
        expected.h(0)
        expected.delay(leftover_delay / 2, 0, unit='s')
        expected.u3(np.pi, 0, np.pi, 0)
        expected.delay(inter_gate_delay, 0, unit='s')
        expected.u3(np.pi, np.pi / 2, np.pi / 2, 0)
        expected.delay(inter_gate_delay, 0, unit='s')
        expected.u3(np.pi, 0, np.pi, 0)
        expected.delay(inter_gate_delay, 0, unit='s')
        expected.u3(np.pi, np.pi / 2, np.pi / 2, 0)
        expected.delay(leftover_delay / 2, 0, unit='s')
        expected.h(0)

        self.assertEqual(actual, expected)

    def test_xy4_multiple(self):
        """Test that the pass replaces large enough delay blocks with multiple XY4 DD sequences.
        """
        delay_len = 800e-9
        inter_gate_delay = 10e-9

        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.delay(delay_len, unit='s')
        circuit.h(0)

        leftover_delay = (delay_len - 8 * self.gate_length - 7 * inter_gate_delay)

        pass_manager = PassManager()
        pass_manager.append(XY4(self.backend_prop))
        actual = pass_manager.run(circuit)

        expected = QuantumCircuit(1)
        expected.h(0)
        expected.delay(leftover_delay / 2, 0, unit='s')

        for i in range(2):
            expected.u3(np.pi, 0, np.pi, 0)
            expected.delay(inter_gate_delay, 0, unit='s')
            expected.u3(np.pi, np.pi / 2, np.pi / 2, 0)
            expected.delay(inter_gate_delay, 0, unit='s')
            expected.u3(np.pi, 0, np.pi, 0)
            expected.delay(inter_gate_delay, 0, unit='s')
            expected.u3(np.pi, np.pi / 2, np.pi / 2, 0)
            if i < 1:
                expected.delay(inter_gate_delay, 0, unit='s')

        expected.delay(leftover_delay / 2, 0, unit='s')
        expected.h(0)

        self.assertEqual(actual, expected)

    def test_xy4_not_first(self):
        """Test that the pass replaces large enough delay blocks with XY4 DD
        sequences except for the first delay block.
        """
        delay_len = 400e-9
        inter_gate_delay = 10e-9

        circuit = QuantumCircuit(1)
        circuit.delay(delay_len, unit='s')
        circuit.h(0)
        circuit.delay(delay_len, unit='s')
        circuit.h(0)

        leftover_delay = (delay_len - 4 * self.gate_length - 3 * inter_gate_delay)

        pass_manager = PassManager()
        pass_manager.append(XY4(self.backend_prop))
        actual = pass_manager.run(circuit)

        expected = QuantumCircuit(1)
        expected.delay(delay_len, unit='s')
        expected.h(0)
        expected.delay(leftover_delay / 2, 0, unit='s')
        expected.u3(np.pi, 0, np.pi, 0)
        expected.delay(inter_gate_delay, 0, unit='s')
        expected.u3(np.pi, np.pi / 2, np.pi / 2, 0)
        expected.delay(inter_gate_delay, 0, unit='s')
        expected.u3(np.pi, 0, np.pi, 0)
        expected.delay(inter_gate_delay, 0, unit='s')
        expected.u3(np.pi, np.pi / 2, np.pi / 2, 0)
        expected.delay(leftover_delay / 2, 0, unit='s')
        expected.h(0)

        self.assertEqual(actual, expected)
