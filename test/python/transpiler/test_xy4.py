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

"""Tests XY4 Dynamical Decoupling Pass"""
import numpy as np

from qiskit import QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passmanager_config import PassManagerConfig
from qiskit.transpiler.passes import XY4Pass
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeAlmaden

class TestXY4(QiskitTestCase):
    """Test the XY4 DD pass."""

    def setUp(self):
        self.backend = FakeAlmaden()
        self.dt_in_sec = self.backend.configuration().dt
        self.backend_prop = self.backend.properties()
        self.gate_length = self.backend_prop._gates['u3'][(0,)]['gate_length'][0]

    def test_xy4_simple(self):
        """Test that the pass replaces large enough delay blocks with XY4 DD sequences.
        """
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.delay(4e-7, unit='s')
        circuit.h(0)

        leftover_delay = (4e-7 - 4 * self.gate_length - 3 * 10e-9) / 2

        pass_manager = PassManager()
        pass_manager.append(XY4Pass(self.backend_prop))
        actual = pass_manager.run(circuit)

        expected = QuantumCircuit(1)
        expected.h(0)
        expected.delay(leftover_delay, 0, unit='s')
        expected.u3(np.pi, 0, np.pi, 0)
        expected.delay(10e-9, 0, unit='s')
        expected.u3(np.pi, np.pi/2, np.pi/2, 0)
        expected.delay(10e-9, 0, unit='s')
        expected.u3(np.pi, 0, np.pi, 0)
        expected.delay(10e-9, 0, unit='s')
        expected.u3(np.pi, np.pi/2, np.pi/2, 0)
        expected.delay(leftover_delay, 0, unit='s')
        expected.h(0)

        self.assertEqual(actual, expected)

    def test_xy4_multiple(self):
        """Test that the pass replaces large enough delay blocks with multiple XY4 DD sequences.
        """
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.delay(8e-7, unit='s')
        circuit.h(0)

        leftover_delay = (8e-7 - 8 * self.gate_length - 7 * 10e-9) / 2

        pass_manager = PassManager()
        pass_manager.append(XY4Pass(self.backend_prop))
        actual = pass_manager.run(circuit)

        expected = QuantumCircuit(1)
        expected.h(0)
        expected.delay(leftover_delay, 0)

        first = True
        for _ in range(2):
            if not first:
                expected.delay(10e-9, 0, unit='s')
            first = False
            expected.u3(np.pi, 0, np.pi, 0)
            expected.delay(10e-9, 0, unit='s')
            expected.u3(np.pi, np.pi/2, np.pi/2, 0)
            expected.delay(10e-9, 0, unit='s')
            expected.u3(np.pi, 0, np.pi, 0)
            expected.delay(10e-9, 0, unit='s')
            expected.u3(np.pi, np.pi/2, np.pi/2, 0)

        expected.delay(leftover_delay, 0)
        expected.h(0)

        self.assertEqual(actual, expected)

    def test_xy4_not_first(self):
        """Test that the pass replaces large enough delay blocks with XY4 DD sequences except 
           for the first delay block.
        """
        circuit = QuantumCircuit(1)
        circuit.delay(4e-7, unit='s')
        circuit.h(0)
        circuit.delay(4e-7, unit='s')
        circuit.h(0)

        leftover_delay = (4e-7 - 4 * self.gate_length - 3 * 10e-9) / 2

        pass_manager = PassManager()
        pass_manager.append(XY4Pass(self.backend_prop))
        actual = pass_manager.run(circuit)

        expected = QuantumCircuit(1)
        expected.delay(4e-7, unit='s')
        expected.h(0)
        expected.delay(leftover_delay, 0, unit='s')
        expected.u3(np.pi, 0, np.pi, 0)
        expected.delay(10e-9, 0, unit='s')
        expected.u3(np.pi, np.pi/2, np.pi/2, 0)
        expected.delay(10e-9, 0, unit='s')
        expected.u3(np.pi, 0, np.pi, 0)
        expected.delay(10e-9, 0, unit='s')
        expected.u3(np.pi, np.pi/2, np.pi/2, 0)
        expected.delay(leftover_delay, 0, unit='s')
        expected.h(0)

        self.assertEqual(actual, expected)
