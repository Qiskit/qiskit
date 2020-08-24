# -*- coding: utf-8 -*-

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

    def test_xy4_simple(self):
        """Test that the pass replaces large enough delay blocks with XY4 DD sequences.
        """
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.delay(2000)
        circuit.h(0)

        pass_manager = PassManager()
        pass_manager.append(XY4Pass(self.backend_prop, self.dt_in_sec))
        actual = pass_manager.run(circuit)
        
        expected = QuantumCircuit(1)
        expected.h(0)
        expected.delay(247, 0)
        expected.delay(45, 0)
        expected.x(0)
        expected.delay(45, 0)
        expected.y(0)
        expected.delay(45, 0)
        expected.x(0)
        expected.delay(45, 0)
        expected.y(0)
        expected.delay(293, 0)
        expected.h(0)

        self.assertEqual(actual, expected)

    def test_xy4_multiple(self):
        """Test that the pass replaces large enough delay blocks with multiple XY4 DD sequences.
        """
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.delay(3000)
        circuit.h(0)

        pass_manager = PassManager()
        pass_manager.append(XY4Pass(self.backend_prop, self.dt_in_sec))
        actual = pass_manager.run(circuit)
        
        expected = QuantumCircuit(1)
        expected.h(0)
        expected.delay(17, 0)
        for _ in range(2):
            expected.delay(45, 0)
            expected.x(0)
            expected.delay(45, 0)
            expected.y(0)
            expected.delay(45, 0)
            expected.x(0)
            expected.delay(45, 0)
            expected.y(0)
        expected.delay(63, 0)
        expected.h(0)

        self.assertEqual(actual, expected)

    def test_xy4_not_first(self):
        """Test that the pass replaces large enough delay blocks with XY4 DD sequences except 
        for the first delay block.
        """
        circuit = QuantumCircuit(1)
        circuit.delay(2000)
        circuit.h(0)
        circuit.delay(2000)
        circuit.h(0)

        pass_manager = PassManager()
        pass_manager.append(XY4Pass(self.backend_prop, self.dt_in_sec))
        actual = pass_manager.run(circuit)
        
        expected = QuantumCircuit(1)
        expected.delay(2000, 0)
        expected.h(0)
        expected.delay(247, 0)
        expected.delay(45, 0)
        expected.x(0)
        expected.delay(45, 0)
        expected.y(0)
        expected.delay(45, 0)
        expected.x(0)
        expected.delay(45, 0)
        expected.y(0)
        expected.delay(293, 0)
        expected.h(0)

        self.assertEqual(actual, expected)
