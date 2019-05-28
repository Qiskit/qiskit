# -*- coding: utf-8 -*-

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

"""Tests preset pass manager functionalities"""

from qiskit.test import QiskitTestCase
from qiskit.compiler import transpile
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit.test.mock import FakeTenerife, FakeMelbourne, FakeRueschlikon, FakeTokyo


class TestPresetPassManager(QiskitTestCase):
    """Test preset passmanagers work as expected."""

    def setUp(self):
        q = QuantumRegister(2, name='q')
        self.circuit = QuantumCircuit(q)
        self.circuit.cz(q[0], q[1])
        self.basis_gates = ['u1', 'u2', 'u3', 'cx']

    def test_no_coupling_map_level_0(self):
        """Test that coupling_map can be None with level 0"""
        result = transpile(self.circuit, basis_gates=self.basis_gates, optimization_level=0)
        self.assertIsInstance(result, QuantumCircuit)

    def test_no_coupling_map_level_1(self):
        """Test that coupling_map can be None with level 1"""
        result = transpile(self.circuit, basis_gates=self.basis_gates, optimization_level=1)
        self.assertIsInstance(result, QuantumCircuit)

    def test_no_coupling_map_level_2(self):
        """Test that coupling_map can be None with level 2"""
        result = transpile(self.circuit, basis_gates=self.basis_gates, optimization_level=2)
        self.assertIsInstance(result, QuantumCircuit)

    def test_no_coupling_map_level_3(self):
        """Test that coupling_map can be None with level 3"""
        result = transpile(self.circuit, basis_gates=self.basis_gates, optimization_level=3)
        self.assertIsInstance(result, QuantumCircuit)


class TestFakeBackends(QiskitTestCase):
    """Test transpiler on fake backends"""

    def create_circuit(self):
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        circuit = QuantumCircuit(q, c)
        circuit.h(q[0])
        circuit.cx(q[0], q[1])
        circuit.measure(q, c)
        return circuit

    def test_all_optimization_levels_and_all_backends(self):
        for backend in [FakeTenerife(), FakeMelbourne(), FakeRueschlikon(), FakeTokyo()]:
            for optimization_level in range(4):
                with self.subTest(optimization_level=optimization_level, backend=backend.name()):
                    result = transpile(self.create_circuit(),
                                       backend=backend,
                                       optimization_level=optimization_level)
                    self.assertIsInstance(result, QuantumCircuit)
