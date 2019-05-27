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


class TestFakeBackendTenerife(QiskitTestCase):
    """Test transpiler on Tenerife fake backend"""

    def setUp(self):
        self.tenerife = FakeTenerife()
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        self.circuit = QuantumCircuit(q, c)
        self.circuit.h(q[0])
        self.circuit.cx(q[0], q[1])
        self.circuit.measure(q, c)

    def test_optimization_level_0(self):
        """Test transpiler on Tenerife fake backend with optimization 0"""
        result = transpile(self.circuit, backend=self.tenerife, optimization_level=0)
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_1(self):
        """Test transpiler on Tenerife fake backend with optimization 1"""
        result = transpile(self.circuit, backend=self.tenerife, optimization_level=1)
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_2(self):
        """Test transpiler on Tenerife fake backend with optimization 2"""
        result = transpile(self.circuit, backend=self.tenerife, optimization_level=2)
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_3(self):
        """Test transpiler on Tenerife fake backend with optimization 3"""
        result = transpile(self.circuit, backend=self.tenerife, optimization_level=3)
        self.assertIsInstance(result, QuantumCircuit)


class TestFakeBackendMelbourne(QiskitTestCase):
    """Test transpiler on Melbourne fake backend"""

    def setUp(self):
        self.melbourne = FakeMelbourne()
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        self.circuit = QuantumCircuit(q, c)
        self.circuit.h(q[0])
        self.circuit.cx(q[0], q[1])
        self.circuit.measure(q, c)

    def test_optimization_level_0(self):
        """Test transpiler on Melbourne fake backend with optimization 0"""
        result = transpile(self.circuit, backend=self.melbourne, optimization_level=0)
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_1(self):
        """Test transpiler on Melbourne fake backend with optimization 1"""
        result = transpile(self.circuit, backend=self.melbourne, optimization_level=1)
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_2(self):
        """Test transpiler on Melbourne fake backend with optimization 2"""
        result = transpile(self.circuit, backend=self.melbourne, optimization_level=2)
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_3(self):
        """Test transpiler on Melbourne fake backend with optimization 3"""
        result = transpile(self.circuit, backend=self.melbourne, optimization_level=3)
        self.assertIsInstance(result, QuantumCircuit)


class TestFakeBackendRueschlikon(QiskitTestCase):
    """Test transpiler on Rueschlikon fake backend"""

    def setUp(self):
        self.rueschlikon = FakeRueschlikon()
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        self.circuit = QuantumCircuit(q, c)
        self.circuit.h(q[0])
        self.circuit.cx(q[0], q[1])
        self.circuit.measure(q, c)

    def test_optimization_level_0(self):
        """Test transpiler on Rueschlikon fake backend with optimization 0"""
        result = transpile(self.circuit, backend=self.rueschlikon, optimization_level=0)
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_1(self):
        """Test transpiler on Rueschlikon fake backend with optimization 1"""
        result = transpile(self.circuit, backend=self.rueschlikon, optimization_level=1)
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_2(self):
        """Test transpiler on Rueschlikon fake backend with optimization 2"""
        result = transpile(self.circuit, backend=self.rueschlikon, optimization_level=2)
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_3(self):
        """Test transpiler on Rueschlikon fake backend with optimization 3"""
        result = transpile(self.circuit, backend=self.rueschlikon, optimization_level=3)
        self.assertIsInstance(result, QuantumCircuit)


class TestFakeBackendTokyo(QiskitTestCase):
    """Test transpiler on Tokyo fake backend"""

    def setUp(self):
        self.tokyo = FakeTokyo()
        q = QuantumRegister(2)
        c = ClassicalRegister(2)
        self.circuit = QuantumCircuit(q, c)
        self.circuit.h(q[0])
        self.circuit.cx(q[0], q[1])
        self.circuit.measure(q, c)

    def test_optimization_level_0(self):
        """Test transpiler on Tokyo fake backend with optimization 0"""
        result = transpile(self.circuit, backend=self.tokyo, optimization_level=0)
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_1(self):
        """Test transpiler on Tokyo fake backend with optimization 1"""
        result = transpile(self.circuit, backend=self.tokyo, optimization_level=1)
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_2(self):
        """Test transpiler on Tokyo fake backend with optimization 2"""
        result = transpile(self.circuit, backend=self.tokyo, optimization_level=2)
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_3(self):
        """Test transpiler on Tokyo fake backend with optimization 3"""
        result = transpile(self.circuit, backend=self.tokyo, optimization_level=3)
        self.assertIsInstance(result, QuantumCircuit)
