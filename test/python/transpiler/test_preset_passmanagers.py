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

"""Tests preset pass manager API"""

from qiskit.test import QiskitTestCase
from qiskit.compiler import transpile
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.test.mock import FakeTenerife, FakeMelbourne, FakeRueschlikon, FakeTokyo


def _create_circuit():
    """A dummy circuit"""
    q = QuantumRegister(2, name='q')
    circuit = QuantumCircuit(q)
    circuit.cz(q[0], q[1])
    return circuit


class TestDefaultPassManager(QiskitTestCase):
    """qiskit/transpiler/preset_passmanagers/default.py"""

    def test_all_optimization_levels_and_all_backends(self):
        """Test transpile API without optimization_level"""
        for backend in [FakeTenerife(), FakeMelbourne(), FakeRueschlikon(), FakeTokyo(), None]:
            with self.subTest(backend=backend):
                result = transpile(_create_circuit(), backend=backend)
                self.assertIsInstance(result, QuantumCircuit)


class TestLevel0PassManager(QiskitTestCase):
    """qiskit/transpiler/preset_passmanagers/level0.py"""

    def test_all_backends(self):
        """Test transpile API with optimization_level 0"""
        for backend in [FakeTenerife(), FakeMelbourne(), FakeRueschlikon(), FakeTokyo(), None]:
            with self.subTest(backend=backend):
                result = transpile(_create_circuit(), backend=backend, optimization_level=0)
                self.assertIsInstance(result, QuantumCircuit)


class TestLevel1PassManager(QiskitTestCase):
    """qiskit/transpiler/preset_passmanagers/level1.py"""

    def test_all_backends(self):
        """Test transpile API with optimization_level 1"""
        for backend in [FakeTenerife(), FakeMelbourne(), FakeRueschlikon(), FakeTokyo(), None]:
            with self.subTest(backend=backend):
                result = transpile(_create_circuit(), backend=backend, optimization_level=1)
                self.assertIsInstance(result, QuantumCircuit)


class TestLevel2PassManager(QiskitTestCase):
    """qiskit/transpiler/preset_passmanagers/level2.py"""

    def test_all_backends(self):
        """Test transpile API with optimization_level 2"""
        for backend in [FakeTenerife(), FakeMelbourne(), FakeRueschlikon(), FakeTokyo(), None]:
            with self.subTest(backend=backend):
                result = transpile(_create_circuit(), backend=backend, optimization_level=2)
                self.assertIsInstance(result, QuantumCircuit)


class TestLevel3PassManager(QiskitTestCase):
    """qiskit/transpiler/preset_passmanagers/level3.py"""

    def test_all_backends(self):
        """Test transpile API with optimization_level 3"""
        for backend in [FakeTenerife(), FakeMelbourne(), FakeRueschlikon(), FakeTokyo(), None]:
            with self.subTest(backend=backend):
                result = transpile(_create_circuit(), backend=backend, optimization_level=3)
                self.assertIsInstance(result, QuantumCircuit)
