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

    def test_no_coupling_map(self):
        """Test that coupling_map can be None"""
        q = QuantumRegister(2, name='q')
        test = QuantumCircuit(q)
        test.cz(q[0], q[1])
        for level in [0, 1, 2, 3]:
            with self.subTest(level=level):
                test2 = transpile(test, basis_gates=['u1', 'u2', 'u3', 'cx'],
                                  optimization_level=level)
                self.assertIsInstance(test2, QuantumCircuit)


class TestFakeBackendGeneric(QiskitTestCase):

    @staticmethod
    def _get_circuit():

        q = QuantumRegister(2)
        c = ClassicalRegister(2)

        circuit = QuantumCircuit(q, c)
        circuit.h(q[0])
        circuit.cx(q[0], q[1])
        circuit.measure(q, c)

        return circuit


class TestFakeBackendTenerife(TestFakeBackendGeneric):

    def setUp(self):
        self._circuit = super()._get_circuit()
        self._tenerife = FakeTenerife()

    def test_optimization_level_0(self):
        result = transpile(
            [self._circuit],
            backend=self._tenerife,
            optimization_level=0
        )
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_1(self):
        result = transpile(
            [self._circuit],
            backend=self._tenerife,
            optimization_level=1
        )
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_2(self):
        result = transpile(
            [self._circuit],
            backend=self._tenerife,
            optimization_level=2
        )
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_3(self):
        result = transpile(
            [self._circuit],
            backend=self._tenerife,
            optimization_level=3
        )
        self.assertIsInstance(result, QuantumCircuit)


class TestFakeBackendMelbourne(TestFakeBackendGeneric):

    def setUp(self):
        self._circuit = super()._get_circuit()
        self._melbourne = FakeMelbourne()

    def test_optimization_level_0(self):
        result = transpile(
            [self._circuit],
            backend=self._melbourne,
            optimization_level=0
        )
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_1(self):
        result = transpile(
            [self._circuit],
            backend=self._melbourne,
            optimization_level=1
        )
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_2(self):
        result = transpile(
            [self._circuit],
            backend=self._melbourne,
            optimization_level=2
        )
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_3(self):
        result = transpile(
            [self._circuit],
            backend=self._melbourne,
            optimization_level=3
        )
        self.assertIsInstance(result, QuantumCircuit)


class TestFakeBackendRueschlikon(TestFakeBackendGeneric):

    def setUp(self):
        self._circuit = super()._get_circuit()
        self._rueschlikon = FakeRueschlikon()

    def test_optimization_level_0(self):
        result = transpile(
            [self._circuit],
            backend=self._rueschlikon,
            optimization_level=0
        )
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_1(self):
        result = transpile(
            [self._circuit],
            backend=self._rueschlikon,
            optimization_level=1
        )
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_2(self):
        result = transpile(
            [self._circuit],
            backend=self._rueschlikon,
            optimization_level=2
        )
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_3(self):
        result = transpile(
            [self._circuit],
            backend=self._rueschlikon,
            optimization_level=3
        )
        self.assertIsInstance(result, QuantumCircuit)


class TestFakeBackendTokyo(TestFakeBackendGeneric):

    def setUp(self):
        self._circuit = super()._get_circuit()
        self._tokyo = FakeTokyo()

    def test_optimization_level_0(self):
        result = transpile(
            [self._circuit],
            backend=self._tokyo,
            optimization_level=0
        )
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_1(self):
        result = transpile(
            [self._circuit],
            backend=self._tokyo,
            optimization_level=1
        )
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_2(self):
        result = transpile(
            [self._circuit],
            backend=self._tokyo,
            optimization_level=2
        )
        self.assertIsInstance(result, QuantumCircuit)

    def test_optimization_level_3(self):
        result = transpile(
            [self._circuit],
            backend=self._tokyo,
            optimization_level=3
        )
        self.assertIsInstance(result, QuantumCircuit)
