# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests preset pass managers with 1Q backend"""
from test import combine
from ddt import ddt

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.test import QiskitTestCase
from qiskit.providers.fake_provider import Fake1Q
from qiskit.transpiler import TranspilerError


def emptycircuit():
    """Empty circuit"""
    return QuantumCircuit()


def circuit_3516():
    """Circuit from https://github.com/Qiskit/qiskit-terra/issues/3516 should fail"""
    circuit = QuantumCircuit(2, 1)
    circuit.h(0)
    circuit.ry(0.11, 1)
    circuit.measure([0], [0])
    return circuit


@ddt
class Test1QFailing(QiskitTestCase):
    """1Q tests that should fail."""

    @combine(
        circuit=[circuit_3516],
        level=[0, 1, 2, 3],
        dsc="Transpiling {circuit.__name__} at level {level} should fail",
        name="{circuit.__name__}_level{level}_fail",
    )
    def test(self, circuit, level):
        """All the levels with all the 1Q backend"""
        with self.assertRaises(TranspilerError):
            transpile(circuit(), backend=Fake1Q(), optimization_level=level, seed_transpiler=42)


@ddt
class Test1QWorking(QiskitTestCase):
    """1Q tests that should work."""

    @combine(
        circuit=[emptycircuit],
        level=[0, 1, 2, 3],
        dsc="Transpiling {circuit.__name__} at level {level} should work",
        name="{circuit.__name__}_level{level}_valid",
    )
    def test_device(self, circuit, level):
        """All the levels with all the 1Q backend"""
        result = transpile(
            circuit(), backend=Fake1Q(), optimization_level=level, seed_transpiler=42
        )
        self.assertIsInstance(result, QuantumCircuit)

    @combine(
        circuit=[circuit_3516],
        level=[0, 1, 2, 3],
        dsc="Transpiling {circuit.__name__} at level {level} should work for simulator",
        name="{circuit.__name__}_level{level}_valid",
    )
    def test_simulator(self, circuit, level):
        """All the levels with all the 1Q simulator backend"""
        # Set fake backend config to simulator
        backend = Fake1Q()
        backend._configuration.simulator = True
        result = transpile(circuit(), backend=backend, optimization_level=level, seed_transpiler=42)
        self.assertIsInstance(result, QuantumCircuit)
