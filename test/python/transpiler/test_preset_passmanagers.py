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

from qiskit.test import QiskitTestCase, METATEST, combine
from qiskit.compiler import transpile
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.test.mock import (FakeTenerife, FakeMelbourne, FakeRueschlikon, FakeTokyo,
                              FakePoughkeepsie)


def emptycircuit():
    """Empty circuit"""
    return QuantumCircuit()


def is_swap_mapped():
    """See https://github.com/Qiskit/qiskit-terra/issues/2532"""
    circuit = QuantumCircuit(5)
    circuit.cx(2, 4)
    return circuit


@METATEST
class TestPresetPassManager(QiskitTestCase):
    """Test preset passmanagers work as expected."""

    @combine(level=[0, 1, 2, 3],
             dsc='Test that coupling_map can be None (level={level})',
             name='coupling_map_none_level{level}')
    def test_no_coupling_map(self, level):
        """Test that coupling_map can be None"""
        q = QuantumRegister(2, name='q')
        circuit = QuantumCircuit(q)
        circuit.cz(q[0], q[1])
        result = transpile(circuit, basis_gates=['u1', 'u2', 'u3', 'cx'], optimization_level=level)
        self.assertIsInstance(result, QuantumCircuit)


@METATEST
class TestTranspileLevels(QiskitTestCase):
    """Test transpiler on fake backend"""

    @combine(circuit=[emptycircuit, is_swap_mapped],
             level=[0, 1, 2, 3],
             backend=[FakeTenerife(), FakeMelbourne(), FakeRueschlikon(), FakeTokyo(),
                      FakePoughkeepsie(), None],
             dsc='Transpiler {circuit.__name__} on {backend} backend at level {level}',
             name='{circuit.__name__}_{backend}_level{level}')
    def test(self, circuit, level, backend):
        """All the levels with all the backends"""
        result = transpile(circuit(), backend=backend, optimization_level=level, seed_transpiler=42)
        self.assertIsInstance(result, QuantumCircuit)
