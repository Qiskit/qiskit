# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests preset pass manager whith faulty backends"""

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from qiskit.test import QiskitTestCase
from qiskit.test.mock import FakeOurenseFaultyQ1, FakeOurenseFaultyCX13


class TestFaultyQ1(QiskitTestCase):
    """Test preset passmanagers with FakeOurenseFaultyQ1."""

    def setUp(self) -> None:
        self.backend = FakeOurenseFaultyQ1()

    def test_level_0(self):
        """Test level 0 Ourense backend with a faulty Q1 """
        q = QuantumRegister(4, name='q')
        circuit = QuantumCircuit(q)
        circuit.h(q)
        circuit.cz(q[0], q[1])
        circuit.measure_all()
        result = transpile(circuit, backend=self.backend, optimization_level=0, seed_transpiler=42)
        print(result)
