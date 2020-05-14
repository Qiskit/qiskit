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

"""Test Qiskit's Gate class."""


import numpy as np

from qiskit.circuit import Gate
from qiskit.circuit import Parameter
from qiskit.circuit import QuantumCircuit
from qiskit.circuit import QuantumRegister
from qiskit.test import QiskitTestCase
from qiskit.circuit.exceptions import CircuitError


class TestGate(QiskitTestCase):
    """Gate class tests."""

    def test_to_matrix_from_definition(self):
        """Test generating matrix from definition."""
        qr = QuantumRegister(2)
        circ = QuantumCircuit(qr, name='circ')
        circ.t(qr[1])
        circ.u3(0.1, 0.2, -0.2, qr[0])
        gate_matrix = circ.to_gate().to_matrix()
        self.assertIsInstance(gate_matrix, np.ndarray)
        self.assertEqual(gate_matrix.shape, (4, 4))

    def test_to_matrix_from_parameterized_definition(self):
        """Test generating matrix from definition."""
        # Although this currently fails it could be allowed if Operator
        # could do symbolic composition.
        qr = QuantumRegister(1)
        angle = Parameter('α')
        circ = QuantumCircuit(qr, name='circ')
        circ.u1(angle, qr[0])
        with self.assertRaises(TypeError):
            circ.to_gate().to_matrix()

    def test_to_matrix_from_opaque_gate(self):
        """Test to_matrix raises for opaque gate."""
        opaque = Gate('opaque', 2, [])
        with self.assertRaises(CircuitError):
            opaque.to_matrix()
