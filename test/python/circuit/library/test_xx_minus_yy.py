# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test XXMinusYYGate."""

import unittest
import numpy as np
from scipy.linalg import expm
from ddt import data, ddt, unpack

from qiskit import QuantumCircuit
from qiskit.circuit.library import RZGate, XGate, YGate
from qiskit.quantum_info import Operator
from qiskit.circuit.library import XXMinusYYGate


@ddt
class TestXXMinusYYGate(unittest.TestCase):
    """Test XXMinusYYGate."""

    @unpack
    @data(
        (0, 0, np.eye(4)),
        (
            np.pi / 2,
            np.pi / 2,
            np.array(
                [
                    [np.sqrt(2) / 2, 0, 0, -np.sqrt(2) / 2],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
                ]
            ),
        ),
        (
            np.pi,
            np.pi / 2,
            np.array([[0, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0], [1, 0, 0, 0]]),
        ),
        (
            2 * np.pi,
            np.pi / 2,
            np.array([[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]]),
        ),
        (
            np.pi / 2,
            np.pi,
            np.array(
                [
                    [np.sqrt(2) / 2, 0, 0, 1j * np.sqrt(2) / 2],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0],
                    [1j * np.sqrt(2) / 2, 0, 0, np.sqrt(2) / 2],
                ]
            ),
        ),
        (4 * np.pi, 0, np.eye(4)),
    )
    def test_matrix(self, theta: float, beta: float, expected: np.ndarray):
        """Test matrix."""
        gate = XXMinusYYGate(theta, beta)
        np.testing.assert_allclose(np.array(gate), expected, atol=1e-7)

    def test_exponential_formula(self):
        """Test exponential formula."""
        theta, beta = np.random.uniform(-10, 10, size=2)
        theta = np.pi / 2
        beta = 0.0
        gate = XXMinusYYGate(theta, beta)
        x = np.array(XGate())
        y = np.array(YGate())
        xx = np.kron(x, x)
        yy = np.kron(y, y)
        rz1 = np.kron(np.array(RZGate(beta)), np.eye(2))
        np.testing.assert_allclose(
            np.array(gate),
            rz1 @ expm(-0.25j * theta * (xx - yy)) @ rz1.T.conj(),
            atol=1e-7,
        )

    def test_inverse(self):
        """Test inverse."""
        theta, beta = np.random.uniform(-10, 10, size=2)
        gate = XXMinusYYGate(theta, beta)
        circuit = QuantumCircuit(2)
        circuit.append(gate, [0, 1])
        circuit.append(gate.inverse(), [0, 1])
        assert Operator(circuit).equiv(np.eye(4), atol=1e-7)

    def test_decompose(self):
        """Test decomposition."""
        theta, beta = np.random.uniform(-10, 10, size=2)
        gate = XXMinusYYGate(theta, beta)
        circuit = QuantumCircuit(2)
        circuit.append(gate, [0, 1])
        decomposed_circuit = circuit.decompose()
        assert len(decomposed_circuit) > len(circuit)
        assert Operator(circuit).equiv(Operator(decomposed_circuit), atol=1e-7)
