# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for the DropNegligible transpiler pass."""

import numpy as np

from qiskit.circuit import Gate, Parameter, QuantumCircuit, QuantumRegister
from qiskit.circuit.library import (
    CPhaseGate,
    RXGate,
    RXXGate,
    RYGate,
    RYYGate,
    RZGate,
    RZZGate,
    XXMinusYYGate,
    XXPlusYYGate,
)
from qiskit.quantum_info import Operator
from qiskit.transpiler.passes import DropNegligible

from test import QiskitTestCase  # pylint: disable=wrong-import-order


class TestDropNegligible(QiskitTestCase):
    """Test the DropNegligible pass."""

    def test_drops_negligible_gates(self):
        """Test that negligible gates are dropped."""
        qubits = QuantumRegister(2)
        circuit = QuantumCircuit(qubits)
        a, b = qubits
        circuit.append(CPhaseGate(1e-5), [a, b])
        circuit.append(CPhaseGate(1e-8), [a, b])
        circuit.append(RXGate(1e-5), [a])
        circuit.append(RXGate(1e-8), [a])
        circuit.append(RYGate(1e-5), [a])
        circuit.append(RYGate(1e-8), [a])
        circuit.append(RZGate(1e-5), [a])
        circuit.append(RZGate(1e-8), [a])
        circuit.append(RXXGate(1e-5), [a, b])
        circuit.append(RXXGate(1e-8), [a, b])
        circuit.append(RYYGate(1e-5), [a, b])
        circuit.append(RYYGate(1e-8), [a, b])
        circuit.append(RZZGate(1e-5), [a, b])
        circuit.append(RZZGate(1e-8), [a, b])
        circuit.append(XXPlusYYGate(1e-5, 1e-8), [a, b])
        circuit.append(XXPlusYYGate(1e-8, 1e-8), [a, b])
        circuit.append(XXMinusYYGate(1e-5, 1e-8), [a, b])
        circuit.append(XXMinusYYGate(1e-8, 1e-8), [a, b])
        transpiled = DropNegligible()(circuit)
        self.assertEqual(circuit.count_ops()["cp"], 2)
        self.assertEqual(transpiled.count_ops()["cp"], 1)
        self.assertEqual(circuit.count_ops()["rx"], 2)
        self.assertEqual(transpiled.count_ops()["rx"], 1)
        self.assertEqual(circuit.count_ops()["ry"], 2)
        self.assertEqual(transpiled.count_ops()["ry"], 1)
        self.assertEqual(circuit.count_ops()["rz"], 2)
        self.assertEqual(transpiled.count_ops()["rz"], 1)
        self.assertEqual(circuit.count_ops()["rxx"], 2)
        self.assertEqual(transpiled.count_ops()["rxx"], 1)
        self.assertEqual(circuit.count_ops()["ryy"], 2)
        self.assertEqual(transpiled.count_ops()["ryy"], 1)
        self.assertEqual(circuit.count_ops()["rzz"], 2)
        self.assertEqual(transpiled.count_ops()["rzz"], 1)
        self.assertEqual(circuit.count_ops()["xx_plus_yy"], 2)
        self.assertEqual(transpiled.count_ops()["xx_plus_yy"], 1)
        self.assertEqual(circuit.count_ops()["xx_minus_yy"], 2)
        self.assertEqual(transpiled.count_ops()["xx_minus_yy"], 1)
        np.testing.assert_allclose(
            np.array(Operator(circuit)), np.array(Operator(transpiled)), atol=1e-7
        )

    def test_handles_parameters(self):
        """Test that gates with parameters are ignored gracefully."""
        qubits = QuantumRegister(2)
        circuit = QuantumCircuit(qubits)
        a, b = qubits
        theta = Parameter("theta")
        circuit.append(CPhaseGate(theta), [a, b])
        circuit.append(CPhaseGate(1e-5), [a, b])
        circuit.append(CPhaseGate(1e-8), [a, b])
        transpiled = DropNegligible()(circuit)
        self.assertEqual(circuit.count_ops()["cp"], 3)
        self.assertEqual(transpiled.count_ops()["cp"], 2)

    def test_handles_number_types(self):
        """Test that gates with different types of numbers are handled correctly."""
        qubits = QuantumRegister(2)
        circuit = QuantumCircuit(qubits)
        a, b = qubits
        circuit.append(CPhaseGate(np.float32(1e-6)), [a, b])
        circuit.append(CPhaseGate(1e-3), [a, b])
        circuit.append(CPhaseGate(1e-8), [a, b])
        transpiled = DropNegligible(atol=1e-5)(circuit)
        self.assertEqual(circuit.count_ops()["cp"], 3)
        self.assertEqual(transpiled.count_ops()["cp"], 1)

    def test_additional_gate_types(self):
        """Test passing additional gate types."""

        class TestGateA(Gate):
            pass

        class TestGateB(Gate):
            pass

        qubits = QuantumRegister(2)
        circuit = QuantumCircuit(qubits)
        a, b = qubits
        circuit.append(CPhaseGate(1e-5), [a, b])
        circuit.append(CPhaseGate(1e-8), [a, b])
        circuit.append(TestGateA("test_gate_a", 1, [1e-5, 1e-5]), [a])
        circuit.append(TestGateA("test_gate_a", 1, [1e-8, 1e-8]), [a])
        circuit.append(TestGateB("test_gate_b", 1, [1e-5, 1e-5]), [a])
        circuit.append(TestGateB("test_gate_b", 1, [1e-8, 1e-8]), [a])
        transpiled = DropNegligible(additional_gate_types=[TestGateA])(circuit)
        self.assertEqual(circuit.count_ops()["cp"], 2)
        self.assertEqual(transpiled.count_ops()["cp"], 1)
        self.assertEqual(circuit.count_ops()["test_gate_a"], 2)
        self.assertEqual(transpiled.count_ops()["test_gate_a"], 1)
        self.assertEqual(circuit.count_ops()["test_gate_b"], 2)
        self.assertEqual(transpiled.count_ops()["test_gate_b"], 2)
