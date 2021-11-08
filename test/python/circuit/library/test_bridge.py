# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test library of Bridge gate."""

import unittest
from ddt import ddt, data

from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import BridgeGate, CXGate
from qiskit.quantum_info import Operator


@ddt
class TestBridgeGate(QiskitTestCase):
    """Test of Bridge gate."""

    def test_threeq_decomposition(self):
        bridge = BridgeGate(3)
        circuit = bridge.definition

        expected = QuantumCircuit(3)
        expected.cx(0, 1)
        expected.cx(1, 2)
        expected.cx(0, 1)
        expected.cx(1, 2)
        self.assertEqual(circuit, expected)

    @data(3, 4, 5)
    def test_equivalence(self, n):
        """Test Bridge on `n` qubits is same as CX_0,{n-1}."""
        circuit = BridgeGate(n)

        expected = QuantumCircuit(n)
        expected.cx(0, n - 1)
        expected = Operator(expected)

        simulated = Operator(circuit)
        self.assertTrue(expected.equiv(simulated))

    @data(3, 4, 5)
    def test_cnots_between_neighbors(self, n):
        """Test Bridge gate decomposition has CNOTs only between neighbors in a line topology."""
        bridge = BridgeGate(n)
        circuit = bridge.definition
        indices = {qb: ix for ix, qb in enumerate(circuit.qubits)}
        for gate, qargs, _ in circuit:
            self.assertTrue(isinstance(gate, CXGate))
            self.assertEqual(indices[qargs[1]] - indices[qargs[0]], 1)

    def test_self_inverse(self):
        """Test Bridge gate is self-inverse."""
        self.assertEqual(BridgeGate(3), BridgeGate(3).inverse())


if __name__ == "__main__":
    unittest.main()
