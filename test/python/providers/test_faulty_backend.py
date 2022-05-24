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

"""Testing a Faulty Ourense Backend."""

from qiskit.test import QiskitTestCase
from .faulty_backends import (
    FakeOurenseFaultyCX01CX10,
    FakeOurenseFaultyQ1,
    FakeOurenseFaultyCX13CX31,
)


class FaultyQubitBackendTestCase(QiskitTestCase):
    """Test operational-related methods of backend.properties() with FakeOurenseFaultyQ1,
    which is like FakeOurense but with a faulty 1Q"""

    backend = FakeOurenseFaultyQ1()

    def test_operational_false(self):
        """Test operation status of the qubit. Q1 is non-operational"""
        self.assertFalse(self.backend.properties().is_qubit_operational(1))

    def test_faulty_qubits(self):
        """Test faulty_qubits method."""
        self.assertEqual(self.backend.properties().faulty_qubits(), [1])


class FaultyGate13BackendTestCase(QiskitTestCase):
    """Test operational-related methods of backend.properties() with FakeOurenseFaultyCX13CX31,
    which is like FakeOurense but with a faulty CX(Q1, Q3) and symmetric."""

    backend = FakeOurenseFaultyCX13CX31()

    def test_operational_gate(self):
        """Test is_gate_operational method."""
        self.assertFalse(self.backend.properties().is_gate_operational("cx", [1, 3]))
        self.assertFalse(self.backend.properties().is_gate_operational("cx", [3, 1]))

    def test_faulty_gates(self):
        """Test faulty_gates method."""
        gates = self.backend.properties().faulty_gates()
        self.assertEqual(len(gates), 2)
        self.assertEqual([gate.gate for gate in gates], ["cx", "cx"])
        self.assertEqual(sorted(gate.qubits for gate in gates), [[1, 3], [3, 1]])


class FaultyGate01BackendTestCase(QiskitTestCase):
    """Test operational-related methods of backend.properties() with FakeOurenseFaultyCX13CX31,
    which is like FakeOurense but with a faulty CX(Q1, Q3) and symmetric."""

    backend = FakeOurenseFaultyCX01CX10()

    def test_operational_gate(self):
        """Test is_gate_operational method."""
        self.assertFalse(self.backend.properties().is_gate_operational("cx", [0, 1]))
        self.assertFalse(self.backend.properties().is_gate_operational("cx", [1, 0]))

    def test_faulty_gates(self):
        """Test faulty_gates method."""
        gates = self.backend.properties().faulty_gates()
        self.assertEqual(len(gates), 2)
        self.assertEqual([gate.gate for gate in gates], ["cx", "cx"])
        self.assertEqual(sorted(gate.qubits for gate in gates), [[0, 1], [1, 0]])
