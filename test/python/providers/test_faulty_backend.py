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

"""Testing a Faulty Backend (1Q)."""

from qiskit.test.mock import FakeOurenseFaultyQ1
from qiskit.test.mock import FakeOurenseFaultyCX13
from qiskit.test import QiskitTestCase


class FaultyQubitBackendTestCase(QiskitTestCase):
    """Test usability methods of backend.properties() with FakeOurenseFaultyQ1,
    which is like FakeOurense but with a faulty 1Q"""

    backend = FakeOurenseFaultyQ1()

    def test_operational_false(self):
        """Test operation status of the qubit. Q1 is non-operational """
        self.assertFalse(self.backend.properties().is_qubit_operational(1))

    def test_faulty_qubits(self):
        """Test faulty_qubits method. """
        self.assertEqual(self.backend.faulty_qubits(), [1])


class FaultyGateBackendTestCase(QiskitTestCase):
    backend = FakeOurenseFaultyCX13()

    def test_operational_gate(self):
        self.assertFalse(self.backend.properties().is_gate_operational('cx', [1, 3]))

    def test_faulty_gates(self):
        """Test faulty_gates method. """
        gates = self.backend.faulty_gates()
        self.assertEqual(len(gates), 1)
        self.assertEqual(gates[0].gate, 'cx')
        self.assertEqual(gates[0].qubits, [1, 3])
        self.assertEqual(gates[0].name, 'cx1_3')
