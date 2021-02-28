# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests for pass cancelling 2 consecutive SWAPs on the same qubits."""

from qiskit import QuantumCircuit
from qiskit.transpiler.passes import SWAPCancellation
from qiskit.test import QiskitTestCase


class TestSWAPCancellation(QiskitTestCase):
    """Test the SWAPCancellation pass."""

    def test_pass_swap_cancellation_simple(self):
        """Test a simple swap cancellation pass, symmetric wires """
        circuit = QuantumCircuit(2)
        circuit.swap(0, 1)
        circuit.swap(1, 0)

        out_circuit = SWAPCancellation()(circuit)
        resources_after = out_circuit.count_ops()

        self.assertNotIn('swap', resources_after)

    def test_pass_swap_cancellation_many(self):
        """Test the swap cancellation pass.

        It should cancel consecutive swap pairs on same qubits.
        """
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.h(0)
        circuit.swap(0, 1)
        circuit.swap(0, 1)
        circuit.swap(0, 1)
        circuit.swap(0, 1)
        circuit.swap(1, 0)
        circuit.swap(1, 0)

        out_circuit = SWAPCancellation()(circuit)
        resources_after = out_circuit.count_ops()

        self.assertNotIn('swap', resources_after)
