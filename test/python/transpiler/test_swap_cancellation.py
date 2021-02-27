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

from qiskit import QuantumRegister, QuantumCircuit
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import SWAPCancellation
from qiskit.test import QiskitTestCase


class TestSWAPCancellation(QiskitTestCase):
    """Test the SWAPCancellation pass."""

    def test_pass_swap_cancellation_simple(self):
        """Test a simple swap cancellation pass, symmetric wires """
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[1], qr[0])

        pass_manager = PassManager()
        pass_manager.append(SWAPCancellation())
        out_circuit = pass_manager.run(circuit)
        resources_after = out_circuit.count_ops()

        self.assertNotIn('swap', resources_after)

    def test_pass_swap_cancellation_many(self):
        """Test the swap cancellation pass.

        It should cancel consecutive swap pairs on same qubits.
        """
        qr = QuantumRegister(2)
        circuit = QuantumCircuit(qr)
        circuit.h(qr[0])
        circuit.h(qr[0])
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[0], qr[1])
        circuit.swap(qr[1], qr[0])
        circuit.swap(qr[1], qr[0])

        pass_manager = PassManager()
        pass_manager.append(SWAPCancellation())
        out_circuit = pass_manager.run(circuit)
        resources_after = out_circuit.count_ops()

        self.assertNotIn('swap', resources_after)
