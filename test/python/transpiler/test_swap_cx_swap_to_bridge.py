# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test the SwapCXSwaptoBridge Optimization pass."""

from qiskit.circuit import QuantumRegister, QuantumCircuit
from qiskit.transpiler.passes.optimization import SwapCXSwaptoBridge
from qiskit.test import QiskitTestCase
from qiskit.transpiler import PassManager
from qiskit.circuit.library.generalized_gates import BridgeGate


class TestSwapCXSwaptoBridge(QiskitTestCase):
    """
    Tests to verify that converting the sub-circuit SWAP-CX-SWAP to Bridge
    works correctly.
    """

    def test_basic_circuit(self):
        """
                                       ┌─────────┐
        q_0: ─X───────X─          q_0: ┤0        ├
              │       │                │         │
        q_1: ─X───■───X─     =    q_1: ┤1 bridge ├
                ┌─┴─┐                  │         │
        q_2: ───┤ X ├───          q_2: ┤2        ├
                └───┘                  └─────────┘
        """

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)

        circuit.swap(0, 1)
        circuit.cx(1, 2)
        circuit.swap(0, 1)

        pm = PassManager()
        pm.append([SwapCXSwaptoBridge()])

        output = pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.append(BridgeGate(3), expected.qubits)

        self.assertEqual(output, expected)

    def test_basic_circuit_inverse(self):
        """
                                       ┌─────────┐
        q_0: ─────■─────          q_0: ┤0        ├
                ┌─┴─┐                  │         │
        q_1: ─X─┤ X ├─X─    =     q_1: ┤1 bridge ├
              │ └───┘ │                │         │
        q_2: ─X───────X─          q_2: ┤2        ├
                                       └─────────┘
        """

        qr = QuantumRegister(3, "q")
        circuit = QuantumCircuit(qr)

        circuit.swap(1, 2)
        circuit.cx(0, 1)
        circuit.swap(1, 2)

        pm = PassManager()
        pm.append([SwapCXSwaptoBridge()])

        output = pm.run(circuit)
        expected = QuantumCircuit(qr)
        expected.append(BridgeGate(3), expected.qubits)

        self.assertEqual(output, expected)
