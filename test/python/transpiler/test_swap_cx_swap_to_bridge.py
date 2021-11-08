# This code is part of Qiskit.
#
# (C) Copyright IBM 2017
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests for the SwapCXSwapToBridge transpiler pass.
"""

import unittest

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.classicalregister import ClassicalRegister
from qiskit.circuit.library import BridgeGate
from qiskit.circuit.library.standard_gates import CXGate, SwapGate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes.optimization import SwapCXSwapToBridge
from qiskit.test import QiskitTestCase


class TestSwapCXSwapToBridge(QiskitTestCase):
    """
    Tests to verify that blocks of Swap-CX-Swap are found and replaced correctly.
    """

    def test_one_occurence(self):
        """Test with input circuit:
        q_0: ─X───────X─
              │       │
        q_1: ─X───■───X─
                ┌─┴─┐
        q_2: ───┤ X ├───
                └───┘
        """
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.cx(1, 2)
        qc.swap(0, 1)

        pm = PassManager(SwapCXSwapToBridge())
        circuit = pm.run(qc)

        expected = QuantumCircuit(3)
        expected.append(BridgeGate(3), [0, 1, 2])
        self.assertEqual(circuit, expected)

    def test_one_occurence_cx_reversed_direction(self):
        """Test with input circuit:
        q_0: ─X───────X─
              │ ┌───┐ │
        q_1: ─X─┤ X ├─X─
                └─┬─┘
        q_2: ─────■─────
        """
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.cx(2, 1)
        qc.swap(0, 1)

        pm = PassManager(SwapCXSwapToBridge())
        circuit = pm.run(qc)

        expected = QuantumCircuit(3)
        expected.append(BridgeGate(3), [2, 1, 0])
        self.assertEqual(circuit, expected)

    def test_cx_with_single_qubit_adjacent_gates(self):
        """Test with input circuit:
        q_0: ──X─────────X──
               │         │
        q_1: ──X────■────X──
             ┌───┐┌─┴─┐┌───┐
        q_2: ┤ Z ├┤ X ├┤ H ├
             └───┘└───┘└───┘
        """
        qc = QuantumCircuit(3)
        qc.z(2)
        qc.swap(0, 1)
        qc.cx(1, 2)
        qc.h(2)
        qc.swap(0, 1)

        pm = PassManager(SwapCXSwapToBridge())
        circuit = pm.run(qc)

        expected = QuantumCircuit(3)
        expected.z(2)
        expected.append(BridgeGate(3), [0, 1, 2])
        expected.h(2)
        self.assertEqual(circuit, expected)

    def test_multiple_occurences(self):
        """Test with input circuit:
        q_0: ─X───────X──────■─────
              │       │    ┌─┴─┐
        q_1: ─X───■───X──X─┤ X ├─X─
                ┌─┴─┐    │ └───┘ │
        q_2: ───┤ X ├────X───────X─
                └───┘
        """
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.cx(1, 2)
        qc.swap(0, 1)
        qc.swap(1, 2)
        qc.cx(0, 1)
        qc.swap(1, 2)

        pm = PassManager(SwapCXSwapToBridge())
        circuit = pm.run(qc)

        expected = QuantumCircuit(3)
        expected.append(BridgeGate(3), [0, 1, 2])
        expected.append(BridgeGate(3), [0, 1, 2])
        self.assertEqual(circuit, expected)

    def test_with_more_swap_pairs(self):
        """Test with input circuit:
        q_0: ─X─────────────X─
              │             │
        q_1: ─X──X───────X──X─
                 │       │
        q_2: ────X───■───X────
                   ┌─┴─┐
        q_3: ──────┤ X ├──────
                   └───┘
        """
        qc = QuantumCircuit(4)
        qc.swap(0, 1)
        qc.swap(1, 2)
        qc.cx(2, 3)
        qc.swap(1, 2)
        qc.swap(0, 1)

        pm = PassManager(SwapCXSwapToBridge())
        circuit = pm.run(qc)

        expected = QuantumCircuit(4)
        expected.append(BridgeGate(4), [0, 1, 2, 3])
        self.assertEqual(circuit, expected)

    def test_with_swap_pairs_in_both_control_and_target_qubits(self):
        """Test with input circuit:
        q_0: ─X───────X─
              │       │
        q_1: ─X───■───X─
                ┌─┴─┐
        q_2: ─X─┤ X ├─X─
              │ └───┘ │
        q_3: ─X───────X─
        """
        qc = QuantumCircuit(4)
        qc.swap(0, 1)
        qc.swap(2, 3)
        qc.cx(1, 2)
        qc.swap(0, 1)
        qc.swap(2, 3)

        pm = PassManager(SwapCXSwapToBridge())
        circuit = pm.run(qc)

        expected = QuantumCircuit(4)
        expected.append(BridgeGate(4), [0, 1, 2, 3])
        self.assertEqual(circuit, expected)

    def test_swap_is_not_collected_twice(self):
        """Test with input circuit:
        q_0: ─X───────X───────X─
              │       │       │
        q_1: ─X───■───X───■───X─
                ┌─┴─┐   ┌─┴─┐
        q_2: ───┤ X ├───┤ X ├───
                └───┘   └───┘
        """
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.cx(1, 2)
        qc.swap(0, 1)
        qc.cx(1, 2)
        qc.swap(0, 1)

        pm = PassManager(SwapCXSwapToBridge())
        circuit = pm.run(qc)

        expected = QuantumCircuit(3)
        expected.append(BridgeGate(3), [0, 1, 2])
        expected.cx(1, 2)
        expected.swap(0, 1)
        self.assertEqual(circuit, expected)

    def test_not_replace_if_gate_between_swaps(self):
        """Test with input circuit:
                ┌───┐
        q_0: ─X─┤ X ├─X─
              │ └───┘ │
        q_1: ─X───■───X─
                ┌─┴─┐
        q_2: ───┤ X ├───
                └───┘
        """
        qc = QuantumCircuit(3)
        qc.swap(0, 1)
        qc.x(0)
        qc.cx(1, 2)
        qc.swap(0, 1)

        pm = PassManager(SwapCXSwapToBridge())
        circuit = pm.run(qc)
        self.assertEqual(circuit, qc)

    def test_with_classical_bit_in_cx(self):
        """Test with input circuit:
        q_0: ─X─────────X─
              │         │
        q_1: ─X────■────X─
                 ┌─┴─┐
        q_2: ────┤ X ├────
                 └─╥─┘
                ┌──╨──┐
        c: 1/═══╡ 0x1 ╞═══
                └─────┘
        """
        qr = QuantumRegister(3)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        qc.swap(0, 1)
        qc.append(CXGate().c_if(cr, 1), (1, 2))
        qc.swap(0, 1)

        pm = PassManager(SwapCXSwapToBridge())
        circuit = pm.run(qc)

        expected = QuantumCircuit(qr, cr)
        expected.append(BridgeGate(3).c_if(cr, 1), [0, 1, 2])
        self.assertEqual(circuit, expected)

    def test_not_replace_with_classical_bit_in_swap(self):
        """Test with input circuit:
        q_0: ───X─────────X─
                │         │
        q_1: ───X─────■───X─
                ║   ┌─┴─┐
        q_2: ───╫───┤ X ├───
             ┌──╨──┐└───┘
        c: 1/╡ 0x1 ╞════════
             └─────┘
        """
        qr = QuantumRegister(3)
        cr = ClassicalRegister(1)
        qc = QuantumCircuit(qr, cr)
        qc.append(SwapGate().c_if(cr, 1), (0, 1))
        qc.cx(1, 2)
        qc.swap(0, 1)

        pm = PassManager(SwapCXSwapToBridge())
        circuit = pm.run(qc)
        self.assertEqual(circuit, qc)


if __name__ == "__main__":
    unittest.main()
