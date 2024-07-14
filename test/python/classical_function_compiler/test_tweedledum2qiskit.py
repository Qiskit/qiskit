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

"""Tests LogicNetwork.Tweedledum2Qiskit converter."""

import unittest

from qiskit.utils.optionals import HAS_TWEEDLEDUM
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import XGate
from test import QiskitTestCase  # pylint: disable=wrong-import-order

if HAS_TWEEDLEDUM:
    # pylint: disable=import-error
    from qiskit.circuit.classicalfunction.utils import tweedledum2qiskit

    from tweedledum.ir import Circuit
    from tweedledum.operators import X


@unittest.skipUnless(HAS_TWEEDLEDUM, "Tweedledum is required for these tests.")
class TestTweedledum2Qiskit(QiskitTestCase):
    # pylint: disable=possibly-used-before-assignment
    """Tests qiskit.transpiler.classicalfunction.utils.tweedledum2qiskit function."""

    def test_x(self):
        """Single uncontrolled X"""
        tweedledum_circuit = Circuit()
        tweedledum_circuit.apply_operator(X(), [tweedledum_circuit.create_qubit()])

        circuit = tweedledum2qiskit(tweedledum_circuit)

        expected = QuantumCircuit(1)
        expected.x(0)

        self.assertEqual(circuit, expected)

    def test_cx_0_1(self):
        """CX(0, 1)"""
        tweedledum_circuit = Circuit()
        qubits = []
        qubits.append(tweedledum_circuit.create_qubit())
        qubits.append(tweedledum_circuit.create_qubit())
        tweedledum_circuit.apply_operator(X(), [qubits[0], qubits[1]])

        circuit = tweedledum2qiskit(tweedledum_circuit)

        expected = QuantumCircuit(2)
        expected.append(XGate().control(1, ctrl_state="1"), [0, 1])

        self.assertEqual(circuit, expected)

    def test_cx_1_0(self):
        """CX(1, 0)"""
        tweedledum_circuit = Circuit()
        qubits = []
        qubits.append(tweedledum_circuit.create_qubit())
        qubits.append(tweedledum_circuit.create_qubit())
        tweedledum_circuit.apply_operator(X(), [qubits[1], qubits[0]])

        circuit = tweedledum2qiskit(tweedledum_circuit)

        expected = QuantumCircuit(2)
        expected.append(XGate().control(1, ctrl_state="1"), [1, 0])

        self.assertEqual(expected, circuit)

    def test_cx_qreg(self):
        """CX(0, 1) with qregs parameter"""
        tweedledum_circuit = Circuit()
        qubits = []
        qubits.append(tweedledum_circuit.create_qubit())
        qubits.append(tweedledum_circuit.create_qubit())
        tweedledum_circuit.apply_operator(X(), [qubits[1], qubits[0]])

        qr = QuantumRegister(2, "qr")
        circuit = tweedledum2qiskit(tweedledum_circuit, qregs=[qr])

        expected = QuantumCircuit(qr)
        expected.append(XGate().control(1, ctrl_state="1"), [qr[1], qr[0]])

        self.assertEqual(expected, circuit)
