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

"""Tests classicalfunction compiler synthesis."""
import unittest
from qiskit.test import QiskitTestCase


from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import XGate
from qiskit.utils.optionals import HAS_TWEEDLEDUM

if HAS_TWEEDLEDUM:
    from qiskit.circuit.classicalfunction import classical_function as compile_classical_function
    from . import examples


@unittest.skipUnless(HAS_TWEEDLEDUM, "Tweedledum is required for these tests.")
class TestSynthesis(QiskitTestCase):
    """Tests ClassicalFunction.synth method."""

    def test_grover_oracle(self):
        """Synthesis of grover_oracle example"""
        oracle = compile_classical_function(examples.grover_oracle)
        quantum_circuit = oracle.synth()

        expected = QuantumCircuit(5)
        expected.append(XGate().control(4, ctrl_state="1010"), [0, 1, 2, 3, 4])

        self.assertEqual(quantum_circuit.name, "grover_oracle")
        self.assertEqual(quantum_circuit, expected)

    def test_grover_oracle_arg_regs(self):
        """Synthesis of grover_oracle example with arg_regs"""
        oracle = compile_classical_function(examples.grover_oracle)
        quantum_circuit = oracle.synth(registerless=False)

        qr_a = QuantumRegister(1, "a")
        qr_b = QuantumRegister(1, "b")
        qr_c = QuantumRegister(1, "c")
        qr_d = QuantumRegister(1, "d")
        qr_return = QuantumRegister(1, "return")
        expected = QuantumCircuit(qr_d, qr_c, qr_b, qr_a, qr_return)
        expected.append(
            XGate().control(4, ctrl_state="1010"),
            [qr_d[0], qr_c[0], qr_b[0], qr_a[0], qr_return[0]],
        )

        self.assertEqual(quantum_circuit.name, "grover_oracle")
        self.assertEqual(quantum_circuit, expected)
