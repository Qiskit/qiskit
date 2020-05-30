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

"""Tests oracle compiler synthesis."""

import unittest

from qiskit.circuit.oracle_compiler import compile_oracle

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.standard_gates import XGate

from . import examples


class TestSynthesis(unittest.TestCase):
    """Tests LogicNetwork.synth method."""

    def test_grover_oracle(self):
        """Synthesis of grover_oracle example"""
        network = compile_oracle(examples.grover_oracle)
        quantum_circuit = network.synth()

        expected = QuantumCircuit(5)
        expected.append(XGate().control(4, ctrl_state='0101'), [0, 1, 2, 3, 4])

        self.assertEqual(quantum_circuit, expected)

    def test_grover_oracle_arg_regs(self):
        """Synthesis of grover_oracle example with arg_regs"""
        network = compile_oracle(examples.grover_oracle)
        quantum_circuit = network.synth(arg_regs=True)

        qr_a = QuantumRegister(1, 'a')
        qr_b = QuantumRegister(1, 'b')
        qr_c = QuantumRegister(1, 'c')
        qr_d = QuantumRegister(1, 'd')
        qr_return = QuantumRegister(1, 'return')
        expected = QuantumCircuit(qr_d, qr_c, qr_b, qr_a, qr_return)
        expected.append(XGate().control(4, ctrl_state='0101'),
                        [qr_d[0], qr_c[0], qr_b[0], qr_a[0], qr_return[0]])

        self.assertEqual(quantum_circuit, expected)


if __name__ == '__main__':
    unittest.main()
