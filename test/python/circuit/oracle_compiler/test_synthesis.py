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
from qiskit import QuantumCircuit
from . import examples


class TestSynthesis(unittest.TestCase):
    """Tests LogicNetwork.synth method."""
    def test_grover_oracle(self):
        network = compile_oracle(examples.grover_oracle)
        quantum_circuit = network.synth()
        self.assertIsInstance(quantum_circuit, QuantumCircuit)

    def test_grover_oracle_arg_regs(self):
        network = compile_oracle(examples.grover_oracle)
        quantum_circuit = network.synth(arg_regs=True)
        self.assertIsInstance(quantum_circuit, QuantumCircuit)


if __name__ == '__main__':
    unittest.main()
