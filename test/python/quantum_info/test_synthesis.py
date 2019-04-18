# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

"""Tests for quantum synthesis methods."""

import unittest

from qiskit import execute
from qiskit.quantum_info.operators.measures import process_fidelity
from qiskit.quantum_info.synthesis import two_qubit_kak
from qiskit.quantum_info.operators import Unitary, Pauli, Operator
from qiskit.quantum_info.random import random_unitary
from qiskit.providers.basicaer import UnitarySimulatorPy
from qiskit.test import QiskitTestCase


class TestSynthesis(QiskitTestCase):
    """Test synthesis methods."""

    def test_two_qubit_kak(self):
        """Verify KAK decomposition for random Haar 4x4 unitaries.
        """
        for _ in range(100):
            unitary = random_unitary(4)
            with self.subTest(unitary=unitary):
                decomp_circuit = two_qubit_kak(unitary)
                result = execute(decomp_circuit, UnitarySimulatorPy()).result()
                decomp_unitary = Unitary(result.get_unitary())
                self.assertAlmostEqual(
                    process_fidelity(unitary.representation, decomp_unitary.representation),
                    1.0, places=7)

    def test_two_qubit_kak_from_paulis(self):
        """Verify decomposing Paulis with KAK
        """
        pauli_xz = Pauli(label='XZ')
        unitary = Unitary(Operator(pauli_xz).data)
        decomp_circuit = two_qubit_kak(unitary)
        result = execute(decomp_circuit, UnitarySimulatorPy()).result()
        decomp_unitary = Unitary(result.get_unitary())
        self.assertAlmostEqual(decomp_unitary, unitary)


if __name__ == '__main__':
    unittest.main()
