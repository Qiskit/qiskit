# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""Tests for quantum synthesis methods."""

import unittest

from qiskit.circuit import QuantumCircuit
from qiskit import execute
from qiskit.quantum_info.synthesis import two_qubit_kak, euler_angles_1q
from qiskit.quantum_info.operators import Pauli, Operator
from qiskit.quantum_info.random import random_unitary
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.providers.basicaer import UnitarySimulatorPy
from qiskit.test import QiskitTestCase


class TestSynthesis(QiskitTestCase):
    """Test synthesis methods."""

    def test_one_qubit_euler_angles(self):
        """Verify euler_angles_1q produces correct Euler angles for
        a single-qubit unitary.
        """
        for _ in range(100):
            unitary = random_unitary(2)
            with self.subTest(unitary=unitary):
                angles = euler_angles_1q(unitary.data)
                decomp_circuit = QuantumCircuit(1)
                decomp_circuit.u3(*angles, 0)
                result = execute(decomp_circuit, UnitarySimulatorPy()).result()
                decomp_unitary = Operator(result.get_unitary())
                equal_up_to_phase = matrix_equal(
                    unitary.data,
                    decomp_unitary.data,
                    ignore_phase=True,
                    atol=1e-7)
                self.assertTrue(equal_up_to_phase)

    def test_two_qubit_kak(self):
        """Verify KAK decomposition for random Haar 4x4 unitaries.
        """
        for _ in range(100):
            unitary = random_unitary(4)
            with self.subTest(unitary=unitary):
                decomp_circuit = two_qubit_kak(unitary)
                result = execute(decomp_circuit, UnitarySimulatorPy()).result()
                decomp_unitary = Operator(result.get_unitary())
                equal_up_to_phase = matrix_equal(
                    unitary.data,
                    decomp_unitary.data,
                    ignore_phase=True,
                    atol=1e-7)
                self.assertTrue(equal_up_to_phase)

    def test_two_qubit_kak_from_paulis(self):
        """Verify decomposing Paulis with KAK
        """
        pauli_xz = Pauli(label='XZ')
        unitary = Operator(pauli_xz)
        decomp_circuit = two_qubit_kak(unitary)
        result = execute(decomp_circuit, UnitarySimulatorPy()).result()
        decomp_unitary = Operator(result.get_unitary())
        equal_up_to_phase = matrix_equal(
            unitary.data, decomp_unitary.data, ignore_phase=True, atol=1e-7)
        self.assertTrue(equal_up_to_phase)


if __name__ == '__main__':
    unittest.main()
