# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test library of Pauli feature map circuits."""

import unittest

import numpy as np

from ddt import ddt, data, unpack

from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap, HGate
from qiskit.quantum_info import Operator


@ddt
class TestDataPreparation(QiskitTestCase):
    """Test the data encoding circuits."""

    def test_pauli_empty(self):
        """Test instantiating an empty Pauli expansion."""
        encoding = PauliFeatureMap()

        with self.subTest(msg="equal to empty circuit"):
            self.assertTrue(Operator(encoding).equiv(QuantumCircuit()))

        with self.subTest(msg="rotation blocks is H gate"):
            self.assertEqual(len(encoding.rotation_blocks), 1)
            self.assertIsInstance(encoding.rotation_blocks[0].data[0][0], HGate)

    @data((2, 3, ["X", "YY"]), (5, 2, ["ZZZXZ", "XZ"]))
    @unpack
    def test_num_parameters(self, num_qubits, reps, pauli_strings):
        """Test the number of parameters equals the number of qubits, independent of reps."""
        encoding = PauliFeatureMap(num_qubits, paulis=pauli_strings, reps=reps)
        self.assertEqual(encoding.num_parameters, num_qubits)
        self.assertEqual(encoding.num_parameters_settable, num_qubits)

    def test_pauli_evolution(self):
        """Test the generation of Pauli blocks."""
        encoding = PauliFeatureMap()
        time = 1.4
        with self.subTest(pauli_string="ZZ"):
            evo = QuantumCircuit(2)
            evo.cx(0, 1)
            evo.p(2 * time, 1)
            evo.cx(0, 1)

            pauli = encoding.pauli_evolution("ZZ", time)
            self.assertTrue(Operator(pauli).equiv(evo))

        with self.subTest(pauli_string="XYZ"):
            evo = QuantumCircuit(3)
            # X on the most-significant, bottom qubit, Z on the top
            evo.h(2)
            evo.rx(np.pi / 2, 1)
            evo.cx(0, 1)
            evo.cx(1, 2)
            evo.p(2 * time, 2)
            evo.cx(1, 2)
            evo.cx(0, 1)
            evo.rx(-np.pi / 2, 1)
            evo.h(2)

            pauli = encoding.pauli_evolution("XYZ", time)
            self.assertTrue(Operator(pauli).equiv(evo))

        with self.subTest(pauli_string="I"):
            evo = QuantumCircuit(1)
            pauli = encoding.pauli_evolution("I", time)
            self.assertTrue(Operator(pauli).equiv(evo))

    def test_first_order_circuit(self):
        """Test a first order expansion circuit."""
        times = [0.2, 1, np.pi, -1.2]
        encoding = ZFeatureMap(4, reps=3).assign_parameters(times)

        ref = QuantumCircuit(4)
        for _ in range(3):
            ref.h([0, 1, 2, 3])
            for i in range(4):
                ref.p(2 * times[i], i)

        self.assertTrue(Operator(encoding).equiv(ref))

    def test_second_order_circuit(self):
        """Test a second order expansion circuit."""
        times = [0.2, 1, np.pi]
        encoding = ZZFeatureMap(3, reps=2).assign_parameters(times)

        def zz_evolution(circuit, qubit1, qubit2):
            time = (np.pi - times[qubit1]) * (np.pi - times[qubit2])
            circuit.cx(qubit1, qubit2)
            circuit.p(2 * time, qubit2)
            circuit.cx(qubit1, qubit2)

        ref = QuantumCircuit(3)
        for _ in range(2):
            ref.h([0, 1, 2])
            for i in range(3):
                ref.p(2 * times[i], i)
            zz_evolution(ref, 0, 1)
            zz_evolution(ref, 0, 2)
            zz_evolution(ref, 1, 2)

        self.assertTrue(Operator(encoding).equiv(ref))

    def test_pauli_alpha(self):
        """Test  Pauli rotation factor (getter, setter)."""
        encoding = PauliFeatureMap()
        self.assertEqual(encoding.alpha, 2.0)
        encoding.alpha = 1.4
        self.assertEqual(encoding.alpha, 1.4)

    def test_zzfeaturemap_raises_if_too_small(self):
        """Test the ``ZZFeatureMap`` raises an error if the number of qubits is smaller than 2."""
        with self.assertRaises(ValueError):
            _ = ZZFeatureMap(1)


if __name__ == "__main__":
    unittest.main()
