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

"""Test library of weighted adder circuits."""

import unittest
from collections import defaultdict
from ddt import ddt, data
import numpy as np

from qiskit.test.base import QiskitTestCase
from qiskit import BasicAer, transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import WeightedAdder


@ddt
class TestWeightedAdder(QiskitTestCase):
    """Test the weighted adder circuit."""

    def assertSummationIsCorrect(self, adder):
        """Assert that ``adder`` correctly implements the summation w.r.t. its set weights."""

        circuit = QuantumCircuit(adder.num_qubits)
        circuit.h(list(range(adder.num_state_qubits)))
        circuit.append(adder.to_instruction(), list(range(adder.num_qubits)))

        backend = BasicAer.get_backend("statevector_simulator")
        statevector = backend.run(transpile(circuit, backend)).result().get_statevector()

        probabilities = defaultdict(float)
        for i, statevector_amplitude in enumerate(statevector):
            i = bin(i)[2:].zfill(circuit.num_qubits)[adder.num_ancillas :]
            probabilities[i] += np.real(np.abs(statevector_amplitude) ** 2)

        expectations = defaultdict(float)
        for x in range(2**adder.num_state_qubits):
            bits = np.array(list(bin(x)[2:].zfill(adder.num_state_qubits)), dtype=int)
            summation = bits.dot(adder.weights[::-1])

            entry = bin(summation)[2:].zfill(adder.num_sum_qubits) + bin(x)[2:].zfill(
                adder.num_state_qubits
            )
            expectations[entry] = 1 / 2**adder.num_state_qubits

        for state, probability in probabilities.items():
            self.assertAlmostEqual(probability, expectations[state])

    @data([0], [1, 2, 1], [4], [1, 2, 1, 1, 4])
    def test_summation(self, weights):
        """Test the weighted adder on some examples."""
        adder = WeightedAdder(len(weights), weights)
        self.assertSummationIsCorrect(adder)

    def test_mutability(self):
        """Test the mutability of the weighted adder."""
        adder = WeightedAdder()

        with self.subTest(msg="missing number of state qubits"):
            with self.assertRaises(AttributeError):
                _ = str(adder.draw())

        with self.subTest(msg="default weights"):
            adder.num_state_qubits = 3
            default_weights = 3 * [1]
            self.assertListEqual(adder.weights, default_weights)

        with self.subTest(msg="specify weights"):
            adder.weights = [3, 2, 1]
            self.assertSummationIsCorrect(adder)

        with self.subTest(msg="mismatching number of state qubits and weights"):
            with self.assertRaises(ValueError):
                adder.weights = [0, 1, 2, 3]
                _ = str(adder.draw())

        with self.subTest(msg="change all attributes"):
            adder.num_state_qubits = 4
            adder.weights = [2, 0, 1, 1]
            self.assertSummationIsCorrect(adder)


if __name__ == "__main__":
    unittest.main()
