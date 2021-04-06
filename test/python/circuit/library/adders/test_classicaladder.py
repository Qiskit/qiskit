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

"""Test adder circuits."""

import unittest
import numpy as np
from ddt import ddt, data, unpack

from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.circuit.library import ClassicalAdder


@ddt
class TestAdder(QiskitTestCase):
    """Test the adder circuits."""

    def assertAdditionIsCorrect(self, num_state_qubits, adder, inplace):
        """Assert that adder correctly implements the summation w.r.t. its set weights."""
        circuit = QuantumCircuit(*adder.qregs)
        # create equal superposition
        circuit.h(range(2 * num_state_qubits))
        # apply adder circuit
        circuit.compose(adder, inplace=True)
        # obtain the statevector
        statevector = Statevector(circuit)
        # trace out the ancillas if necessary
        if circuit.num_ancillas > 0:
            ancillas = list(range(circuit.num_qubits - circuit.num_ancillas, circuit.num_qubits))
            probabilities = np.diagonal(partial_trace(statevector, ancillas))
        else:
            probabilities = np.abs(statevector) ** 2
        # compute the expected results
        expectations = np.zeros_like(probabilities)
        num_bits_sum = num_state_qubits + 1
        # iterate over all possible inputs
        for x in range(2 ** num_state_qubits):
            for y in range(2 ** num_state_qubits):
                # compute the sum
                addition = x + y
                # compute correct index in statevector
                bin_x = bin(x)[2:].zfill(num_state_qubits)
                bin_y = bin(y)[2:].zfill(num_state_qubits)
                bin_res = bin(addition)[2:].zfill(num_bits_sum)
                bin_index = bin_res + bin_x if inplace else bin_res + bin_y + bin_x
                index = int(bin_index, 2)
                expectations[index] += 1 / 2 ** (2 * num_state_qubits)
        np.testing.assert_array_almost_equal(expectations, probabilities)

    @data(
        (1, ClassicalAdder, True),
        (2, ClassicalAdder, True),
        (5, ClassicalAdder, True)
        # other adders to be added here
    )
    @unpack
    def test_summation(self, num_state_qubits, adder, inplace):
        """Test summation for all implemented adders."""
        adder = adder(num_state_qubits)
        self.assertAdditionIsCorrect(num_state_qubits, adder, inplace)

    @data(
        ClassicalAdder
        # other adders to be added here
    )
    def test_raises_on_wrong_num_bits(self, adder):
        """Test an error is raised for a bad number of qubits."""
        with self.assertRaises(ValueError):
            _ = adder(-1)


if __name__ == '__main__':
    unittest.main()