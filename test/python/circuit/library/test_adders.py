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
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import RippleCarryAdder, QFTAdder, PlainAdder


@ddt
class TestAdder(QiskitTestCase):
    """Test the adder circuits."""

    def assertAdditionIsCorrect(self,
                                num_state_qubits: int,
                                adder: QuantumCircuit,
                                inplace: bool,
                                modular: bool):
        """Assert that adder correctly implements the summation.

        Args:
            num_state_qubits: The number of bits in the numbers that are added.
            adder: The circuit performing the addition of two numbers with ``num_state_qubits``
                bits.
            inplace: If True, compare against an inplace addition where the result is written into
                the second register plus carry qubit. If False, assume that the result is written
                into a third register of appropriate size.
            modular: If True, omit the carry qubit to obtain an addition modulo
                ``2^num_state_qubits``.
        """
        circuit = QuantumCircuit(*adder.qregs)

        # create equal superposition
        circuit.h(range(2 * num_state_qubits))

        # apply adder circuit
        circuit.compose(adder, inplace=True)

        # obtain the statevector and the probabilities, we don't trace out the ancilla qubits
        # as we verify that all ancilla qubits have been uncomputed to state 0 again
        statevector = Statevector(circuit)
        probabilities = statevector.probabilities()
        pad = '0' * circuit.num_ancillas  # state of the ancillas

        # compute the expected results
        expectations = np.zeros_like(probabilities)
        num_bits_sum = num_state_qubits + 1
        # iterate over all possible inputs
        for x in range(2 ** num_state_qubits):
            for y in range(2 ** num_state_qubits):
                # compute the sum
                addition = (x + y) % (2 ** num_state_qubits) if modular else x + y
                # compute correct index in statevector
                bin_x = bin(x)[2:].zfill(num_state_qubits)
                bin_y = bin(y)[2:].zfill(num_state_qubits)
                bin_res = bin(addition)[2:].zfill(num_bits_sum)
                bin_index = pad + bin_res + bin_x if inplace else pad + bin_res + bin_y + bin_x
                index = int(bin_index, 2)
                expectations[index] += 1 / 2 ** (2 * num_state_qubits)
        np.testing.assert_array_almost_equal(expectations, probabilities)

    @data(
        (3, RippleCarryAdder, True),
        (5, RippleCarryAdder, True),
        (3, RippleCarryAdder, True, True),
        (5, RippleCarryAdder, True, True),
        (3, QFTAdder, True),
        (5, QFTAdder, True),
        (3, QFTAdder, True, True),
        (5, QFTAdder, True, True),
        (3, PlainAdder, True),
        (5, PlainAdder, True),
        (3, PlainAdder, True, True),
        (5, PlainAdder, True, True),
    )
    @unpack
    def test_summation(self, num_state_qubits, adder, inplace, modular=False):
        """Test summation for all implemented adders."""
        adder = adder(num_state_qubits, modular=modular)
        self.assertAdditionIsCorrect(num_state_qubits, adder, inplace, modular)

    @data(
        RippleCarryAdder,
        QFTAdder,
        PlainAdder
    )
    def test_raises_on_wrong_num_bits(self, adder):
        """Test an error is raised for a bad number of qubits."""
        with self.assertRaises(ValueError):
            _ = adder(-1)


if __name__ == '__main__':
    unittest.main()
