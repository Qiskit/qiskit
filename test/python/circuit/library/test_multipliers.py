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

"""Test multiplier circuits."""

import unittest
import numpy as np
from ddt import ddt, data, unpack

from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.circuit.library import (
    RGQFTMultiplier,
    HRSCumulativeMultiplier,
    CDKMRippleCarryAdder,
    DraperQFTAdder,
    VBERippleCarryAdder,
)


@ddt
class TestMultiplier(QiskitTestCase):
    """Test the multiplier circuits."""

    def assertMultiplicationIsCorrect(
        self, num_state_qubits: int, num_result_qubits: int, multiplier: QuantumCircuit
    ):
        """Assert that multiplier correctly implements the product.

        Args:
            num_state_qubits: The number of bits in the numbers that are multiplied.
            num_result_qubits: The number of qubits to limit the output to with modulo.
            multiplier: The circuit performing the multiplication of two numbers with
                ``num_state_qubits`` bits.
        """
        circuit = QuantumCircuit(*multiplier.qregs)

        # create equal superposition
        circuit.h(range(2 * num_state_qubits))

        # apply multiplier circuit
        circuit.compose(multiplier, inplace=True)

        # obtain the statevector and the probabilities, we don't trace out the ancilla qubits
        # as we verify that all ancilla qubits have been uncomputed to state 0 again
        statevector = Statevector(circuit)
        probabilities = statevector.probabilities()
        pad = "0" * circuit.num_ancillas  # state of the ancillas

        # compute the expected results
        expectations = np.zeros_like(probabilities)
        num_bits_product = num_state_qubits * 2
        # iterate over all possible inputs
        for x in range(2 ** num_state_qubits):
            for y in range(2 ** num_state_qubits):
                # compute the product
                product = x * y % (2 ** num_result_qubits)
                # compute correct index in statevector
                bin_x = bin(x)[2:].zfill(num_state_qubits)
                bin_y = bin(y)[2:].zfill(num_state_qubits)
                bin_res = bin(product)[2:].zfill(num_bits_product)
                bin_index = pad + bin_res + bin_y + bin_x
                index = int(bin_index, 2)
                expectations[index] += 1 / 2 ** (2 * num_state_qubits)
        np.testing.assert_array_almost_equal(expectations, probabilities)

    @data(
        (3, RGQFTMultiplier),
        (3, RGQFTMultiplier, 5),
        (3, RGQFTMultiplier, 4),
        (3, RGQFTMultiplier, 3),
        (3, HRSCumulativeMultiplier),
        (3, HRSCumulativeMultiplier, 5),
        (3, HRSCumulativeMultiplier, 4),
        (3, HRSCumulativeMultiplier, 3),
        (3, HRSCumulativeMultiplier, None, CDKMRippleCarryAdder),
        (3, HRSCumulativeMultiplier, None, DraperQFTAdder),
        (3, HRSCumulativeMultiplier, None, VBERippleCarryAdder),
    )
    @unpack
    def test_multiplication(self, num_state_qubits, multiplier, num_result_qubits=None, adder=None):
        """Test multiplication for all implemented multipliers."""
        if num_result_qubits is None:
            num_result_qubits = 2 * num_state_qubits
        if adder is not None:
            adder = adder(num_state_qubits, kind="half")
            multiplier = multiplier(num_state_qubits, num_result_qubits, adder=adder)
        else:
            multiplier = multiplier(num_state_qubits, num_result_qubits)
        self.assertMultiplicationIsCorrect(num_state_qubits, num_result_qubits, multiplier)

    @data(
        (RGQFTMultiplier, -1),
        (HRSCumulativeMultiplier, -1),
        (RGQFTMultiplier, 0, 0),
        (HRSCumulativeMultiplier, 0, 0),
        (RGQFTMultiplier, 0, 1),
        (HRSCumulativeMultiplier, 0, 1),
        (RGQFTMultiplier, 1, 0),
        (HRSCumulativeMultiplier, 1, 0),
        (RGQFTMultiplier, 3, 2),
        (HRSCumulativeMultiplier, 3, 2),
        (RGQFTMultiplier, 3, 7),
        (HRSCumulativeMultiplier, 3, 7),
    )
    @unpack
    def test_raises_on_wrong_num_bits(self, multiplier, num_state_qubits, num_result_qubits=None):
        """Test an error is raised for a bad number of state or result qubits."""
        with self.assertRaises(ValueError):
            _ = multiplier(num_state_qubits, num_result_qubits)

    def test_modular_cumulative_multiplier_custom_adder(self):
        """Test an error is raised when a custom adder is used with modular cumulative multiplier."""
        with self.assertRaises(NotImplementedError):
            _ = HRSCumulativeMultiplier(3, 3, adder=VBERippleCarryAdder(3))


if __name__ == "__main__":
    unittest.main()
