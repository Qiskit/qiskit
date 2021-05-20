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
from qiskit.circuit.library import CDKMRippleCarryAdder, DraperQFTAdder, VBERippleCarryAdder


@ddt
class TestAdder(QiskitTestCase):
    """Test the adder circuits."""

    def assertAdditionIsCorrect(
        self, num_state_qubits: int, adder: QuantumCircuit, inplace: bool, kind: str
    ):
        """Assert that adder correctly implements the summation.

        This test prepares a equal superposition state in both input registers, then performs
        the addition on the superposition and checks that the output state is the expected
        superposition of all possible additions.

        Args:
            num_state_qubits: The number of bits in the numbers that are added.
            adder: The circuit performing the addition of two numbers with ``num_state_qubits``
                bits.
            inplace: If True, compare against an inplace addition where the result is written into
                the second register plus carry qubit. If False, assume that the result is written
                into a third register of appropriate size.
            kind: TODO
        """
        circuit = QuantumCircuit(*adder.qregs)

        # create equal superposition
        if kind == "full":
            num_superpos_qubits = 2 * num_state_qubits + 1
        else:
            num_superpos_qubits = 2 * num_state_qubits
        circuit.h(range(num_superpos_qubits))

        # apply adder circuit
        circuit.compose(adder, inplace=True)

        # obtain the statevector and the probabilities, we don't trace out the ancilla qubits
        # as we verify that all ancilla qubits have been uncomputed to state 0 again
        statevector = Statevector(circuit)
        probabilities = statevector.probabilities()
        pad = "0" * circuit.num_ancillas  # state of the ancillas

        # compute the expected results
        expectations = np.zeros_like(probabilities)
        num_bits_sum = num_state_qubits + 1
        # iterate over all possible inputs
        for x in range(2 ** num_state_qubits):
            for y in range(2 ** num_state_qubits):
                # compute the sum
                if kind == "full":
                    additions = [x + y, 1 + x + y]
                elif kind == "half":
                    additions = [x + y]
                else:
                    additions = [(x + y) % (2 ** num_state_qubits)]

                # compute correct index in statevector
                bin_x = bin(x)[2:].zfill(num_state_qubits)
                bin_y = bin(y)[2:].zfill(num_state_qubits)

                for i, addition in enumerate(additions):
                    bin_res = bin(addition)[2:].zfill(num_bits_sum)
                    if kind == "full":
                        cin = str(i)
                        bin_index = (
                            pad + bin_res + bin_x + cin
                            if inplace
                            else pad + bin_res + bin_y + bin_x + cin
                        )
                    else:
                        bin_index = (
                            pad + bin_res + bin_x if inplace else pad + bin_res + bin_y + bin_x
                        )

                    index = int(bin_index, 2)
                    expectations[index] += 1 / 2 ** num_superpos_qubits

        np.testing.assert_array_almost_equal(expectations, probabilities)

    @data(
        (3, CDKMRippleCarryAdder, True),
        (5, CDKMRippleCarryAdder, True),
        (3, CDKMRippleCarryAdder, True, "fixed"),
        (5, CDKMRippleCarryAdder, True, "fixed"),
        (1, CDKMRippleCarryAdder, True, "full"),
        (3, CDKMRippleCarryAdder, True, "full"),
        (5, CDKMRippleCarryAdder, True, "full"),
        (3, DraperQFTAdder, True),
        (5, DraperQFTAdder, True),
        (3, DraperQFTAdder, True, "fixed"),
        (5, DraperQFTAdder, True, "fixed"),
        (1, VBERippleCarryAdder, True, "full"),
        (3, VBERippleCarryAdder, True, "full"),
        (5, VBERippleCarryAdder, True, "full"),
        (1, VBERippleCarryAdder, True),
        (2, VBERippleCarryAdder, True),
        (5, VBERippleCarryAdder, True),
        (1, VBERippleCarryAdder, True, "fixed"),
        (2, VBERippleCarryAdder, True, "fixed"),
        (4, VBERippleCarryAdder, True, "fixed"),
    )
    @unpack
    def test_summation(self, num_state_qubits, adder, inplace, kind="half"):
        """Test summation for all implemented adders."""
        adder = adder(num_state_qubits, kind=kind)
        self.assertAdditionIsCorrect(num_state_qubits, adder, inplace, kind)

    @data(CDKMRippleCarryAdder, DraperQFTAdder, VBERippleCarryAdder)
    def test_raises_on_wrong_num_bits(self, adder):
        """Test an error is raised for a bad number of qubits."""
        with self.assertRaises(ValueError):
            _ = adder(-1)


if __name__ == "__main__":
    unittest.main()
