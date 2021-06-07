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
from qiskit.circuit.library import (
    CDKMRippleCarryAdder,
    DraperQFTAdder,
    VBERippleCarryAdder,
    TwosComplement,
    Subtractor,
    )
@ddt
class TestSubtractor(QiskitTestCase):
    """Test the subtractor circuits."""
    

    def assertSubtractionIsCorrect(
        self, num_state_qubits: int, subtractor: QuantumCircuit
    ):
        """Assert that subtractor correctly implements the subtraction.

        This test prepares a equal superposition state in both input registers, then performs
        the subtraction on the superposition and checks that the output state is the expected
        superposition of all possible subtractions.

        Args:
            num_state_qubits: The number of bits in the numbers that are subtracted.
            subtractor: The circuit performing the subtraction of two numbers with ``num_state_qubits``
                bits.
            inplace: If True, compare against an inplace subtraction where the result is written into
                the second register plus carry qubit. If False, assume that the result is written
                into a third register of appropriate size.
            kind: TODO
        """
        def twoscomplement(value, bits):
            if value > 0:  # not negative, hence no twos complement
                return bin(value)[2:].zfill(bits + 1)
            binary = bin(-value)[2:].zfill(bits)
            print('b', binary)
            flipped = ''.join(['0' if bit == '1' else '1' for bit in binary])
            print('f', flipped)
            one_added = int(flipped, 2) + 1
            twos = '1' + bin(one_added)[2:].zfill(bits)  # leading 1 for negative number
            return twos

        circuit = QuantumCircuit(*subtractor.qregs)

        # create equal superposition
        #if kind == "full":
        #    num_superpos_qubits = 2 * num_state_qubits + 1
        #else:
        num_superpos_qubits = 2 * num_state_qubits
        circuit.h(range(num_superpos_qubits))

        # apply subtractor circuit
        #circuit.compose(subtractor, inplace=True)
        circuit.compose(subtractor)

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
                # compute the difference
                difference = x - y
                #Check for two's complement
                #bin_res=twoscomplement(difference,len(str(difference)))              
                #bin_res=twoscomplement(difference,len(difference))
                # compute correct index in statevector
                bin_x = bin(x)[2:].zfill(num_state_qubits)
                bin_y = bin(y)[2:].zfill(num_state_qubits)

                bin_res=twoscomplement(difference,len(str(difference)))
                bin_index = (
                            pad + bin_res + bin_x 
                            )
                
                index = int(bin_index, 2)
                expectations[index] += 1 / 2 ** num_superpos_qubits

        np.testing.assert_array_almost_equal(expectations, probabilities)

    @data(
         (3, Subtractor),
    #    (3, Subtractor, True),
    #    (5, Subtractor, True),
    #    (3, Subtractor, True, "fixed"),
    #    (5, Subtractor, True, "fixed"),
    #    (1, Subtractor, True, "full"),
    #    (3, Subtractor, True, "full"),
    #    (5, Subtractor, True, "full"),
    )
    @unpack
    #def test_subtraction(self, num_state_qubits, subtractor, inplace, kind="half"):
    def test_subtraction(self, num_state_qubits, subtractor):
        """Test subtraction for implemented subtractors."""
        #subtractor = subtractor(num_state_qubits, kind=kind)
        subtractor = Subtractor(num_state_qubits)
        #self.assertSubtractionIsCorrect(num_state_qubits, subtractor, inplace, kind)
        self.assertSubtractionIsCorrect(num_state_qubits, subtractor)
    @data(Subtractor)

    def test_raises_on_wrong_num_bits(self, adder):
        """Test an error is raised for a bad number of qubits."""
        with self.assertRaises(ValueError):
            _ = Subtractor(-1)

    def twoscomplement(value, bits):
        if value > 0:  # not negative, hence no twos complement
            return bin(value)[2:].zfill(bits + 1)
        binary = bin(-value)[2:].zfill(bits)
        print('b', binary)
        flipped = ''.join(['0' if bit == '1' else '1' for bit in binary])
        print('f', flipped)
        one_added = int(flipped, 2) + 1
        twos = '1' + bin(one_added)[2:].zfill(bits)  # leading 1 for negative number
        return twos

    #print(twoscomplement(1, 3))
    #print(twoscomplement(-1, 3))
    #print(twoscomplement(-2, 3))
    #print(twoscomplement(-4, 3))
    

if __name__ == "__main__":
    unittest.main()
