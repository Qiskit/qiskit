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

"""Test arithmetic circuits."""

import unittest
import operator
from typing import Callable
import numpy as np
from ddt import ddt, data, unpack

from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector, partial_trace
from qiskit.circuit.library import RippleCarryAdder, QFTAdder, ClassicalAdder, ClassicalMultiplier


@ddt
class TestArithmetic(QiskitTestCase):
    """Test the arithmetic circuits."""

    def assertArithmeticIsCorrect(self,
                                  num_state_qubits: int,
                                  arithmetic_circuit: QuantumCircuit,
                                  op: Callable[[int, int], int],
                                  num_res_qubits: int,
                                  inplace: bool,
                                  modular: bool = False):
        """Assert that arithmetic circuit correctly implements the operation.

        Args:
            num_state_qubits: The number of bits in the numbers that are used for arithmethic
                operation.
            arithmetic_circuit: The circuit performing the arithmetic operation
                of two numbers with ``num_state_qubits`` bits.
            op: The arithmetic operator to be tested.
            num_res_qubits: The number of qubits required to store arithmetic result.
            inplace: If True, compare against an inplace operation where the result is written
                into the second register plus auxiliary qubits. If False, assume that the
                result is written into a third register of appropriate size.
            modular: If True, omit the carry qubit to obtain a modulo ``2^num_state_qubits``
                operation.
        """
        circuit = QuantumCircuit(*arithmetic_circuit.qregs)
        # create equal superposition
        circuit.h(range(2 * num_state_qubits))
        # apply arithmetic circuit
        circuit.compose(arithmetic_circuit, inplace=True)
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
        # iterate over all possible inputs
        for x in range(2 ** num_state_qubits):
            for y in range(2 ** num_state_qubits):
                # compute the arithmetic result
                arithmetic_res = op(x, y) % (2 ** num_state_qubits) if modular else op(x, y)
                # compute correct index in statevector
                bin_x = bin(x)[2:].zfill(num_state_qubits)
                bin_y = bin(y)[2:].zfill(num_state_qubits)
                bin_res = bin(arithmetic_res)[2:].zfill(num_res_qubits)
                bin_index = bin_res + bin_x if inplace else bin_res + bin_y + bin_x
                index = int(bin_index, 2)
                expectations[index] += 1 / 2 ** (2 * num_state_qubits)
        np.testing.assert_array_almost_equal(expectations, probabilities)

    @data(
        (3, RippleCarryAdder, True),
        (5, RippleCarryAdder, True),
        (3, QFTAdder, True),
        (5, QFTAdder, True),
        (3, QFTAdder, True, True),
        (5, QFTAdder, True, True),
        (3, ClassicalAdder, True),
        (5, ClassicalAdder, True),
    )
    @unpack
    def test_summation(self, num_state_qubits, adder, inplace, modular=False):
        """Test summation for all implemented adders."""
        num_res_qubits = num_state_qubits + 1
        adder = adder(num_state_qubits, modular=True) if modular else adder(num_state_qubits)
        self.assertArithmeticIsCorrect(num_state_qubits, adder, operator.add,
                                       num_res_qubits, inplace, modular)

    @data(
        (3, ClassicalMultiplier, False, None),
        (3, ClassicalMultiplier, False, ClassicalAdder),
        (3, ClassicalMultiplier, False, RippleCarryAdder),
        (3, ClassicalMultiplier, False, QFTAdder)
    )
    @unpack
    def test_multiplication(self, num_state_qubits, multiplier, inplace, adder):
        """Test multiplication for all implemented multipliers."""
        num_res_qubits = 2 * num_state_qubits
        if adder:
            adder = adder(num_state_qubits)
        multiplier = multiplier(num_state_qubits, adder=adder)
        self.assertArithmeticIsCorrect(num_state_qubits, multiplier, operator.mul,
                                       num_res_qubits, inplace)

    @data(
        RippleCarryAdder,
        QFTAdder,
        ClassicalAdder,
        ClassicalMultiplier
    )
    def test_raises_on_wrong_num_bits(self, adder):
        """Test an error is raised for a bad number of qubits."""
        with self.assertRaises(ValueError):
            _ = adder(-1)


if __name__ == '__main__':
    unittest.main()
