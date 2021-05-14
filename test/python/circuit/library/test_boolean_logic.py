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

"""Test the boolean logic circuits."""

import unittest
from ddt import ddt, data, unpack
import numpy as np

from qiskit.test.base import QiskitTestCase
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import XOR, InnerProduct, AND, OR
from qiskit.quantum_info import Statevector


@ddt
class TestBooleanLogicLibrary(QiskitTestCase):
    """Test library of boolean logic quantum circuits."""

    def assertBooleanFunctionIsCorrect(self, boolean_circuit, reference):
        """Assert that ``boolean_circuit`` implements the reference boolean function correctly."""
        circuit = QuantumCircuit(boolean_circuit.num_qubits)
        circuit.h(list(range(boolean_circuit.num_variable_qubits)))
        circuit.append(boolean_circuit.to_instruction(), list(range(boolean_circuit.num_qubits)))

        # compute the statevector of the circuit
        statevector = Statevector.from_label("0" * circuit.num_qubits)
        statevector = statevector.evolve(circuit)

        # trace out ancillas
        probabilities = statevector.probabilities(
            qargs=list(range(boolean_circuit.num_variable_qubits + 1))
        )

        # compute the expected outcome by computing the entries of the statevector that should
        # have a 1 / sqrt(2**n) factor
        expectations = np.zeros_like(probabilities)
        for x in range(2 ** boolean_circuit.num_variable_qubits):
            bits = np.array(list(bin(x)[2:].zfill(boolean_circuit.num_variable_qubits)), dtype=int)
            result = reference(bits[::-1])

            entry = int(str(int(result)) + bin(x)[2:].zfill(boolean_circuit.num_variable_qubits), 2)
            expectations[entry] = 1 / 2 ** boolean_circuit.num_variable_qubits

        np.testing.assert_array_almost_equal(probabilities, expectations)

    def test_xor(self):
        """Test xor circuit.

        TODO add a test using assertBooleanFunctionIsCorrect
        """
        circuit = XOR(num_qubits=3, amount=4)
        expected = QuantumCircuit(3)
        expected.x(2)
        self.assertEqual(circuit.decompose(), expected)

    def test_inner_product(self):
        """Test inner product circuit.

        TODO add a test using assertBooleanFunctionIsCorrect
        """
        circuit = InnerProduct(num_qubits=3)
        expected = QuantumCircuit(*circuit.qregs)
        expected.cz(0, 3)
        expected.cz(1, 4)
        expected.cz(2, 5)
        self.assertEqual(circuit.decompose(), expected)

    @data(
        (2, None, "noancilla"),
        (5, None, "noancilla"),
        (2, [-1, 1], "v-chain"),
        (2, [-1, 1], "noancilla"),
        (5, [0, 0, -1, 1, -1], "noancilla"),
        (5, [-1, 0, 0, 1, 1], "v-chain"),
    )
    @unpack
    def test_or(self, num_variables, flags, mcx_mode):
        """Test the or circuit."""
        or_circuit = OR(num_variables, flags, mcx_mode=mcx_mode)
        flags = flags or [1] * num_variables

        def reference(bits):
            flagged = []
            for flag, bit in zip(flags, bits):
                if flag < 0:
                    flagged += [1 - bit]
                elif flag > 0:
                    flagged += [bit]
            return np.any(flagged)

        self.assertBooleanFunctionIsCorrect(or_circuit, reference)

    @data(
        (2, None, "noancilla"),
        (2, [-1, 1], "v-chain"),
        (5, [0, 0, -1, 1, -1], "noancilla"),
        (5, [-1, 0, 0, 1, 1], "v-chain"),
    )
    @unpack
    def test_and(self, num_variables, flags, mcx_mode):
        """Test the and circuit."""
        and_circuit = AND(num_variables, flags, mcx_mode=mcx_mode)
        flags = flags or [1] * num_variables

        def reference(bits):
            flagged = []
            for flag, bit in zip(flags, bits):
                if flag < 0:
                    flagged += [1 - bit]
                elif flag > 0:
                    flagged += [bit]
            return np.all(flagged)

        self.assertBooleanFunctionIsCorrect(and_circuit, reference)


if __name__ == "__main__":
    unittest.main()
