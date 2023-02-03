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

"""Tests for LinearFunction class."""

import unittest
import numpy as np
from ddt import ddt, data

from qiskit.test import QiskitTestCase
from qiskit.circuit import QuantumCircuit

from qiskit.circuit.library.standard_gates import CXGate, SwapGate
from qiskit.circuit.library.generalized_gates import LinearFunction
from qiskit.circuit.exceptions import CircuitError
from qiskit.synthesis.linear import random_invertible_binary_matrix

from qiskit.quantum_info.operators import Operator


def random_linear_circuit(num_qubits, num_gates, seed=None):
    """Generate a pseudo random linear circuit."""

    instructions = {
        "cx": (CXGate(), 2),
        "swap": (SwapGate(), 2),
    }

    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    name_samples = rng.choice(tuple(instructions), num_gates)

    circ = QuantumCircuit(num_qubits)

    for name in name_samples:
        gate, nqargs = instructions[name]
        qargs = rng.choice(range(num_qubits), nqargs, replace=False).tolist()
        circ.append(gate, qargs)

    return circ


@ddt
class TestLinearFunctions(QiskitTestCase):
    """Tests for clifford append gate functions."""

    @data(2, 3, 4, 5, 6, 7, 8)
    def test_conversion_to_matrix_and_back(self, num_qubits):
        """Test correctness of first constructing a linear function from a linear quantum circuit,
        and then synthesizing this linear function to a quantum circuit."""
        rng = np.random.default_rng(1234)

        for _ in range(10):
            for num_gates in [0, 5, 5 * num_qubits]:
                # create a random linear circuit
                linear_circuit = random_linear_circuit(num_qubits, num_gates, seed=rng)
                self.assertIsInstance(linear_circuit, QuantumCircuit)

                # convert it to a linear function
                linear_function = LinearFunction(linear_circuit, validate_input=True)

                # check that the internal matrix has right dimensions
                self.assertEqual(linear_function.linear.shape, (num_qubits, num_qubits))

                # synthesize linear function
                synthesized_linear_function = linear_function.definition
                self.assertIsInstance(synthesized_linear_function, QuantumCircuit)

                # check that the synthesized linear function only contains CX and SWAP gates
                for instruction in synthesized_linear_function.data:
                    self.assertIsInstance(instruction.operation, (CXGate, SwapGate))

                # check equivalence to the original function
                self.assertEqual(Operator(linear_circuit), Operator(synthesized_linear_function))

    @data(2, 3, 4, 5, 6, 7, 8)
    def test_conversion_to_linear_function_and_back(self, num_qubits):
        """Test correctness of first synthesizing a linear circuit from a linear function,
        and then converting this linear circuit to a linear function."""
        rng = np.random.default_rng(5678)

        for _ in range(10):
            # create a random invertible binary matrix
            binary_matrix = random_invertible_binary_matrix(num_qubits, seed=rng)

            # create a linear function with this matrix
            linear_function = LinearFunction(binary_matrix, validate_input=True)
            self.assertTrue(np.all(linear_function.linear == binary_matrix))

            # synthesize linear function
            synthesized_circuit = linear_function.definition
            self.assertIsInstance(synthesized_circuit, QuantumCircuit)

            # check that the synthesized linear function only contains CX and SWAP gates
            for instruction in synthesized_circuit.data:
                self.assertIsInstance(instruction.operation, (CXGate, SwapGate))

            # construct a linear function out of this linear circuit
            synthesized_linear_function = LinearFunction(synthesized_circuit, validate_input=True)

            # check equivalence of the two linear matrices
            self.assertTrue(np.all(synthesized_linear_function.linear == binary_matrix))

    def test_patel_markov_hayes(self):
        """Checks the explicit example from Patel-Markov-Hayes's paper."""

        # This code is adapted from test_gray_synthesis.py
        binary_matrix = [
            [1, 1, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 0],
            [0, 1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1],
            [0, 0, 1, 1, 1, 0],
        ]

        # Construct linear function from matrix: here, we copy a matrix
        linear_function_from_matrix = LinearFunction(binary_matrix, validate_input=True)

        # Create the circuit displayed above:
        linear_circuit = QuantumCircuit(6)
        linear_circuit.cx(4, 3)
        linear_circuit.cx(5, 2)
        linear_circuit.cx(1, 0)
        linear_circuit.cx(3, 1)
        linear_circuit.cx(4, 2)
        linear_circuit.cx(4, 3)
        linear_circuit.cx(5, 4)
        linear_circuit.cx(2, 3)
        linear_circuit.cx(3, 2)
        linear_circuit.cx(3, 5)
        linear_circuit.cx(2, 4)
        linear_circuit.cx(1, 2)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(0, 4)
        linear_circuit.cx(0, 3)

        # Construct linear function from matrix: we build a matrix
        linear_function_from_circuit = LinearFunction(linear_circuit, validate_input=True)

        # Compare the matrices
        self.assertTrue(
            np.all(linear_function_from_circuit.linear == linear_function_from_matrix.linear)
        )

        self.assertTrue(
            Operator(linear_function_from_matrix.definition) == Operator(linear_circuit)
        )

    def test_bad_matrix_non_rectangular(self):
        """Tests that an error is raised if the matrix is not rectangular."""
        mat = [[1, 1, 0, 0], [1, 0, 0], [0, 1, 0, 0], [1, 1, 1, 1]]
        with self.assertRaises(CircuitError):
            LinearFunction(mat)

    def test_bad_matrix_non_square(self):
        """Tests that an error is raised if the matrix is not square."""
        mat = [[1, 1, 0], [1, 0, 0], [0, 1, 0], [1, 1, 1]]
        with self.assertRaises(CircuitError):
            LinearFunction(mat)

    def test_bad_matrix_non_two_dimensional(self):
        """Tests that an error is raised if the matrix is not two-dimensional."""
        mat = [1, 0, 0, 1, 0]
        with self.assertRaises(CircuitError):
            LinearFunction(mat)

    def test_bad_matrix_non_invertible(self):
        """Tests that an error is raised if the matrix is not invertible."""
        mat = [[1, 0, 0], [0, 1, 1], [1, 1, 1]]
        with self.assertRaises(CircuitError):
            LinearFunction(mat, validate_input=True)

    def test_bad_circuit_non_linear(self):
        """Tests that an error is raised if a circuit is not linear."""
        non_linear_circuit = QuantumCircuit(4)
        non_linear_circuit.cx(0, 1)
        non_linear_circuit.swap(2, 3)
        non_linear_circuit.h(2)
        non_linear_circuit.swap(1, 2)
        non_linear_circuit.cx(1, 3)
        with self.assertRaises(CircuitError):
            LinearFunction(non_linear_circuit)

    def test_is_permutation(self):
        """Tests that a permutation is detected correctly."""
        mat = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]
        linear_function = LinearFunction(mat)
        self.assertTrue(linear_function.is_permutation())

    def test_permutation_pattern(self):
        """Tests that a permutation pattern is returned correctly when
        the linear function is a permutation."""
        mat = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]
        linear_function = LinearFunction(mat)
        pattern = linear_function.permutation_pattern()
        self.assertIsInstance(pattern, np.ndarray)

    def test_is_not_permutation(self):
        """Tests that a permutation is detected correctly."""
        mat = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]]
        linear_function = LinearFunction(mat)
        self.assertFalse(linear_function.is_permutation())

    def test_no_permutation_pattern(self):
        """Tests that an error is raised when when
        the linear function is not a permutation."""
        mat = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 0]]
        linear_function = LinearFunction(mat)
        with self.assertRaises(CircuitError):
            linear_function.permutation_pattern()

    def test_original_definition(self):
        """Tests that when a linear function is constructed from
        a QuantumCircuit, it saves the original definition."""
        linear_circuit = QuantumCircuit(4)
        linear_circuit.cx(0, 1)
        linear_circuit.cx(1, 2)
        linear_circuit.cx(2, 3)
        linear_function = LinearFunction(linear_circuit)
        self.assertIsNotNone(linear_function.original_circuit)

    def test_no_original_definition(self):
        """Tests that when a linear function is constructed from
        a matrix, there is no original definition."""
        mat = [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]]
        linear_function = LinearFunction(mat)
        self.assertIsNone(linear_function.original_circuit)


if __name__ == "__main__":
    unittest.main()
