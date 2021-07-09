# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Tests the implementation of parametric circuit.
"""

import unittest
from typing import Union

import numpy as np

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.aqc.parametric_circuit import ParametricCircuit


class TestParametricCircuit(QiskitTestCase):
    """Tests ParametricCircuit."""

    def test_matrix_conversion(self):
        """Tests matrix conversion."""

        # pick random number of qubits and cnots.
        num_qubits = np.random.randint(2, 10)
        num_cnots = np.random.randint(10, 100)

        circuit = ParametricCircuit(
            num_qubits=num_qubits, layout="spin", connectivity="full", depth=num_cnots
        )

        thetas = np.random.rand(circuit.num_thetas) * (2.0 * np.pi)
        circuit.set_thetas(thetas)
        residual = self._compare_circuits(
            target_circuit=circuit.to_matrix(), approx_circuit=circuit.to_circuit(reverse=False)
        )
        print(residual)
        self.assertLess(residual, 1e-10)

    def test_basic_functions(self):
        """Tests basic functions."""

        # pick random number of qubits and cnots.
        num_qubits = np.random.randint(2, 10)
        num_cnots = np.random.randint(10, 100)
        # for num_qubits in range(2, 10):
        #     for num_cnots in np.random.permutation(range(10, 100))[0:10]:
        circuit = ParametricCircuit(
            num_qubits=num_qubits, layout="spin", connectivity="full", depth=num_cnots
        )

        thetas = np.random.rand(circuit.num_thetas) * (2.0 * np.pi)
        circuit.set_thetas(thetas)
        self.assertTrue(np.allclose(thetas, circuit.thetas))

        self.assertEqual(circuit.cnots.shape, (2, circuit.num_cnots))

        # test that cnots are placed on the correct qubits
        # pylint: disable=misplaced-comparison-constant
        self.assertTrue(np.all(1 <= circuit.cnots))
        self.assertTrue(np.all(circuit.cnots <= circuit.num_qubits))

    def _compare_circuits(
        self,
        target_circuit: Union[np.ndarray, QuantumCircuit],
        approx_circuit: Union[np.ndarray, QuantumCircuit],
    ) -> float:
        """
        Compares two circuits (or their underlying matrices) for equivalence

        Args:
            target_circuit: the circuit that we try to approximate.
            approx_circuit: the circuit obtained by approximate compiling.

        Returns:
            relative difference between two circuits.
        """
        target_unitary = self._circuit_to_numpy(target_circuit)
        approx_unitary = self._circuit_to_numpy(approx_circuit)

        return 0.5 * (np.linalg.norm(approx_unitary - target_unitary, "fro") ** 2)

    def _circuit_to_numpy(self, circuit: Union[np.ndarray, QuantumCircuit]) -> np.ndarray:
        """
        Converts a quantum circuit to a Numpy matrix or returns an array if the input is already
        a Numpy matrix.

        Args:
            circuit: the circuit to be converted into a Numpy matrix.

        Returns:
            Numpy matrix underlying the input circuit.
        """
        if isinstance(circuit, QuantumCircuit):
            return Operator(circuit).data
        else:
            return circuit


if __name__ == "__main__":
    unittest.main()
