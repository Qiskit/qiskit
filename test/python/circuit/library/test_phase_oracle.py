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

"""Test the phase oracle circuits."""

import unittest
from ddt import ddt, data, unpack
from numpy import sqrt, isclose

from qiskit.circuit import QuantumCircuit
from qiskit.test.base import QiskitTestCase
from qiskit.circuit.library import PhaseOracle
from qiskit.quantum_info import Statevector
from qiskit.utils.optionals import HAS_TWEEDLEDUM


@unittest.skipUnless(HAS_TWEEDLEDUM, "Tweedledum is required for these tests")
@ddt
class TestPhaseOracle(QiskitTestCase):
    """Test phase oracle object."""

    @data(
        ("x | x", "1", True),
        ("x & x", "0", False),
        ("(x0 & x1 | ~x2) ^ x4", "0110", False),
        ("xx & xxx | ( ~z ^ zz)", "0111", True),
    )
    @unpack
    def test_evaluate_bitstring(self, expression, input_bitstring, expected):
        """PhaseOracle(...).evaluate_bitstring"""
        oracle = PhaseOracle(expression)
        result = oracle.evaluate_bitstring(input_bitstring)
        self.assertEqual(result, expected)

    @data(
        ("x | x", [1]),
        ("x & y", [3]),
        ("(x0 & x1 | ~x2) ^ x4", [0, 1, 2, 3, 7, 12, 13, 14]),
        ("x & y ^ ( ~z1 | z2)", [0, 1, 2, 7, 8, 9, 10, 12, 13, 14]),
    )
    @unpack
    def test_statevector(self, expression, good_states):
        """Circuit generation"""
        oracle = PhaseOracle(expression)
        num_qubits = oracle.num_qubits
        circuit = QuantumCircuit(num_qubits)
        circuit.h(range(num_qubits))
        circuit.compose(oracle, inplace=True)
        statevector = Statevector.from_instruction(circuit)

        valid_state = -1 / sqrt(2**num_qubits)
        invalid_state = 1 / sqrt(2**num_qubits)

        states = list(range(2**num_qubits))
        expected_valid = [state in good_states for state in states]
        result_valid = [isclose(statevector.data[state], valid_state) for state in states]

        expected_invalid = [state not in good_states for state in states]
        result_invalid = [isclose(statevector.data[state], invalid_state) for state in states]
        self.assertListEqual(expected_valid, result_valid)
        self.assertListEqual(expected_invalid, result_invalid)

    @data(
        ("((A & C) | (B & D)) & ~(C & D)", None, [3, 7, 12, 13]),
        ("((A & C) | (B & D)) & ~(C & D)", ["A", "B", "C", "D"], [5, 7, 10, 11]),
    )
    @unpack
    def test_variable_order(self, expression, var_order, good_states):
        """Circuit generation"""
        oracle = PhaseOracle(expression, var_order=var_order)
        num_qubits = oracle.num_qubits
        circuit = QuantumCircuit(num_qubits)
        circuit.h(range(num_qubits))
        circuit.compose(oracle, inplace=True)
        statevector = Statevector.from_instruction(circuit)

        valid_state = -1 / sqrt(2**num_qubits)
        invalid_state = 1 / sqrt(2**num_qubits)

        states = list(range(2**num_qubits))
        expected_valid = [state in good_states for state in states]
        result_valid = [isclose(statevector.data[state], valid_state) for state in states]

        expected_invalid = [state not in good_states for state in states]
        result_invalid = [isclose(statevector.data[state], invalid_state) for state in states]
        self.assertListEqual(expected_valid, result_valid)
        self.assertListEqual(expected_invalid, result_invalid)


if __name__ == "__main__":
    unittest.main()
