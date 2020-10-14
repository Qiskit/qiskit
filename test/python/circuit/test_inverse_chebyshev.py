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

"""Test the inverse Chebyshev approximation class."""

import unittest
from collections import defaultdict
import numpy as np
from ddt import ddt, data, unpack

from qiskit.test.base import QiskitTestCase
from qiskit import BasicAer, execute
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library.arithmetic.inverse_chebyshev import InverseChebyshev


@ddt
class TestFunctionalPauliRotations(QiskitTestCase):
    """Test the inverse Chebyshev approximation class."""

    def assertFunctionIsCorrect(self, function_circuit, reference):
        """Assert that ``function_circuit`` implements the reference function ``reference``."""
        function_circuit._build()
        num_state_qubits = function_circuit.num_state_qubits
        num_ancilla_qubits = function_circuit.num_ancillas
        circuit = QuantumCircuit(num_state_qubits + 1 + num_ancilla_qubits)
        circuit.h(list(range(num_state_qubits)))
        circuit.append(function_circuit.to_instruction(), list(range(circuit.num_qubits)))

        backend = BasicAer.get_backend('statevector_simulator')
        statevector = execute(circuit, backend).result().get_statevector()

        probabilities = defaultdict(float)
        for i, statevector_amplitude in enumerate(statevector):
            i = bin(i)[2:].zfill(circuit.num_qubits)[num_ancilla_qubits:]
            probabilities[i] += np.real(np.abs(statevector_amplitude) ** 2)

        unrolled_probabilities = []
        unrolled_expectations = []
        for i, probability in probabilities.items():
            x, last_qubit = int(i[1:], 2), i[0]
            if last_qubit == '0':
                expected_amplitude = np.cos(reference(x)) / np.sqrt(2 ** num_state_qubits)
            else:
                expected_amplitude = np.sin(reference(x)) / np.sqrt(2 ** num_state_qubits)

            unrolled_probabilities += [probability]
            unrolled_expectations += [np.real(np.abs(expected_amplitude) ** 2)]

        assert np.linalg.norm(np.asarray(unrolled_probabilities) -
                              np.asarray(unrolled_expectations), 2) < \
               function_circuit.epsilon, "Error too large"

    # Last variable is set to 1 because it doesn't make sense unless used within the HHL class
    @data((4, 0.07, 1, 1),
          (3, 0.07, 2.3, 1),
          (3, 0.001, 1.11, 1),
          )
    @unpack
    def test_inverse_chebyshev(self, num_state_qubits, epsilon, constant, kappa):
        """Test the inverse Chebyshev approximation class."""

        def asin_inv(x):
            # calculate :math:`a`, where the domain is :math:`[a, 2^n - 1]`.
            left_domain = int(round(2 ** (2 * num_state_qubits / 3)))
            if x >= left_domain:
                return np.arcsin(constant / x)
            else:
                return np.arcsin(1)

        inv_cheb = InverseChebyshev(num_state_qubits, epsilon, constant, kappa)

        self.assertFunctionIsCorrect(inv_cheb, asin_inv)

    def test_inverse_chebyshev_mutability(self):
        """Test the mutability of the inverse Chebyshev approximation class."""

        def asin_inv(x):
            # calculate :math:`a`, where the domain is :math:`[a, 2^n - 1]`.
            left_domain = int(round(2 ** (2 * num_state_qubits / 3)))
            if x >= left_domain:
                return np.arcsin(constant / x)
            else:
                return np.arcsin(1)

        inv_cheb = InverseChebyshev()

        with self.subTest(msg='missing number of state qubits'):
            with self.assertRaises(AttributeError):  # no state qubits set
                print(inv_cheb.draw())

        with self.subTest(msg='default setup, just setting number of state qubits'):
            num_state_qubits = 3
            constant = 1
            inv_cheb.num_state_qubits = num_state_qubits
            self.assertFunctionIsCorrect(inv_cheb, asin_inv)

        with self.subTest(msg='setting non-default values'):
            epsilon = 0.08
            constant = 2.2
            inv_cheb.epsilon = epsilon
            inv_cheb.constant = constant
            self.assertFunctionIsCorrect(inv_cheb, asin_inv)

        with self.subTest(msg='changing all values'):
            num_state_qubits = 3
            epsilon = 0.1
            constant = 1.3
            inv_cheb.epsilon = epsilon
            inv_cheb.constant = constant
            self.assertFunctionIsCorrect(inv_cheb, asin_inv)


if __name__ == '__main__':
    unittest.main()
