# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Generalized Uniform Superposition Gate test.
"""

import unittest
import numpy as np
from ddt import ddt, data

from qiskit.circuit.library.data_preparation import (
    Generalized_Uniform_Superposition_Gate,
)
from test import QiskitTestCase


@ddt
class TestGeneralizedUniformSuperposition(QiskitTestCase):
    """Test initialization with Generalized_Uniform_Superposition_Gate class"""

    def test_generalized_uniform_superposition_gate(self):
        """Test Generalized Uniform Superposition Gate"""
        M_min = 3
        M_max = 80
        for M in range(M_min, M_max):
            if (M & (M - 1)) == 0:  # If M is an integer power of 2
                n = int(np.log2(M))
            else:  # If M is not an integer power of 2
                n = int(np.ceil(np.log2(M)))
            desired_sv = (1 / np.sqrt(M)) * np.array([1] * M + [0] * (2**n - M))
            gate = Generalized_Uniform_Superposition_Gate(M, n)
            unitary_matrix = np.real(gate.to_unitary())
            actual_sv = unitary_matrix[:, 0]
            self.assertTrue(np.allclose(desired_sv, actual_sv))

    def test_incompatible_M(self):
        """Test error raised if M not valid"""
        M_min = -2
        M_max = 2
        n = 1
        for M in range(M_min, M_max):
            with self.assertRaises(ValueError):
                Generalized_Uniform_Superposition_Gate(M, n)

    def test_incompatible_int_M_and_qubit_args(self):
        """Test error raised if number of qubits not compatible with  integer state M (n >= log2(M) )"""
        n_min = 1
        n_max = 5
        M = 50
        for n in range(n_min, n_max):
            with self.assertRaises(ValueError):
                Generalized_Uniform_Superposition_Gate(M, n)

    def test_no_qubit_args(self):
        """Tests error raised if number of qubits not compatible with  integer state M (n >= log2(M) )"""
        M_min = 3
        M_max = 10
        for M in range(M_min, M_max):
            if (M & (M - 1)) == 0:  # If M is an integer power of 2
                n = int(np.log2(M))
            else:  # If M is not an integer power of 2
                n = int(np.ceil(np.log2(M)))
            desired_sv = (1 / np.sqrt(M)) * np.array([1] * M + [0] * (2**n - M))
            num_qubits = None
            gate = Generalized_Uniform_Superposition_Gate(M, num_qubits)
            unitary_matrix = np.real(gate.to_unitary())
            actual_sv = unitary_matrix[:, 0]
            self.assertTrue(np.allclose(desired_sv, actual_sv))


if __name__ == "__main__":
    unittest.main()
