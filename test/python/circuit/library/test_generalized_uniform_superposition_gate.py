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
from test import QiskitTestCase

from qiskit.circuit.library.data_preparation import (
    GeneralizedUniformSuperpositionGate,
)

import numpy as np


class TestGeneralizedUniformSuperposition(QiskitTestCase):
    """Test initialization with GeneralizedUniformSuperpositionGate class"""

    def test_generalized_uniform_superposition_gate(self):
        """Test Generalized Uniform Superposition Gate"""
        m_value_min = 3
        m_value_max = 80
        for m_value in range(m_value_min, m_value_max):
            if (m_value & (m_value - 1)) == 0:  # If m_value is an integer power of 2
                n = int(np.log2(m_value))
            else:  # If m_value is not an integer power of 2
                n = int(np.ceil(np.log2(m_value)))
            desired_sv = (1 / np.sqrt(m_value)) * np.array([1] * m_value + [0] * (2**n - m_value))
            gate = GeneralizedUniformSuperpositionGate(m_value, n)
            unitary_matrix = np.real(gate.to_unitary())
            actual_sv = unitary_matrix[:, 0]
            self.assertTrue(np.allclose(desired_sv, actual_sv))

    def test_incompatible_m_value(self):
        """Test error raised if m_value not valid"""
        m_value_min = -2
        m_value_max = 2
        n = 1
        for m_value in range(m_value_min, m_value_max):
            with self.assertRaises(ValueError):
                GeneralizedUniformSuperpositionGate(m_value, n)

    def test_incompatible_int_m_value_and_qubit_args(self):
        """Test error raised if number of qubits not compatible with integer
        state m_value (n >= log2(m_value) )"""
        n_min = 1
        n_max = 5
        m_value = 50
        for n in range(n_min, n_max):
            with self.assertRaises(ValueError):
                GeneralizedUniformSuperpositionGate(m_value, n)

    def test_no_qubit_args(self):
        """Tests error raised if number of qubits not compatible with integer
        state m_value (n >= log2(m_value) )"""
        m_value_min = 3
        m_value_max = 10
        for m_value in range(m_value_min, m_value_max):
            if (m_value & (m_value - 1)) == 0:  # If m_value is an integer power of 2
                n = int(np.log2(m_value))
            else:  # If m_value is not an integer power of 2
                n = int(np.ceil(np.log2(m_value)))
            desired_sv = (1 / np.sqrt(m_value)) * np.array([1] * m_value + [0] * (2**n - m_value))
            num_qubits = None
            gate = GeneralizedUniformSuperpositionGate(m_value, num_qubits)
            unitary_matrix = np.real(gate.to_unitary())
            actual_sv = unitary_matrix[:, 0]
            self.assertTrue(np.allclose(desired_sv, actual_sv))


if __name__ == "__main__":
    unittest.main()
