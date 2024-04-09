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

from qiskit.circuit.library.data_preparation import Generalized_Uniform_Superposition_Gate
from qiskit.test import QiskitTestCase

class TestGeneralizedUniformSuperposition(QiskitTestCase):
    """Test initialization with Generalized_Uniform_Superposition_Gate class"""

    def test_generalized_uniform_superposition_gate(self):
        """Test Generalized Uniform Superposition Gate"""
        M_min = 3
        M_max = 130
        for M in range(M_min, M_max):
            if (M & (M-1)) == 0: # If M is an integer power of 2
                n = int(np.log2(M))
            else: # If M is not an integer power of 2
                n = int(np.ceil(np.log2(M)))
            desired_sv = (1/np.sqrt(M))*np.array([1]*M + [0]*(2**n - M))
            gate = Generalized_Uniform_Superposition_Gate(M, n)
            unitary_matrix = np.real(gate.to_unitary())
            actual_sv = unitary_matrix[:,0]
            self.assertTrue(np.allclose(desired_sv, actual_sv))

if __name__ == "__main__":
    unittest.main()
