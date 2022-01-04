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
"""Tests AnsatzMesValidator class."""
import unittest

import numpy as np

from qiskit import Aer
from qiskit.algorithms.gibbs_state_preparation.ansatz_mes_validator import _build_n_mes
from qiskit.quantum_info import Statevector
from test.python.algorithms import QiskitAlgorithmsTestCase


class TestAnsatzMesValidator(QiskitAlgorithmsTestCase):
    """Tests AnsatzMesValidator class."""

    def test_build_n_mes(self):
        """Test building 2 Maximally Entangled States."""
        num_states = 2
        backend = Aer.get_backend("statevector_simulator")
        mes = _build_n_mes(num_states, backend)
        expected_mes = Statevector(
            [
                0.5 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.5 + 0.0j,
                -0.0 + 0.0j,
                -0.0 + 0.0j,
                -0.0 + 0.0j,
                -0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.0 + 0.0j,
                0.5 + 0.0j,
                -0.0 + 0.0j,
                -0.0 + 0.0j,
                0.5 + 0.0j,
            ],
            dims=(2, 2, 2, 2),
        )
        expected_num_qubits = 4

        np.testing.assert_almost_equal(np.asarray(mes), expected_mes)
        np.testing.assert_equal(mes.num_qubits, expected_num_qubits)


if __name__ == "__main__":
    unittest.main()
