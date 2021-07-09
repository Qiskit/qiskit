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
Tests AQC framework using hardcoded and randomly generated circuits.
"""
import unittest

import numpy as np

from qiskit.test import QiskitTestCase
from qiskit.transpiler.synthesis.aqc.aqc import AQC
from qiskit.transpiler.synthesis.aqc.cnot_structures import make_cnot_network
# from .test_sample_data import ORIGINAL_CIRCUIT, INITIAL_THETAS
from test.python.transpiler.aqc.test_sample_data import ORIGINAL_CIRCUIT, INITIAL_THETAS


class TestAqc(QiskitTestCase):
    """Main tests of approximate quantum compiler."""

    def setUp(self) -> None:
        super().setUp()
        self._maxiter = 1000
        self._eta = 0.01  # .1 for n=3, .01 for n=5
        self._tol = 0.01
        self._eps = 0.0
        self._reg = 0.7

    def test_aqc(self):
        """Tests AQC on a hardcoded circuit/matrix."""

        np.random.seed(12345)

        num_qubits = int(round(np.log2(np.array(ORIGINAL_CIRCUIT).shape[0])))
        cnots = make_cnot_network(
            num_qubits=num_qubits, network_layout="spin", connectivity_type="full", depth=0
        )

        aqc = AQC(
            method="nesterov",
            maxiter=self._maxiter,
            eta=self._eta,
            tol=self._tol,
            eps=self._eps,
        )

        optimized_circuit = aqc.compile_unitary(
            target_matrix=np.array(ORIGINAL_CIRCUIT),
            cnots=cnots,
            thetas0=np.array(INITIAL_THETAS),
        )

        error = 0.5 * (np.linalg.norm(optimized_circuit.to_matrix() - ORIGINAL_CIRCUIT, "fro") ** 2)
        print(error)
        self.assertTrue(error < 1e-3)


if __name__ == "__main__":
    unittest.main()
