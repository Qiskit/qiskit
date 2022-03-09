# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""Tests evaluator of auxiliary operators for algorithms."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

import numpy as np
from ddt import ddt

from qiskit import BasicAer
from qiskit.algorithms.aux_ops_evaluator import eval_observables
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import PauliExpectation, PauliSumOp
from qiskit.utils import QuantumInstance


@ddt
class TestAuxOpsEvaluator(QiskitAlgorithmsTestCase):
    """Tests evaluator of auxiliary operators for algorithms."""

    def test_eval_observables(self):
        """Tests evaluator of auxiliary operators for algorithms."""

        quantum_instance = QuantumInstance(
            BasicAer.get_backend("qasm_simulator"),
            shots=1,
            seed_simulator=7,
            seed_transpiler=7,
        )
        ansatz = EfficientSU2(1)
        parameters = np.array([1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0], dtype=float)
        expectation = PauliExpectation()
        threshold = 1e-8

        observables = [
            PauliSumOp.from_list([("II", 2.0)]),
            PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)]),
        ]

        result = eval_observables(
            quantum_instance, ansatz, parameters, observables, expectation, threshold
        )

        expected_result = [(2.0, 0.0), (0.0, 0.0)]

        np.testing.assert_array_almost_equal(result, expected_result)


if __name__ == "__main__":
    unittest.main()
