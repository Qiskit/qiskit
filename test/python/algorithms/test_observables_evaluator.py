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
from typing import Tuple, Sequence, List
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np
from ddt import ddt, data

from qiskit.algorithms.observables_evaluator import eval_observables
from qiskit.primitives import Estimator
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import (
    PauliSumOp,
    OperatorBase,
)
from qiskit.utils import algorithm_globals


@ddt
class TestObservablesEvaluator(QiskitAlgorithmsTestCase):
    """Tests evaluator of auxiliary operators for algorithms."""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed

        self.threshold = 1e-8

    def get_exact_expectation(self, ansatz: QuantumCircuit, observables: Sequence[OperatorBase]):
        """
        Calculates the exact expectation to be used as an expected result for unit tests.
        """

        # the exact value is a list of (mean, variance) where we expect 0 variance
        exact = [
            (Statevector(ansatz).expectation_value(observable), 0) for observable in observables
        ]

        return exact

    def _run_test(
        self,
        expected_result: List[Tuple[complex, complex]],
        quantum_state: Sequence[QuantumCircuit],
        decimal: int,
        observables: Sequence[OperatorBase],
        estimator: Estimator,
    ):
        result = eval_observables(estimator, quantum_state, observables, self.threshold)

        np.testing.assert_array_almost_equal(result, expected_result, decimal=decimal)

    @data(
        [
            PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)]),
            PauliSumOp.from_list([("II", 2.0)]),
        ],
        [
            PauliSumOp.from_list([("ZZ", 2.0)]),
        ],
    )
    def test_eval_observables(self, observables: Sequence[OperatorBase]):
        """Tests evaluator of auxiliary operators for algorithms."""

        ansatz = EfficientSU2(2)
        parameters = np.array(
            [1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0],
            dtype=float,
        )

        bound_ansatz = ansatz.bind_parameters(parameters)
        states = bound_ansatz
        expected_result = self.get_exact_expectation(bound_ansatz, observables)
        estimator = Estimator()
        decimal = 6
        self._run_test(
            expected_result,
            states,
            decimal,
            observables,
            estimator,
        )


if __name__ == "__main__":
    unittest.main()
