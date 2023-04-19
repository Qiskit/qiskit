# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Tests evaluator of auxiliary operators for algorithms."""

from __future__ import annotations
import unittest
from typing import Tuple

from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np
from ddt import ddt, data

from qiskit.algorithms.list_or_dict import ListOrDict
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.algorithms import estimate_observables
from qiskit.primitives import Estimator
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit import QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import PauliSumOp
from qiskit.utils import algorithm_globals


@ddt
class TestObservablesEvaluator(QiskitAlgorithmsTestCase):
    """Tests evaluator of auxiliary operators for algorithms."""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed

        self.threshold = 1e-8

    def get_exact_expectation(
        self, ansatz: QuantumCircuit, observables: ListOrDict[BaseOperator | PauliSumOp]
    ):
        """
        Calculates the exact expectation to be used as an expected result for unit tests.
        """
        if isinstance(observables, dict):
            observables_list = list(observables.values())
        else:
            observables_list = observables
        # the exact value is a list of (mean, (variance, shots)) where we expect 0 variance and
        # 0 shots
        exact = [
            (Statevector(ansatz).expectation_value(observable), {})
            for observable in observables_list
        ]

        if isinstance(observables, dict):
            return dict(zip(observables.keys(), exact))

        return exact

    def _run_test(
        self,
        expected_result: ListOrDict[Tuple[complex, complex]],
        quantum_state: QuantumCircuit,
        decimal: int,
        observables: ListOrDict[BaseOperator | PauliSumOp],
        estimator: Estimator,
    ):
        result = estimate_observables(estimator, quantum_state, observables, None, self.threshold)

        if isinstance(observables, dict):
            np.testing.assert_equal(list(result.keys()), list(expected_result.keys()))
            means = [element[0] for element in result.values()]
            expected_means = [element[0] for element in expected_result.values()]
            np.testing.assert_array_almost_equal(means, expected_means, decimal=decimal)

            vars_and_shots = [element[1] for element in result.values()]
            expected_vars_and_shots = [element[1] for element in expected_result.values()]
            np.testing.assert_array_equal(vars_and_shots, expected_vars_and_shots)
        else:
            means = [element[0] for element in result]
            expected_means = [element[0] for element in expected_result]
            np.testing.assert_array_almost_equal(means, expected_means, decimal=decimal)

            vars_and_shots = [element[1] for element in result]
            expected_vars_and_shots = [element[1] for element in expected_result]
            np.testing.assert_array_equal(vars_and_shots, expected_vars_and_shots)

    @data(
        [
            PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)]),
            PauliSumOp.from_list([("II", 2.0)]),
        ],
        [
            PauliSumOp.from_list([("ZZ", 2.0)]),
        ],
        {
            "op1": PauliSumOp.from_list([("II", 2.0)]),
            "op2": PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)]),
        },
        {
            "op1": PauliSumOp.from_list([("ZZ", 2.0)]),
        },
        [],
        {},
    )
    def test_estimate_observables(self, observables: ListOrDict[BaseOperator | PauliSumOp]):
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

    def test_estimate_observables_zero_op(self):
        """Tests if a zero operator is handled correctly."""
        ansatz = EfficientSU2(2)
        parameters = np.array(
            [1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0],
            dtype=float,
        )

        bound_ansatz = ansatz.bind_parameters(parameters)
        state = bound_ansatz
        estimator = Estimator()
        observables = [SparsePauliOp(["XX", "YY"]), 0]
        result = estimate_observables(estimator, state, observables, None, self.threshold)
        expected_result = [(0.015607318055509564, {}), (0.0, {})]
        means = [element[0] for element in result]
        expected_means = [element[0] for element in expected_result]
        np.testing.assert_array_almost_equal(means, expected_means, decimal=0.01)

        vars_and_shots = [element[1] for element in result]
        expected_vars_and_shots = [element[1] for element in expected_result]
        np.testing.assert_array_equal(vars_and_shots, expected_vars_and_shots)

    def test_estimate_observables_shots(self):
        """Tests that variances and shots are returned properly."""
        ansatz = EfficientSU2(2)
        parameters = np.array(
            [1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0],
            dtype=float,
        )

        bound_ansatz = ansatz.bind_parameters(parameters)
        state = bound_ansatz
        estimator = Estimator(options={"shots": 2048})
        with self.assertWarns(DeprecationWarning):
            observables = [PauliSumOp.from_list([("ZZ", 2.0)])]
            result = estimate_observables(estimator, state, observables, None, self.threshold)
        exact_result = self.get_exact_expectation(bound_ansatz, observables)
        expected_result = [(exact_result[0][0], {"variance": 1.0898, "shots": 2048})]

        means = [element[0] for element in result]
        expected_means = [element[0] for element in expected_result]
        np.testing.assert_array_almost_equal(means, expected_means, decimal=0.01)

        vars_and_shots = [element[1] for element in result]
        expected_vars_and_shots = [element[1] for element in expected_result]
        for computed, expected in zip(vars_and_shots, expected_vars_and_shots):
            self.assertAlmostEqual(computed.pop("variance"), expected.pop("variance"), 2)
            self.assertEqual(computed.pop("shots"), expected.pop("shots"))


if __name__ == "__main__":
    unittest.main()
