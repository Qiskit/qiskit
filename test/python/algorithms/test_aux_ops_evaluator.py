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

import unittest
from typing import Tuple, Union

from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np
from ddt import ddt, data

from qiskit.algorithms.list_or_dict import ListOrDict
from qiskit.providers import Backend
from qiskit.quantum_info import Statevector
from qiskit.algorithms import eval_observables
from qiskit import BasicAer, QuantumCircuit
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import (
    PauliSumOp,
    X,
    Z,
    I,
    ExpectationFactory,
    OperatorBase,
    ExpectationBase,
    StateFn,
)
from qiskit.utils import QuantumInstance, algorithm_globals


@ddt
class TestAuxOpsEvaluator(QiskitAlgorithmsTestCase):
    """Tests evaluator of auxiliary operators for algorithms."""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        with self.assertWarns(DeprecationWarning):
            self.h2_op = (
                -1.052373245772859 * (I ^ I)
                + 0.39793742484318045 * (I ^ Z)
                - 0.39793742484318045 * (Z ^ I)
                - 0.01128010425623538 * (Z ^ Z)
                + 0.18093119978423156 * (X ^ X)
            )

        self.threshold = 1e-8
        self.backend_names = ["statevector_simulator", "qasm_simulator"]

    def get_exact_expectation(self, ansatz: QuantumCircuit, observables: ListOrDict[OperatorBase]):
        """
        Calculates the exact expectation to be used as an expected result for unit tests.
        """
        if isinstance(observables, dict):
            observables_list = list(observables.values())
        else:
            observables_list = observables

        # the exact value is a list of (mean, variance) where we expect 0 variance
        exact = [
            (Statevector(ansatz).expectation_value(observable), 0)
            for observable in observables_list
        ]

        if isinstance(observables, dict):
            return dict(zip(observables.keys(), exact))

        return exact

    def _run_test(
        self,
        expected_result: ListOrDict[Tuple[complex, complex]],
        quantum_state: Union[QuantumCircuit, Statevector],
        decimal: int,
        expectation: ExpectationBase,
        observables: ListOrDict[OperatorBase],
        quantum_instance: Union[QuantumInstance, Backend],
    ):

        with self.assertWarns(DeprecationWarning):
            result = eval_observables(
                quantum_instance, quantum_state, observables, expectation, self.threshold
            )

        if isinstance(observables, dict):
            np.testing.assert_equal(list(result.keys()), list(expected_result.keys()))
            np.testing.assert_array_almost_equal(
                list(result.values()), list(expected_result.values()), decimal=decimal
            )
        else:
            np.testing.assert_array_almost_equal(result, expected_result, decimal=decimal)

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
    )
    def test_eval_observables(self, observables: ListOrDict[OperatorBase]):
        """Tests evaluator of auxiliary operators for algorithms."""

        ansatz = EfficientSU2(2)
        parameters = np.array(
            [1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0],
            dtype=float,
        )

        bound_ansatz = ansatz.bind_parameters(parameters)
        expected_result = self.get_exact_expectation(bound_ansatz, observables)

        for backend_name in self.backend_names:
            shots = 4096 if backend_name == "qasm_simulator" else 1
            decimal = (
                1 if backend_name == "qasm_simulator" else 6
            )  # to accommodate for qasm being imperfect
            with self.subTest(msg=f"Test {backend_name} backend."):
                backend = BasicAer.get_backend(backend_name)
                with self.assertWarns(DeprecationWarning):
                    quantum_instance = QuantumInstance(
                        backend=backend,
                        shots=shots,
                        seed_simulator=self.seed,
                        seed_transpiler=self.seed,
                    )
                    expectation = ExpectationFactory.build(
                        operator=self.h2_op,
                        backend=quantum_instance,
                    )

                with self.subTest(msg="Test QuantumCircuit."):
                    self._run_test(
                        expected_result,
                        bound_ansatz,
                        decimal,
                        expectation,
                        observables,
                        quantum_instance,
                    )

                with self.subTest(msg="Test QuantumCircuit with Backend."):
                    self._run_test(
                        expected_result,
                        bound_ansatz,
                        decimal,
                        expectation,
                        observables,
                        backend,
                    )

                with self.subTest(msg="Test Statevector."):
                    statevector = Statevector(bound_ansatz)
                    self._run_test(
                        expected_result,
                        statevector,
                        decimal,
                        expectation,
                        observables,
                        quantum_instance,
                    )
                with self.assertWarns(DeprecationWarning):
                    with self.subTest(msg="Test StateFn."):
                        statefn = StateFn(bound_ansatz)
                        self._run_test(
                            expected_result,
                            statefn,
                            decimal,
                            expectation,
                            observables,
                            quantum_instance,
                        )


if __name__ == "__main__":
    unittest.main()
