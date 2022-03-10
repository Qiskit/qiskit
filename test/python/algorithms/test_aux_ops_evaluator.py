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
from ddt import ddt, data, unpack

from qiskit.algorithms.aux_ops_evaluator import eval_observables
from qiskit import BasicAer
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import PauliSumOp, X, Z, I, ExpectationFactory
from qiskit.utils import QuantumInstance, algorithm_globals


@ddt
class TestAuxOpsEvaluator(QiskitAlgorithmsTestCase):
    """Tests evaluator of auxiliary operators for algorithms."""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        self.h2_op = (
            -1.052373245772859 * (I ^ I)
            + 0.39793742484318045 * (I ^ Z)
            - 0.39793742484318045 * (Z ^ I)
            - 0.01128010425623538 * (Z ^ Z)
            + 0.18093119978423156 * (X ^ X)
        )

        self.threshold = 1e-8
        self.backend_names = ["statevector_simulator", "qasm_simulator"]

    @data(
        (
            [
                PauliSumOp.from_list([("II", 2.0)]),
                PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)]),
            ],
            [(1.9999999999999998, 0.0), (0.3819044157958812, 0.0)],
        ),
        (
            [
                PauliSumOp.from_list([("ZZ", 2.0)]),
            ],
            [(-0.4723823368164749, 0.0)],
        ),
        (
            {
                "op1": PauliSumOp.from_list([("II", 2.0)]),
                "op2": PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)]),
            },
            {"op1": (1.9999999999999998, 0.0), "op2": (0.3819044157958812, 0.0)},
        ),
        (
            {
                "op1": PauliSumOp.from_list([("ZZ", 2.0)]),
            },
            {"op1": (-0.4723823368164749, 0.0)},
        ),
    )
    @unpack
    def test_eval_observables_statevector(self, observables, expected_result):
        """Tests evaluator of auxiliary operators for algorithms."""

        ansatz = EfficientSU2(1)
        parameters = np.array([1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0], dtype=float)
        param_dict = dict(zip(ansatz.ordered_parameters, parameters))
        bound_ansatz = ansatz.bind_parameters(param_dict)

        for backend_name in self.backend_names:
            shots = 2048 if backend_name == "qasm_simulator" else 1
            decimal = (
                1 if backend_name == "qasm_simulator" else 6
            )  # to accommodate for qasm being imperfect
            with self.subTest(msg=f"Test {backend_name} backend."):
                backend = BasicAer.get_backend(backend_name)
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
                result = eval_observables(
                    quantum_instance, bound_ansatz, observables, expectation, self.threshold
                )

                if isinstance(result, dict):
                    np.testing.assert_equal(list(result.keys()), list(expected_result.keys()))
                    np.testing.assert_array_almost_equal(
                        list(result.values()), list(expected_result.values()), decimal=decimal
                    )
                else:
                    np.testing.assert_array_almost_equal(result, expected_result, decimal=decimal)


if __name__ == "__main__":
    unittest.main()
