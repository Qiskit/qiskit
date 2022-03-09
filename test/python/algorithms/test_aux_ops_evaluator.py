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

from qiskit import BasicAer
from qiskit.algorithms.aux_ops_evaluator import eval_observables
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
    )
    @unpack
    def test_eval_observables_statevector(self, observables, expected_result):
        """Tests evaluator of auxiliary operators for algorithms."""

        backend = BasicAer.get_backend("statevector_simulator")
        quantum_instance = QuantumInstance(
            backend=backend, shots=1, seed_simulator=self.seed, seed_transpiler=self.seed
        )

        ansatz = EfficientSU2(1)
        parameters = np.array([1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0], dtype=float)
        expectation = ExpectationFactory.build(
            operator=self.h2_op,
            backend=quantum_instance,
        )

        result = eval_observables(
            quantum_instance, ansatz, parameters, observables, expectation, self.threshold
        )

        np.testing.assert_array_almost_equal(result, expected_result)

    @data(
        (
            [
                PauliSumOp.from_list([("II", 2.0)]),
                PauliSumOp.from_list([("II", 0.5), ("ZZ", 0.5), ("YY", 0.5), ("XX", -0.5)]),
            ],
            [(2.0, 0.0), (0.39355468750000006, 0.015266813075786104)],
        ),
        (
            [
                PauliSumOp.from_list([("ZZ", 2.0)]),
            ],
            [(-0.42578124999999967, 0.06106725230314441)],
        ),
    )
    @unpack
    def test_eval_observables_qasm(self, observables, expected_result):
        """Tests evaluator of auxiliary operators for algorithms."""
        backend = BasicAer.get_backend("qasm_simulator")
        quantum_instance = QuantumInstance(
            backend=backend, shots=1024, seed_simulator=self.seed, seed_transpiler=self.seed
        )

        ansatz = EfficientSU2(1)
        parameters = np.array([1.2, 4.2, 1.4, 2.0, 1.2, 4.2, 1.4, 2.0], dtype=float)
        expectation = ExpectationFactory.build(
            operator=self.h2_op,
            backend=quantum_instance,
        )

        result = eval_observables(
            quantum_instance, ansatz, parameters, observables, expectation, self.threshold
        )

        np.testing.assert_array_almost_equal(result, expected_result)


if __name__ == "__main__":
    unittest.main()
