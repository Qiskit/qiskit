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
import unittest

import numpy as np
from ddt import ddt, data

from qiskit.algorithms.quantum_time_evolution.variational.calculators.bures_distance_calculator import (
    _calculate_bures_distance,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z, StateFn
from test.python.algorithms import QiskitAlgorithmsTestCase


@ddt
class TestBuresDistanceCalculator(QiskitAlgorithmsTestCase):
    @data("cobyla", "nelder-mead", "l-bfgs-b")
    def test_calculate_bures_distance_same_state(self, optimizer):
        observable = SummedOp(
            [
                0.2252 * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        ).reduce()

        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        # Define a set of initial parameters
        parameters = ansatz.ordered_parameters

        operator = ~StateFn(observable) @ StateFn(ansatz)
        param_dict = {param: np.pi / 4 for param in parameters}
        state1 = operator[-1].assign_parameters(param_dict).eval().primitive.data

        bures_distance = _calculate_bures_distance(state1, state1, optimizer=optimizer)
        expected_bures_distance = 0

        np.testing.assert_almost_equal(bures_distance, expected_bures_distance, decimal=6)

    @data("cobyla", "nelder-mead", "l-bfgs-b")
    def test_calculate_bures_distance_different_states(self, optimizer):
        observable = SummedOp(
            [
                0.2252 * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        ).reduce()

        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        # Define a set of initial parameters
        parameters = ansatz.ordered_parameters

        operator = ~StateFn(observable) @ StateFn(ansatz)
        param_dict = {param: np.pi / 4 for param in parameters}
        param_dict2 = {param: np.pi / 2 for param in parameters}
        state1 = operator[-1].assign_parameters(param_dict).eval().primitive.data
        state2 = operator[-1].assign_parameters(param_dict2).eval().primitive.data

        bures_distance = _calculate_bures_distance(state1, state2, optimizer=optimizer)
        expected_bures_distance = 0.9719181217119478

        np.testing.assert_almost_equal(bures_distance, expected_bures_distance, decimal=6)


if __name__ == "__main__":
    unittest.main()
