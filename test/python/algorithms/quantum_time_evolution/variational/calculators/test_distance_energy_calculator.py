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
from ddt import data, ddt
from mpmath import expm

from qiskit.algorithms.quantum_time_evolution.variational.calculators.distance_energy_calculator import (
    _calculate_distance_energy,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z, StateFn
from test.python.algorithms import QiskitAlgorithmsTestCase


@ddt
class TestDistanceEnergyCalculator(QiskitAlgorithmsTestCase):

    # TODO address ValueError
    @data("cobyla", "nelder-mead", "l-bfgs-b")
    def test__calculate_distance_energy_same_state(self, optimizer):
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
        time = 3
        param_dict = {param: np.pi / 4 for param in parameters}
        state1 = operator[-1]

        h = operator.oplist[0].primitive * operator.oplist[0].coeff
        h_matrix = h.to_matrix(massive=True)

        init_state = state1.assign_parameters(param_dict).eval().primitive.data
        exact_state = np.dot(expm(-1j * h_matrix * time), init_state)

        distance_energy = _calculate_distance_energy(state1, exact_state, h_matrix, param_dict)
        expected_distance_energy = 0
        print(distance_energy)
        # np.testing.assert_almost_equal(distance_energy, expected_bures_distance, decimal=6)


if __name__ == "__main__":
    unittest.main()
