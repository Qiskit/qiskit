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

from qiskit.algorithms.quantum_time_evolution.variational.calculators.evolution_grad_calculator import (
    calculate,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z
from test.python.algorithms import QiskitAlgorithmsTestCase


class TestEvolutionGradCalculator(QiskitAlgorithmsTestCase):
    def test_calculate(self):
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
        evolution_grad = calculate(observable, ansatz, parameters)

        values_dict = [
            {param: np.pi / 4 for param in parameters},
            {param: np.pi / 2 for param in parameters},
        ]

        correct_values = [
            [
                (-0.38617868191914206 + 0j),
                (-0.014055349300198364 + 0j),
                (-0.06385049040183734 + 0j),
                (0.13620629212619334 + 0j),
                (-0.15180743339043595 + 0j),
                (-0.2378393653877069 + 0j),
                (0.0024060546876464237 + 0j),
                (0.09977051760912459 + 0j),
                (0.40357721595080603 + 0j),
                (0.010453846462186653 + 0j),
                (-0.04578581127401049 + 0j),
                (0.04578581127401063 + 0j),
            ],
            [
                (0.4346999999999997 + 0j),
                (-1.506822e-16 + 0j),
                (2.382098e-16 + 0j),
                (0.6625999999999991 + 0j),
                (1.2198439999999998e-16 + 0j),
                (-3.84024e-17 + 0j),
                (-0.34349999999999986 + 0j),
                (-1.5190600000000002e-16 + 0j),
                (1.314352e-16 + 0j),
                (-1.1215120000000001e-16 + 0j),
                (-4.69858e-17 + 0j),
                (-8.13302e-17 + 0j),
            ],
        ]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                evolution_grad.assign_parameters(value_dict).eval(), correct_values[i]
            )


if __name__ == "__main__":
    unittest.main()
