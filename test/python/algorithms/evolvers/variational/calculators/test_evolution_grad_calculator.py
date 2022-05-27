# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test evolution gradient calculator."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import data, unpack, ddt
import numpy as np

from qiskit.algorithms.evolvers.variational.calculators.evolution_grad_calculator import (
    calculate,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z


@ddt
class TestEvolutionGradCalculator(QiskitAlgorithmsTestCase):
    """Test evolution gradient calculator."""

    def test_calculate(self):
        """Test calculating evolution gradient."""
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

        grad_method = "lin_comb"
        parameters = list(ansatz.parameters)
        evolution_grad = calculate(observable, ansatz, parameters, grad_method)

        values_dict = [
            {param: np.pi / 4 for param in parameters},
            {param: np.pi / 2 for param in parameters},
        ]

        correct_values = [
            [
                (-0.38617868191914206 + 0j),
                (-0.014055349300198328 + 0j),
                (-0.06385049040183735 + 0j),
                (0.13620629212619334 + 0j),
                (-0.15180743339043595 + 0j),
                (-0.23783936538770686 + 0j),
                (0.002406054687646396 + 0j),
                (0.09977051760912457 + 0j),
                (0.40357721595080603 + 0j),
                (0.010453846462186639 + 0j),
                (-0.0457858112740105 + 0j),
                (0.04578581127401066 + 0j),
            ],
            [
                (0.43469999999999975 + 0j),
                (-9.958339999999998e-17 + 0j),
                (2.286804e-16 + 0j),
                (0.6625999999999991 + 0j),
                (1.038073e-16 + 0j),
                (-3.0849399999999995e-17 + 0j),
                (-0.34349999999999986 + 0j),
                (-1.4965850000000002e-16 + 0j),
                (1.180053e-16 + 0j),
                (-1.0798280000000001e-16 + 0j),
                (-7.76323e-17 + 0j),
                (-6.83375e-17 + 0j),
            ],
        ]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                evolution_grad.assign_parameters(value_dict).eval(), correct_values[i]
            )

    @data(
        (
            -Y,
            [
                [
                    (0.09029822949045127 + 1e-17j),
                    (-0.19262478560541488 - 9e-18j),
                    (-0.6510746098262302 + 6e-18j),
                    (-0.3879436778936384 - 6e-18j),
                    (-0.12595729749408652 - 2.1e-17j),
                    (-0.3297422949956381 + 1e-18j),
                    (-0.5040008514312755 + 4e-18j),
                    (-0.33348367536103796 - 9e-18j),
                    (-0.17322660907305487 + 1.9e-17j),
                    (-0.37018200464796197 + 1e-18j),
                    (-0.4079639544930501 - 3e-18j),
                    (-0.05062557967159678 + 5e-18j),
                ],
                [
                    (3.6e-17 + 7e-18j),
                    (-0.6625999999999996 - 2e-18j),
                    (0.43469999999999953 + 3e-18j),
                    0j,
                    (0.6625999999999996 - 9e-18j),
                    (0.43469999999999975 + 1.4e-17j),
                    (-2.5e-17 - 7e-18j),
                    (-0.6625999999999992 + 3e-18j),
                    (0.13419999999999993 + 9e-18j),
                    (-0.13419999999999993 + 1.1e-17j),
                    (-0.34349999999999975 - 9e-18j),
                    (0.43469999999999953 - 1.1e-17j),
                ],
            ],
        ),
        (
            Z - 1j * Y,
            [
                [
                    (-0.38617868191914206 + 0.09029822949045127j),
                    (-0.014055349300198292 - 0.19262478560541474j),
                    (-0.06385049040183727 - 0.6510746098262302j),
                    (0.13620629212619334 - 0.3879436778936384j),
                    (-0.151807433390436 - 0.12595729749408652j),
                    (-0.23783936538770684 - 0.3297422949956381j),
                    (0.00240605468764643 - 0.5040008514312755j),
                    (0.09977051760912456 - 0.33348367536103796j),
                    (0.403577215950806 - 0.17322660907305487j),
                    (0.010453846462186688 - 0.370182004647962j),
                    (-0.04578581127401051 - 0.40796395449305006j),
                    (0.04578581127401063 - 0.05062557967159678j),
                ],
                [
                    (0.43469999999999975 + 3.6e-17j),
                    (-2.1e-16 - 0.6625999999999996j),
                    (2.25e-16 + 0.4346999999999995j),
                    (0.6625999999999991 + 0j),
                    (1.05e-16 + 0.6625999999999996j),
                    (-3.3e-17 + 0.43469999999999975j),
                    (-0.3434999999999998 - 2.5e-17j),
                    (-1.58e-16 - 0.6625999999999992j),
                    (1.02e-16 + 0.13419999999999993j),
                    (-1.2e-16 - 0.13419999999999999j),
                    (-4.2e-17 - 0.34349999999999975j),
                    (-2.9e-17 + 0.4346999999999995j),
                ],
            ],
        ),
    )
    @unpack  # TODO verify if values correct
    def test_calculate_bases(self, basis, correct_values):
        """Test calculating evolution gradient with non-default bases."""
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

        grad_method = "lin_comb"
        parameters = list(ansatz.parameters)
        evolution_grad = calculate(observable, ansatz, parameters, grad_method, basis)

        values_dict = [
            {param: np.pi / 4 for param in parameters},
            {param: np.pi / 2 for param in parameters},
        ]

        for i, value_dict in enumerate(values_dict):
            np.testing.assert_array_almost_equal(
                evolution_grad.assign_parameters(value_dict).eval(), correct_values[i]
            )

    @data(
        ("param_shift", -Y),
        ("fin_diff", -Y),
        ("param_shift", Z - 1j * Y),
        ("fin_diff", Z - 1j * Y),
        ("lin_comb_full", Z),
    )
    @unpack
    def test_calculate_with_errors(self, grad_method, basis):
        """Test calculating evolution gradient when errors expected."""
        observable = 0.2252 * (I ^ I)

        d = 1
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        with self.assertRaises(ValueError):
            _ = calculate(observable, ansatz, list(ansatz.parameters), grad_method, basis)


if __name__ == "__main__":
    unittest.main()
