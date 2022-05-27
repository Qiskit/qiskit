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
from test.python.algorithms.evolvers.variational.calculators.expected_results.test_evolution_grad_expected1 import (
    correct_values_1,
)
from test.python.algorithms.evolvers.variational.calculators.expected_results.test_evolution_grad_expected2 import (
    correct_values_2,
)
from test.python.algorithms.evolvers.variational.calculators.expected_results.test_evolution_grad_expected3 import (
    correct_values_3,
)
from test.python.algorithms.evolvers.variational.calculators.expected_results.test_evolution_grad_expected4 import (
    correct_values_4,
)
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

    @data(
        (
            SummedOp(
                [
                    0.2252 * (I ^ I),
                    0.5716 * (Z ^ Z),
                    0.3435 * (I ^ Z),
                    -0.4347 * (Z ^ I),
                    0.091 * (Y ^ Y),
                    0.091 * (X ^ X),
                ]
            ),
            Z,
            correct_values_1,
        ),
        (
            0.2252 * (I ^ I)
            + 0.5716 * (Z ^ Z)
            + 0.3435 * (I ^ Z)
            + -0.4347 * (Z ^ I)
            + 0.091 * (Y ^ Y)
            + 0.091 * (X ^ X),
            Z,
            correct_values_1,
        ),
        (
            SummedOp(
                [
                    0.2252 * (I ^ I),
                    0.5716 * (Z ^ Z),
                    0.3435 * (I ^ Z),
                    -0.4347 * (Z ^ I),
                    0.091 * (Y ^ Y),
                    0.091 * (X ^ X),
                ]
            ),
            -Y,
            correct_values_2,
        ),
        (
            0.2252 * (I ^ I)
            + 0.5716 * (Z ^ Z)
            + 0.3435 * (I ^ Z)
            + -0.4347 * (Z ^ I)
            + 0.091 * (Y ^ Y)
            + 0.091 * (X ^ X),
            Z - 1j * Y,
            correct_values_3,
        ),
        (-X, Z - 1j * Y, correct_values_4),
    )
    @unpack
    def test_calculate(self, observable, basis, expected_result):
        """Test calculating evolution gradient."""
        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        grad_method = "lin_comb"
        parameters = list(ansatz.parameters)
        evolution_grad = calculate(observable, ansatz, parameters, grad_method, basis=basis)

        values_dict = [
            {param: np.pi / 4 for param in parameters},
            {param: np.pi / 2 for param in parameters},
        ]
        for i, value_dict in enumerate(values_dict):
            result = evolution_grad.assign_parameters(value_dict).eval()
            np.testing.assert_array_almost_equal(result, expected_result[i])

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
