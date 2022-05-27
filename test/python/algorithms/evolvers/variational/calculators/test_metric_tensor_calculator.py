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

"""Test metric tensor calculator."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from test.python.algorithms.evolvers.variational.calculators.expected_results.test_metric_calculator_expected1 import (
    correct_values_1,
)
from test.python.algorithms.evolvers.variational.calculators.expected_results.test_metric_calculator_expected2 import (
    correct_values_2,
)
from ddt import unpack, data, ddt
import numpy as np
from qiskit.algorithms.evolvers.variational.calculators.metric_tensor_calculator import (
    calculate,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z
from test.python.algorithms.evolvers.variational.calculators.expected_results.test_metric_calculator_expected3 import (
    correct_values_3,
)
from test.python.algorithms.evolvers.variational.calculators.expected_results.test_metric_calculator_expected4 import (
    correct_values_4,
)


@ddt
class TestMetricTensorCalculator(QiskitAlgorithmsTestCase):
    """Test metric tensor calculator."""

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
            -Y,
            correct_values_2,
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
        (X, -Y, correct_values_3),
        (5 * Z, -Y, correct_values_4),
    )
    @unpack
    def test_calculate_imaginary(self, observable, basis, expected_result):
        """Test calculating the imaginary part of a metric tensor."""
        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)
        parameters = list(ansatz.parameters)
        metric_tensor = calculate(ansatz, parameters, basis=basis)

        values_dict = [
            {param: np.pi / 4 for param in parameters},
            {param: np.pi / 2 for param in parameters},
        ]

        for i, value_dict in enumerate(values_dict):
            result = metric_tensor.assign_parameters(value_dict).eval()
            np.testing.assert_array_almost_equal(result, expected_result[i])

    @data(
        ("param_shift", Z, True),
        ("circuit_qfi", -Y, True),
        ("circuit_qfi", Z - 1j * Y, True),
        ("circuit_qfi", Z, False),
        ("circuit_qfi", Z, False),
    )
    @unpack
    def test_calculate_with_errors(self, qfi_method, basis, phase_fix):
        """Test calculating metric tensor when errors expected."""
        observable = 0.2252 * (I ^ I)

        d = 1
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        with self.assertRaises(ValueError):
            _ = calculate(ansatz, list(ansatz.parameters), qfi_method, basis, phase_fix)


if __name__ == "__main__":
    unittest.main()
