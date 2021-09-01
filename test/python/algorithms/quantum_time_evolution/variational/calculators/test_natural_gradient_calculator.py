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

from qiskit.algorithms.quantum_time_evolution.variational.calculators.natural_gradient_calculator import (
    calculate,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.imaginary.implementations.imaginary_mc_lachlan_variational_principle import (
    ImaginaryMcLachlanVariationalPrinciple,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z
from test.python.algorithms import QiskitAlgorithmsTestCase


class TestNaturalGradientCalculator(QiskitAlgorithmsTestCase):
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
        params_dict = {param: np.pi / 4 for param in parameters}

        var_principle = ImaginaryMcLachlanVariationalPrinciple(None)
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, parameters)

        correct_values = [
            -0.8842908,
            0.0441611,
            -0.21244606,
            0.2349356,
            -0.50246622,
            -0.6425113,
            0.12545623,
            0.07241851,
            1.01843757,
            0.3669189,
            0.10147791,
            0.18632604,
        ]
        natural_grad = calculate(var_principle, params_dict)

        np.testing.assert_array_almost_equal(
            natural_grad.assign_parameters(params_dict).eval(), correct_values
        )

    def test_calculate_regularized(self):
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
        params_dict = {param: np.pi / 4 for param in parameters}

        var_principle = ImaginaryMcLachlanVariationalPrinciple(None)
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, parameters)

        correct_values = [
            -0.8842908,
            0.0441611,
            -0.21244606,
            0.2349356,
            -0.50246622,
            -0.6425113,
            0.12545623,
            0.07241851,
            1.01843757,
            0.3669189,
            0.10147791,
            0.18632604,
        ]
        natural_grad = calculate(var_principle, params_dict, "ridge")

        np.testing.assert_array_almost_equal(
            natural_grad.assign_parameters(params_dict).eval(), correct_values
        )


if __name__ == "__main__":
    unittest.main()
