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
    # checked, correct
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
        var_principle = ImaginaryMcLachlanVariationalPrinciple()
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, parameters)

        correct_values = [
            0.442145,
            -0.022081,
            0.106223,
            -0.117468,
            0.251233,
            0.321256,
            -0.062728,
            -0.036209,
            -0.509219,
            -0.183459,
            -0.050739,
            -0.093163,
        ]
        natural_grad = calculate(-var_principle._operator, parameters)
        natural_grad_bound = natural_grad.assign_parameters(params_dict).eval()

        np.testing.assert_array_almost_equal(natural_grad_bound, correct_values)

    # result slightly different than above, due to a regularization
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
        regularization = "ridge"

        var_principle = ImaginaryMcLachlanVariationalPrinciple()
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, parameters)

        correct_values = [
            0.44209285,
            -0.02368672,
            0.10331097,
            -0.11583335,
            0.24387236,
            0.31642126,
            -0.06179241,
            -0.04077569,
            -0.50458966,
            -0.17833621,
            -0.04812563,
            -0.09106516,
        ]
        natural_grad = calculate(
            -var_principle._operator, parameters, regularization=regularization
        )
        natural_grad_bound = natural_grad.assign_parameters(params_dict).eval()

        np.testing.assert_array_almost_equal(natural_grad_bound, correct_values)


if __name__ == "__main__":
    unittest.main()
