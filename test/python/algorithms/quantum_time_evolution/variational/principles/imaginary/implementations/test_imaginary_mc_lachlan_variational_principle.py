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

from qiskit.algorithms.quantum_time_evolution.variational.calculators import (
    metric_tensor_calculator,
    evolution_grad_calculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.imaginary.implementations.imaginary_mc_lachlan_variational_principle import (
    ImaginaryMcLachlanVariationalPrinciple,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z
from test.python.algorithms import QiskitAlgorithmsTestCase


class TestImaginaryMcLachlanVariationalPrinciple(QiskitAlgorithmsTestCase):
    def test_calc_calc_metric_tensor(self):
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
        param_dict = {param: np.pi / 4 for param in parameters}
        var_principle = ImaginaryMcLachlanVariationalPrinciple()
        regularization = "ridge"
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, param_dict, regularization)

        raw_metric_tensor = metric_tensor_calculator.calculate(
            ansatz, parameters, var_principle._qfi_method
        )
        metric_tensor = var_principle.metric_tensor

        bound_raw_metric_tensor = raw_metric_tensor.bind_parameters(param_dict)
        expected_metric_tensor = bound_raw_metric_tensor / 4.0

        np.testing.assert_almost_equal(
            metric_tensor.eval(), expected_metric_tensor.eval()
        )

    def test_calc_calc_evolution_grad(self):
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
        param_dict = {param: np.pi / 4 for param in parameters}
        var_principle = ImaginaryMcLachlanVariationalPrinciple()
        regularization = "ridge"
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, param_dict, regularization)

        raw_evolution_grad = evolution_grad_calculator.calculate(
            observable, ansatz, parameters, var_principle._grad_method
        )
        evolution_grad = var_principle.evolution_grad

        bound_raw_evolution_grad = raw_evolution_grad.bind_parameters(param_dict)
        expected_evolution_grad = -bound_raw_evolution_grad

        np.testing.assert_almost_equal(
            evolution_grad.eval(), expected_evolution_grad.eval()
        )


if __name__ == "__main__":
    unittest.main()
