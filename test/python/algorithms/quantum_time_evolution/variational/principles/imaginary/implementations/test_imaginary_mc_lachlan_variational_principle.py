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

        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, parameters)
        metric_tensor = var_principle._raw_metric_tensor

        bound_metric_tensor = metric_tensor.bind_parameters(param_dict)

        expected_bound_metric_tensor = [
            [
                2.50000000e-01 + 0.0j,
                1.59600000e-33 + 0.0j,
                5.90075760e-18 + 0.0j,
                -8.49242405e-19 + 0.0j,
                8.83883476e-02 + 0.0j,
                1.33253788e-17 + 0.0j,
                6.25000000e-02 + 0.0j,
                1.40000000e-17 + 0.0j,
                -1.41735435e-01 + 0.0j,
                3.12500000e-02 + 0.0j,
                1.00222087e-01 + 0.0j,
                -3.12500000e-02 + 0.0j,
            ],
            [
                1.59600000e-33 + 0.0j,
                2.50000000e-01 + 0.0j,
                1.34350288e-17 + 0.0j,
                6.43502884e-18 + 0.0j,
                -8.83883476e-02 + 0.0j,
                1.25000000e-01 + 0.0j,
                6.25000000e-02 + 0.0j,
                1.25000000e-01 + 0.0j,
                -8.45970869e-02 + 0.0j,
                7.54441738e-02 + 0.0j,
                1.48207521e-01 + 0.0j,
                2.00444174e-01 + 0.0j,
            ],
            [
                5.90075760e-18 + 0.0j,
                1.34350288e-17 + 0.0j,
                1.25000000e-01 + 0.0j,
                -1.38777878e-17 + 0.0j,
                -4.41941738e-02 + 0.0j,
                6.25000000e-02 + 0.0j,
                1.19638348e-01 + 0.0j,
                6.25000000e-02 + 0.0j,
                -5.14514565e-02 + 0.0j,
                6.89720869e-02 + 0.0j,
                1.04933262e-02 + 0.0j,
                -6.89720869e-02 + 0.0j,
            ],
            [
                -8.49242405e-19 + 0.0j,
                6.43502884e-18 + 0.0j,
                -1.38777878e-17 + 0.0j,
                1.25000000e-01 + 0.0j,
                -4.41941738e-02 + 0.0j,
                -6.25000000e-02 + 0.0j,
                3.12500000e-02 + 0.0j,
                1.25000000e-01 + 0.0j,
                5.14514565e-02 + 0.0j,
                -6.89720869e-02 + 0.0j,
                7.81250000e-03 + 0.0j,
                1.94162607e-02 + 0.0j,
            ],
            [
                8.83883476e-02 + 0.0j,
                -8.83883476e-02 + 0.0j,
                -4.41941738e-02 + 0.0j,
                -4.41941738e-02 + 0.0j,
                2.34375000e-01 + 0.0j,
                -1.10485435e-01 + 0.0j,
                -2.02014565e-02 + 0.0j,
                -4.41941738e-02 + 0.0j,
                1.49547935e-02 + 0.0j,
                -2.24896848e-02 + 0.0j,
                -1.42172278e-03 + 0.0j,
                -1.23822206e-01 + 0.0j,
            ],
            [
                1.33253788e-17 + 0.0j,
                1.25000000e-01 + 0.0j,
                6.25000000e-02 + 0.0j,
                -6.25000000e-02 + 0.0j,
                -1.10485435e-01 + 0.0j,
                2.18750000e-01 + 0.0j,
                -2.68082618e-03 + 0.0j,
                -1.59099026e-17 + 0.0j,
                -1.57197815e-01 + 0.0j,
                2.53331304e-02 + 0.0j,
                9.82311963e-03 + 0.0j,
                1.06138957e-01 + 0.0j,
            ],
            [
                6.25000000e-02 + 0.0j,
                6.25000000e-02 + 0.0j,
                1.19638348e-01 + 0.0j,
                3.12500000e-02 + 0.0j,
                -2.02014565e-02 + 0.0j,
                -2.68082618e-03 + 0.0j,
                2.23881674e-01 + 0.0j,
                1.37944174e-01 + 0.0j,
                -3.78033966e-02 + 0.0j,
                1.58423239e-01 + 0.0j,
                1.34535646e-01 + 0.0j,
                -5.49651086e-02 + 0.0j,
            ],
            [
                1.40000000e-17 + 0.0j,
                1.25000000e-01 + 0.0j,
                6.25000000e-02 + 0.0j,
                1.25000000e-01 + 0.0j,
                -4.41941738e-02 + 0.0j,
                -1.59099026e-17 + 0.0j,
                1.37944174e-01 + 0.0j,
                2.50000000e-01 + 0.0j,
                -2.10523539e-17 + 0.0j,
                1.15574269e-17 + 0.0j,
                9.75412607e-02 + 0.0j,
                5.71383476e-02 + 0.0j,
            ],
            [
                -1.41735435e-01 + 0.0j,
                -8.45970869e-02 + 0.0j,
                -5.14514565e-02 + 0.0j,
                5.14514565e-02 + 0.0j,
                1.49547935e-02 + 0.0j,
                -1.57197815e-01 + 0.0j,
                -3.78033966e-02 + 0.0j,
                -2.10523539e-17 + 0.0j,
                1.95283753e-01 + 0.0j,
                -3.82941440e-02 + 0.0j,
                -6.11392595e-02 + 0.0j,
                -4.51588288e-02 + 0.0j,
            ],
            [
                3.12500000e-02 + 0.0j,
                7.54441738e-02 + 0.0j,
                6.89720869e-02 + 0.0j,
                -6.89720869e-02 + 0.0j,
                -2.24896848e-02 + 0.0j,
                2.53331304e-02 + 0.0j,
                1.58423239e-01 + 0.0j,
                1.15574269e-17 + 0.0j,
                -3.82941440e-02 + 0.0j,
                2.17629701e-01 + 0.0j,
                1.32431810e-01 + 0.0j,
                -1.91961467e-02 + 0.0j,
            ],
            [
                1.00222087e-01 + 0.0j,
                1.48207521e-01 + 0.0j,
                1.04933262e-02 + 0.0j,
                7.81250000e-03 + 0.0j,
                -1.42172278e-03 + 0.0j,
                9.82311963e-03 + 0.0j,
                1.34535646e-01 + 0.0j,
                9.75412607e-02 + 0.0j,
                -6.11392595e-02 + 0.0j,
                1.32431810e-01 + 0.0j,
                1.81683746e-01 + 0.0j,
                7.28902444e-02 + 0.0j,
            ],
            [
                -3.12500000e-02 + 0.0j,
                2.00444174e-01 + 0.0j,
                -6.89720869e-02 + 0.0j,
                1.94162607e-02 + 0.0j,
                -1.23822206e-01 + 0.0j,
                1.06138957e-01 + 0.0j,
                -5.49651086e-02 + 0.0j,
                5.71383476e-02 + 0.0j,
                -4.51588288e-02 + 0.0j,
                -1.91961467e-02 + 0.0j,
                7.28902444e-02 + 0.0j,
                2.38616353e-01 + 0.0j,
            ],
        ]

        np.testing.assert_almost_equal(bound_metric_tensor.eval(), expected_bound_metric_tensor)

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

        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, parameters)

        evolution_grad = var_principle._raw_evolution_grad

        bound_raw_evolution_grad = evolution_grad.bind_parameters(param_dict)

        expected_bound_evolution_grad = [
            (0.19308934095957098 - 1.4e-17j),
            (0.007027674650099142 - 0j),
            (0.03192524520091862 - 0j),
            (-0.06810314606309673 - 1e-18j),
            (0.07590371669521798 - 7e-18j),
            (0.11891968269385343 + 1.5e-18j),
            (-0.0012030273438232639 + 0j),
            (-0.049885258804562266 + 1.8500000000000002e-17j),
            (-0.20178860797540302 - 5e-19j),
            (-0.0052269232310933195 + 1e-18j),
            (0.022892905637005266 - 3e-18j),
            (-0.022892905637005294 + 3.5e-18j),
        ]

        np.testing.assert_almost_equal(
            bound_raw_evolution_grad.eval(), expected_bound_evolution_grad
        )


if __name__ == "__main__":
    unittest.main()
