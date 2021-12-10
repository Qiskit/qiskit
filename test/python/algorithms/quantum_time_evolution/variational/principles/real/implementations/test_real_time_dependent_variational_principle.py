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

"""Test real time dependent variational principle."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np
from qiskit.algorithms.quantum_time_evolution.variational.principles.real.implementations.real_time_dependent_variational_principle import (
    RealTimeDependentVariationalPrinciple,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z


class TestRealTimeDependentVariationalPrinciple(QiskitAlgorithmsTestCase):
    """Test real time dependent variational principle."""

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
        var_principle = RealTimeDependentVariationalPrinciple()

        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, parameters)

        metric_tensor = var_principle._raw_metric_tensor

        bound_metric_tensor = metric_tensor.bind_parameters(param_dict)

        expected_bound_metric_tensor = [
            [
                -1.76400000e-33 + 0.00000000e00j,
                1.59600000e-33 + 0.00000000e00j,
                -1.48492424e-17 + 1.76776695e-01j,
                -1.48492424e-17 - 1.40000000e-17j,
                -5.25000000e-18 - 6.25000000e-02j,
                -7.42462120e-18 + 8.83883476e-02j,
                -6.78768940e-18 + 1.69194174e-01j,
                -3.78000000e-33 + 8.83883476e-02j,
                -9.82443180e-18 - 7.27633476e-02j,
                -7.55653410e-18 + 9.75412607e-02j,
                -1.09776989e-17 + 1.48398042e-02j,
                -4.48115530e-18 - 9.75412607e-02j,
            ],
            [
                1.59600000e-33 + 0.00000000e00j,
                -1.44400000e-33 + 0.00000000e00j,
                1.34350288e-17 + 0.00000000e00j,
                1.34350288e-17 + 1.76776695e-01j,
                4.75000000e-18 - 6.25000000e-02j,
                6.71751442e-18 - 8.83883476e-02j,
                6.14124279e-18 + 4.41941738e-02j,
                3.42000000e-33 + 1.76776695e-01j,
                8.88877163e-18 + 7.27633476e-02j,
                6.83686418e-18 - 9.75412607e-02j,
                9.93220372e-18 + 1.10485435e-02j,
                4.05437861e-18 + 2.74587393e-02j,
            ],
            [
                -1.48492424e-17 + 1.76776695e-01j,
                1.34350288e-17 + 0.00000000e00j,
                -1.25000000e-01 + 0.00000000e00j,
                -1.25000000e-01 + 0.00000000e00j,
                -4.41941738e-02 - 6.25000000e-02j,
                -6.25000000e-02 + 7.00000000e-18j,
                -5.71383476e-02 - 4.41941738e-02j,
                -3.18198052e-17 + 4.17500000e-17j,
                -8.27014565e-02 + 1.00222087e-01j,
                -6.36104346e-02 - 2.20970869e-02j,
                -9.24095869e-02 - 7.08677173e-02j,
                -3.77220869e-02 + 2.20970869e-02j,
            ],
            [
                -1.48492424e-17 - 1.40000000e-17j,
                1.34350288e-17 + 1.76776695e-01j,
                -1.25000000e-01 + 0.00000000e00j,
                -1.25000000e-01 + 0.00000000e00j,
                -4.41941738e-02 + 6.25000000e-02j,
                -6.25000000e-02 - 8.83883476e-02j,
                -5.71383476e-02 - 4.41941738e-02j,
                -3.18198052e-17 - 8.83883476e-02j,
                -8.27014565e-02 + 5.98191738e-02j,
                -6.36104346e-02 - 5.33470869e-02j,
                -9.24095869e-02 - 1.04798543e-01j,
                -3.77220869e-02 - 1.41735435e-01j,
            ],
            [
                -5.25000000e-18 - 6.25000000e-02j,
                4.75000000e-18 - 6.25000000e-02j,
                -4.41941738e-02 - 6.25000000e-02j,
                -4.41941738e-02 + 6.25000000e-02j,
                -1.56250000e-02 + 0.00000000e00j,
                -2.20970869e-02 + 1.40000000e-17j,
                -2.02014565e-02 + 1.69194174e-01j,
                -1.12500000e-17 + 2.07500000e-17j,
                -2.92393804e-02 - 3.77220869e-02j,
                -2.24896848e-02 + 2.22541261e-01j,
                -3.26717228e-02 + 1.46311891e-01j,
                -1.33367717e-02 - 3.50412607e-02j,
            ],
            [
                -7.42462120e-18 + 8.83883476e-02j,
                6.71751442e-18 - 8.83883476e-02j,
                -6.25000000e-02 + 7.00000000e-18j,
                -6.25000000e-02 - 8.83883476e-02j,
                -2.20970869e-02 + 1.40000000e-17j,
                -3.12500000e-02 + 0.00000000e00j,
                -2.85691738e-02 + 1.40000000e-17j,
                -1.59099026e-17 + 1.76776695e-01j,
                -4.13507283e-02 + 4.68750000e-02j,
                -3.18052173e-02 - 9.75412607e-02j,
                -4.62047935e-02 - 1.89563037e-03j,
                -1.88610435e-02 + 7.16529131e-02j,
            ],
            [
                -6.78768940e-18 + 1.69194174e-01j,
                6.14124279e-18 + 4.41941738e-02j,
                -5.71383476e-02 - 4.41941738e-02j,
                -5.71383476e-02 - 4.41941738e-02j,
                -2.02014565e-02 + 1.69194174e-01j,
                -2.85691738e-02 + 1.40000000e-17j,
                -2.61183262e-02 + 0.00000000e00j,
                -1.45450487e-17 + 0.00000000e00j,
                -3.78033966e-02 + 1.04013348e-01j,
                -2.90767610e-02 + 1.72500000e-17j,
                -4.22410488e-02 - 8.27014565e-02j,
                -1.72430217e-02 - 1.55000000e-17j,
            ],
            [
                -3.78000000e-33 + 8.83883476e-02j,
                3.42000000e-33 + 1.76776695e-01j,
                -3.18198052e-17 + 4.17500000e-17j,
                -3.18198052e-17 - 8.83883476e-02j,
                -1.12500000e-17 + 2.07500000e-17j,
                -1.59099026e-17 + 1.76776695e-01j,
                -1.45450487e-17 + 0.00000000e00j,
                -9.32500000e-33 + 0.00000000e00j,
                -2.10523539e-17 + 1.79457521e-01j,
                -1.61925731e-17 - 5.33470869e-02j,
                -2.35236404e-17 - 9.56456304e-02j,
                -9.60247564e-18 - 1.32582521e-01j,
            ],
            [
                -9.82443180e-18 - 7.27633476e-02j,
                8.88877163e-18 + 7.27633476e-02j,
                -8.27014565e-02 + 1.00222087e-01j,
                -8.27014565e-02 + 5.98191738e-02j,
                -2.92393804e-02 - 3.77220869e-02j,
                -4.13507283e-02 + 4.68750000e-02j,
                -3.78033966e-02 + 1.04013348e-01j,
                -2.10523539e-17 + 1.79457521e-01j,
                -5.47162473e-02 + 0.00000000e00j,
                -4.20854047e-02 - 2.77500000e-17j,
                -6.11392595e-02 - 1.64101958e-02j,
                -2.49573723e-02 - 2.77500000e-17j,
            ],
            [
                -7.55653410e-18 + 9.75412607e-02j,
                6.83686418e-18 - 9.75412607e-02j,
                -6.36104346e-02 - 2.20970869e-02j,
                -6.36104346e-02 - 5.33470869e-02j,
                -2.24896848e-02 + 2.22541261e-01j,
                -3.18052173e-02 - 9.75412607e-02j,
                -2.90767610e-02 + 1.72500000e-17j,
                -1.61925731e-17 - 5.33470869e-02j,
                -4.20854047e-02 - 2.77500000e-17j,
                -3.23702991e-02 + 0.00000000e00j,
                -4.70257118e-02 - 1.05000000e-17j,
                -1.91961467e-02 + 1.41735435e-01j,
            ],
            [
                -1.09776989e-17 + 1.48398042e-02j,
                9.93220372e-18 + 1.10485435e-02j,
                -9.24095869e-02 - 7.08677173e-02j,
                -9.24095869e-02 - 1.04798543e-01j,
                -3.26717228e-02 + 1.46311891e-01j,
                -4.62047935e-02 - 1.89563037e-03j,
                -4.22410488e-02 - 8.27014565e-02j,
                -2.35236404e-17 - 9.56456304e-02j,
                -6.11392595e-02 - 1.64101958e-02j,
                -4.70257118e-02 - 1.05000000e-17j,
                -6.83162540e-02 + 0.00000000e00j,
                -2.78870598e-02 + 0.00000000e00j,
            ],
            [
                -4.48115530e-18 - 9.75412607e-02j,
                4.05437861e-18 + 2.74587393e-02j,
                -3.77220869e-02 + 2.20970869e-02j,
                -3.77220869e-02 - 1.41735435e-01j,
                -1.33367717e-02 - 3.50412607e-02j,
                -1.88610435e-02 + 7.16529131e-02j,
                -1.72430217e-02 - 1.55000000e-17j,
                -9.60247564e-18 - 1.32582521e-01j,
                -2.49573723e-02 - 2.77500000e-17j,
                -1.91961467e-02 + 1.41735435e-01j,
                -2.78870598e-02 + 0.00000000e00j,
                -1.13836467e-02 + 0.00000000e00j,
            ],
        ]
        np.testing.assert_almost_equal(
            bound_metric_tensor.eval(), expected_bound_metric_tensor, decimal=5
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
        var_principle = RealTimeDependentVariationalPrinciple()

        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, parameters)

        evolution_grad = var_principle._raw_evolution_grad

        bound_raw_evolution_grad = evolution_grad.bind_parameters(param_dict)

        expected_bound_evolution_grad = [
            (-0.19308934095957098 + 1.4e-17j),
            (-0.007027674650099142 + 0j),
            (-0.03192524520091862 + 0j),
            (0.06810314606309673 + 1e-18j),
            (-0.07590371669521798 + 7e-18j),
            (-0.11891968269385343 - 1.5e-18j),
            (0.0012030273438232639 + 0j),
            (0.049885258804562266 - 1.8500000000000002e-17j),
            (0.20178860797540302 + 5e-19j),
            (0.0052269232310933195 - 1e-18j),
            (-0.022892905637005266 + 3e-18j),
            (0.022892905637005294 - 3.5e-18j),
        ]

        np.testing.assert_almost_equal(
            bound_raw_evolution_grad.eval(), expected_bound_evolution_grad, decimal=5
        )


if __name__ == "__main__":
    unittest.main()
