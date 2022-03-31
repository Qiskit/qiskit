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

"""Test real time dependent variational principle."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np
from qiskit.algorithms.evolvers.variational.variational_principles.real.implementations.real_time_dependent_variational_principle import (
    RealTimeDependentVariationalPrinciple,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z


class TestRealTimeDependentVariationalPrinciple(QiskitAlgorithmsTestCase):
    """Test real time dependent variational principle."""

    def test_calc_calc_metric_tensor(self):
        """Test calculating a metric tensor."""
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

        metric_tensor = var_principle.calc_metric_tensor(ansatz, parameters)

        bound_metric_tensor = metric_tensor.bind_parameters(param_dict)
        expected_bound_metric_tensor = [
            [
                -1.21000000e-34 + 0.00e00j,
                1.21000000e-34 + 2.50e-19j,
                1.76776695e-01 - 1.00e-18j,
                -1.40000000e-17 + 0.00e00j,
                -6.25000000e-02 + 0.00e00j,
                8.83883476e-02 - 1.25e-18j,
                1.69194174e-01 + 2.25e-18j,
                8.83883476e-02 - 2.50e-19j,
                -7.27633476e-02 + 0.00e00j,
                9.75412607e-02 + 7.50e-19j,
                1.48398042e-02 - 1.75e-18j,
                -9.75412607e-02 + 3.75e-18j,
            ],
            [
                1.21000000e-34 + 2.50e-19j,
                -1.21000000e-34 + 0.00e00j,
                1.10000000e-34 + 2.75e-18j,
                1.76776695e-01 - 2.25e-18j,
                -6.25000000e-02 + 0.00e00j,
                -8.83883476e-02 + 4.00e-18j,
                4.41941738e-02 - 1.25e-18j,
                1.76776695e-01 - 2.50e-19j,
                7.27633476e-02 - 7.50e-19j,
                -9.75412607e-02 - 7.50e-19j,
                1.10485435e-02 - 7.50e-19j,
                2.74587393e-02 + 2.50e-19j,
            ],
            [
                1.76776695e-01 - 1.00e-18j,
                1.10000000e-34 + 2.75e-18j,
                -1.25000000e-01 + 0.00e00j,
                -1.25000000e-01 + 0.00e00j,
                -1.06694174e-01 + 1.25e-18j,
                -6.25000000e-02 + 1.75e-18j,
                -1.01332521e-01 + 7.50e-19j,
                4.67500000e-17 - 7.50e-19j,
                1.75206304e-02 + 5.00e-19j,
                -8.57075215e-02 - 1.00e-18j,
                -1.63277304e-01 + 1.00e-18j,
                -1.56250000e-02 + 0.00e00j,
            ],
            [
                -1.40000000e-17 + 0.00e00j,
                1.76776695e-01 - 2.25e-18j,
                -1.25000000e-01 + 0.00e00j,
                -1.25000000e-01 + 0.00e00j,
                1.83058262e-02 - 1.50e-18j,
                -1.50888348e-01 - 1.50e-18j,
                -1.01332521e-01 + 2.50e-19j,
                -8.83883476e-02 - 1.00e-18j,
                -2.28822827e-02 - 1.00e-18j,
                -1.16957521e-01 + 1.00e-18j,
                -1.97208130e-01 + 0.00e00j,
                -1.79457521e-01 + 1.25e-18j,
            ],
            [
                -6.25000000e-02 + 0.00e00j,
                -6.25000000e-02 + 0.00e00j,
                -1.06694174e-01 + 1.25e-18j,
                1.83058262e-02 - 1.50e-18j,
                -1.56250000e-02 + 0.00e00j,
                -2.20970869e-02 - 2.00e-18j,
                1.48992717e-01 - 1.00e-18j,
                2.60000000e-17 - 1.50e-18j,
                -6.69614673e-02 - 5.00e-19j,
                2.00051576e-01 + 5.00e-19j,
                1.13640168e-01 + 1.25e-18j,
                -4.83780325e-02 - 1.00e-18j,
            ],
            [
                8.83883476e-02 - 1.25e-18j,
                -8.83883476e-02 + 4.00e-18j,
                -6.25000000e-02 + 1.75e-18j,
                -1.50888348e-01 - 1.50e-18j,
                -2.20970869e-02 - 2.00e-18j,
                -3.12500000e-02 + 0.00e00j,
                -2.85691738e-02 + 4.25e-18j,
                1.76776695e-01 + 0.00e00j,
                5.52427173e-03 + 1.00e-18j,
                -1.29346478e-01 + 5.00e-19j,
                -4.81004238e-02 + 4.25e-18j,
                5.27918696e-02 + 2.50e-19j,
            ],
            [
                1.69194174e-01 + 2.25e-18j,
                4.41941738e-02 - 1.25e-18j,
                -1.01332521e-01 + 7.50e-19j,
                -1.01332521e-01 + 2.50e-19j,
                1.48992717e-01 - 1.00e-18j,
                -2.85691738e-02 + 4.25e-18j,
                -2.61183262e-02 + 0.00e00j,
                -6.88900000e-33 + 0.00e00j,
                6.62099510e-02 - 1.00e-18j,
                -2.90767610e-02 + 1.75e-18j,
                -1.24942505e-01 + 0.00e00j,
                -1.72430217e-02 + 2.50e-19j,
            ],
            [
                8.83883476e-02 - 2.50e-19j,
                1.76776695e-01 - 2.50e-19j,
                4.67500000e-17 - 7.50e-19j,
                -8.83883476e-02 - 1.00e-18j,
                2.60000000e-17 - 1.50e-18j,
                1.76776695e-01 + 0.00e00j,
                -6.88900000e-33 + 0.00e00j,
                -6.88900000e-33 + 0.00e00j,
                1.79457521e-01 - 1.75e-18j,
                -5.33470869e-02 + 2.00e-18j,
                -9.56456304e-02 + 3.00e-18j,
                -1.32582521e-01 + 2.50e-19j,
            ],
            [
                -7.27633476e-02 + 0.00e00j,
                7.27633476e-02 - 7.50e-19j,
                1.75206304e-02 + 5.00e-19j,
                -2.28822827e-02 - 1.00e-18j,
                -6.69614673e-02 - 5.00e-19j,
                5.52427173e-03 + 1.00e-18j,
                6.62099510e-02 - 1.00e-18j,
                1.79457521e-01 - 1.75e-18j,
                -5.47162473e-02 + 0.00e00j,
                -4.20854047e-02 + 4.00e-18j,
                -7.75494553e-02 - 2.50e-18j,
                -2.49573723e-02 + 7.50e-19j,
            ],
            [
                9.75412607e-02 + 7.50e-19j,
                -9.75412607e-02 - 7.50e-19j,
                -8.57075215e-02 - 1.00e-18j,
                -1.16957521e-01 + 1.00e-18j,
                2.00051576e-01 + 5.00e-19j,
                -1.29346478e-01 + 5.00e-19j,
                -2.90767610e-02 + 1.75e-18j,
                -5.33470869e-02 + 2.00e-18j,
                -4.20854047e-02 + 4.00e-18j,
                -3.23702991e-02 + 0.00e00j,
                -4.70257118e-02 + 0.00e00j,
                1.22539288e-01 - 2.25e-18j,
            ],
            [
                1.48398042e-02 - 1.75e-18j,
                1.10485435e-02 - 7.50e-19j,
                -1.63277304e-01 + 1.00e-18j,
                -1.97208130e-01 + 0.00e00j,
                1.13640168e-01 + 1.25e-18j,
                -4.81004238e-02 + 4.25e-18j,
                -1.24942505e-01 + 0.00e00j,
                -9.56456304e-02 + 3.00e-18j,
                -7.75494553e-02 - 2.50e-18j,
                -4.70257118e-02 + 0.00e00j,
                -6.83162540e-02 + 0.00e00j,
                -2.78870598e-02 + 0.00e00j,
            ],
            [
                -9.75412607e-02 + 3.75e-18j,
                2.74587393e-02 + 2.50e-19j,
                -1.56250000e-02 + 0.00e00j,
                -1.79457521e-01 + 1.25e-18j,
                -4.83780325e-02 - 1.00e-18j,
                5.27918696e-02 + 2.50e-19j,
                -1.72430217e-02 + 2.50e-19j,
                -1.32582521e-01 + 2.50e-19j,
                -2.49573723e-02 + 7.50e-19j,
                1.22539288e-01 - 2.25e-18j,
                -2.78870598e-02 + 0.00e00j,
                -1.13836467e-02 + 0.00e00j,
            ],
        ]
        np.testing.assert_almost_equal(
            bound_metric_tensor.eval(), expected_bound_metric_tensor, decimal=5
        )

    def test_calc_calc_evolution_grad(self):
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

        # Define a set of initial parameters
        parameters = ansatz.ordered_parameters
        param_dict = {param: np.pi / 4 for param in parameters}
        var_principle = RealTimeDependentVariationalPrinciple()

        evolution_grad = var_principle.calc_evolution_grad(observable, ansatz, parameters)

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
