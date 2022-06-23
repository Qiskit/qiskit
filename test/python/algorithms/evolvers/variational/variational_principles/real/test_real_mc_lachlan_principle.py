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

"""Test real McLachlan's variational principle."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np
from qiskit.algorithms.evolvers.variational import (
    RealMcLachlanPrinciple,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z
from ..expected_results.test_imaginary_mc_lachlan_variational_principle_expected2 import (
    expected_bound_metric_tensor_2,
)


class TestRealMcLachlanPrinciple(QiskitAlgorithmsTestCase):
    """Test real McLachlan's variational principle."""

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
        )

        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        # Define a set of initial parameters
        parameters = list(ansatz.parameters)
        param_dict = {param: np.pi / 4 for param in parameters}
        var_principle = RealMcLachlanPrinciple()

        bound_metric_tensor = var_principle.metric_tensor(
            ansatz, parameters, parameters, list(param_dict.values()), None, None
        )

        np.testing.assert_almost_equal(
            bound_metric_tensor, expected_bound_metric_tensor_2, decimal=5
        )

    def test_calc_evolution_grad(self):
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
        )

        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        # Define a set of initial parameters
        parameters = list(ansatz.parameters)
        param_dict = {param: np.pi / 4 for param in parameters}
        var_principle = RealMcLachlanPrinciple()

        bound_evolution_grad = var_principle.evolution_grad(
            observable,
            ansatz,
            None,
            param_dict,
            parameters,
            parameters,
            list(param_dict.values()),
            None,
            None,
        )

        expected_bound_evolution_grad = [
            (-0.04514911474522546 + 4e-18j),
            (0.0963123928027075 - 1.5e-18j),
            (0.1365347823673539 - 7e-18j),
            (0.004969316401057883 - 4.9999999999999996e-18j),
            (-0.003843833929692342 - 4.999999999999998e-19j),
            (0.07036988622493834 - 7e-18j),
            (0.16560609099860682 - 3.5e-18j),
            (0.16674183768051887 + 1e-18j),
            (-0.03843296670360974 - 6e-18j),
            (0.08891074158680243 - 6e-18j),
            (0.06425681697616654 + 7e-18j),
            (-0.03172376682078948 - 7e-18j),
        ]

        np.testing.assert_almost_equal(
            bound_evolution_grad, expected_bound_evolution_grad, decimal=5
        )


if __name__ == "__main__":
    unittest.main()
