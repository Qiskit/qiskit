# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
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

# fmt: off
from test.python.algorithms.time_evolvers.variational.variational_principles.expected_results.\
    test_imaginary_mc_lachlan_variational_principle_expected2 import expected_bound_metric_tensor_2
# fmt: on
import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.time_evolvers.variational import (
    RealMcLachlanPrinciple,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.algorithms.gradients import LinCombEstimatorGradient, DerivativeType
from qiskit.primitives import Estimator


class TestRealMcLachlanPrinciple(QiskitAlgorithmsTestCase):
    """Test real McLachlan's variational principle."""

    def test_calc_calc_metric_tensor(self):
        """Test calculating a metric tensor."""
        observable = SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        )

        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        # Define a set of initial parameters
        parameters = list(ansatz.parameters)
        param_dict = {param: np.pi / 4 for param in parameters}
        var_principle = RealMcLachlanPrinciple()

        bound_metric_tensor = var_principle.metric_tensor(ansatz, list(param_dict.values()))

        np.testing.assert_almost_equal(
            bound_metric_tensor, expected_bound_metric_tensor_2, decimal=5
        )

    def test_calc_evolution_gradient(self):
        """Test calculating evolution gradient."""
        observable = SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        )

        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        # Define a set of initial parameters
        parameters = list(ansatz.parameters)
        param_dict = {param: np.pi / 4 for param in parameters}
        var_principle = RealMcLachlanPrinciple()

        bound_evolution_gradient = var_principle.evolution_gradient(
            observable, ansatz, list(param_dict.values()), parameters
        )

        expected_evolution_gradient = [
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
            bound_evolution_gradient, expected_evolution_gradient, decimal=5
        )

    def test_gradient_setting(self):
        """Test reactions to wrong gradient settings.."""
        estimator = Estimator()
        gradient = LinCombEstimatorGradient(estimator, derivative_type=DerivativeType.REAL)

        with self.assertWarns(Warning):
            var_principle = RealMcLachlanPrinciple(gradient=gradient)

        np.testing.assert_equal(var_principle.gradient._derivative_type, DerivativeType.IMAG)


if __name__ == "__main__":
    unittest.main()
