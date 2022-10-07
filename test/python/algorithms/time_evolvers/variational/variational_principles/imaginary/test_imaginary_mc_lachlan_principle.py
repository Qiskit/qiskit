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

"""Test imaginary McLachlan's variational principle."""

import unittest
from test.python.algorithms.time_evolvers.variational.variational_principles.\
    expected_results.test_imaginary_mc_lachlan_variational_principle_expected1 import (
    expected_bound_metric_tensor_1,
)
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.time_evolvers.variational import (
    ImaginaryMcLachlanPrinciple,
)
from qiskit.circuit.library import EfficientSU2


class TestImaginaryMcLachlanPrinciple(QiskitAlgorithmsTestCase):
    """Test imaginary McLachlan's variational principle."""

    def test_calc_metric_tensor(self):
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
        var_principle = ImaginaryMcLachlanPrinciple()

        bound_metric_tensor = var_principle.metric_tensor(ansatz, list(param_dict.values()))

        np.testing.assert_almost_equal(bound_metric_tensor, expected_bound_metric_tensor_1)

    def test_calc_calc_evolution_grad(self):
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
        var_principle = ImaginaryMcLachlanPrinciple()

        bound_evolution_grad = var_principle.evolution_grad(
            observable,
            ansatz,
            param_dict,
            parameters,
            parameters,
            list(param_dict.values()),
        )

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

        np.testing.assert_almost_equal(bound_evolution_grad, expected_bound_evolution_grad)


if __name__ == "__main__":
    unittest.main()
