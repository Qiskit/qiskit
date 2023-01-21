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

"""Test solver of linear equations."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase

# fmt: off
from test.python.algorithms.time_evolvers.variational.solvers.expected_results.\
    test_varqte_linear_solver_expected_1 import expected_metric_res_1
# fmt: on

import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.time_evolvers.variational import (
    ImaginaryMcLachlanPrinciple,
)
from qiskit.algorithms.time_evolvers.variational.solvers.var_qte_linear_solver import (
    VarQTELinearSolver,
)
from qiskit.circuit.library import EfficientSU2


class TestVarQTELinearSolver(QiskitAlgorithmsTestCase):
    """Test solver of linear equations."""

    def test_solve_lse(self):
        """Test SLE solver."""

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

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(ansatz.num_qubits):
            init_param_values[-(ansatz.num_qubits + i + 1)] = np.pi / 2

        param_dict = dict(zip(parameters, init_param_values))

        var_principle = ImaginaryMcLachlanPrinciple()
        t_param = None
        linear_solver = None
        linear_solver = VarQTELinearSolver(
            var_principle,
            observable,
            ansatz,
            parameters,
            t_param,
            linear_solver,
        )

        nat_grad_res, metric_res, grad_res = linear_solver.solve_lse(param_dict)

        expected_nat_grad_res = [
            3.43500000e-01,
            -2.89800000e-01,
            2.43575264e-16,
            1.31792695e-16,
            -9.61200000e-01,
            -2.89800000e-01,
            1.27493709e-17,
            1.12587456e-16,
            3.43500000e-01,
            -2.89800000e-01,
            3.69914720e-17,
            1.95052083e-17,
        ]

        expected_grad_res = [
            (0.17174999999999926 - 0j),
            (-0.21735000000000085 + 0j),
            (4.114902862895087e-17 - 0j),
            (4.114902862895087e-17 - 0j),
            (-0.24030000000000012 + 0j),
            (-0.21735000000000085 + 0j),
            (4.114902862895087e-17 - 0j),
            (4.114902862895087e-17 - 0j),
            (0.17174999999999918 - 0j),
            (-0.21735000000000076 + 0j),
            (1.7789936190837538e-17 - 0j),
            (-8.319872568662832e-17 + 0j),
        ]

        np.testing.assert_array_almost_equal(nat_grad_res, expected_nat_grad_res, decimal=4)
        np.testing.assert_array_almost_equal(grad_res, expected_grad_res, decimal=4)
        np.testing.assert_array_almost_equal(metric_res, expected_metric_res_1, decimal=4)


if __name__ == "__main__":
    unittest.main()
