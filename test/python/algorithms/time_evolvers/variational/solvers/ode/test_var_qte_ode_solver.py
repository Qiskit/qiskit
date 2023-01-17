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

"""Test solver of ODEs."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import ddt, data, unpack
import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit.algorithms.time_evolvers.variational.solvers.ode.forward_euler_solver import (
    ForwardEulerSolver,
)
from qiskit.algorithms.time_evolvers.variational.solvers.var_qte_linear_solver import (
    VarQTELinearSolver,
)
from qiskit.algorithms.time_evolvers.variational.solvers.ode.var_qte_ode_solver import (
    VarQTEOdeSolver,
)
from qiskit.algorithms.time_evolvers.variational.solvers.ode.ode_function import (
    OdeFunction,
)
from qiskit.algorithms.time_evolvers.variational import (
    ImaginaryMcLachlanPrinciple,
)
from qiskit.circuit.library import EfficientSU2


@ddt
class TestVarQTEOdeSolver(QiskitAlgorithmsTestCase):
    """Test solver of ODEs."""

    @data(
        (
            "RK45",
            [
                -0.30076755873631345,
                -0.8032811383782005,
                1.1674108371914734e-15,
                3.2293849116821145e-16,
                2.541585055586039,
                1.155475184255733,
                -2.966331417968169e-16,
                9.604292449638343e-17,
            ],
        ),
        (
            ForwardEulerSolver,
            [
                -3.2707e-01,
                -8.0960e-01,
                3.4323e-16,
                8.9034e-17,
                2.5290e00,
                1.1563e00,
                3.0227e-16,
                -2.2769e-16,
            ],
        ),
    )
    @unpack
    def test_run_no_backend(self, ode_solver, expected_result):
        """Test ODE solver with no backend."""
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

        d = 1
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        # Define a set of initial parameters
        parameters = list(ansatz.parameters)

        init_param_values = np.zeros(len(parameters))
        for i in range(ansatz.num_qubits):
            init_param_values[-(ansatz.num_qubits + i + 1)] = np.pi / 2

        param_dict = dict(zip(parameters, init_param_values))

        var_principle = ImaginaryMcLachlanPrinciple()

        time = 1

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
        ode_function_generator = OdeFunction(linear_solver, param_dict, t_param)

        var_qte_ode_solver = VarQTEOdeSolver(
            list(param_dict.values()),
            ode_function_generator,
            ode_solver=ode_solver,
            num_timesteps=25,
        )

        result, _, _ = var_qte_ode_solver.run(time)

        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)


if __name__ == "__main__":
    unittest.main()
