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

"""Test solver of ODEs."""

import unittest

from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np

from qiskit.algorithms.time_evolution.variational.solvers.var_qte_linear_solver import (
    VarQteLinearSolver,
)
from qiskit.algorithms.time_evolution.variational.solvers.ode.var_qte_ode_solver import (
    VarQteOdeSolver,
)
from qiskit.algorithms.time_evolution.variational.solvers.ode.ode_function_generator import (
    OdeFunctionGenerator,
)
from qiskit import Aer
from qiskit.algorithms.time_evolution.variational.variational_principles.imaginary\
    .implementations.imaginary_mc_lachlan_variational_principle import (
    ImaginaryMcLachlanVariationalPrinciple,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import (
    SummedOp,
    X,
    Y,
    I,
    Z,
    CircuitSampler,
)


class TestVarQteOdeSolver(QiskitAlgorithmsTestCase):
    """Test solver of ODEs."""

    # TODO runs slowly
    def test_run_no_backend(self):
        """Test ODE solver with no backend."""
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

        d = 1
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        # Define a set of initial parameters
        parameters = ansatz.ordered_parameters

        init_param_values = np.zeros(len(ansatz.ordered_parameters))
        for i in range(ansatz.num_qubits):
            init_param_values[-(ansatz.num_qubits + i + 1)] = np.pi / 2

        param_dict = dict(zip(parameters, init_param_values))

        backend = Aer.get_backend("statevector_simulator")

        var_principle = ImaginaryMcLachlanVariationalPrinciple()

        time = 1

        ode_function_generator = OdeFunctionGenerator()

        metric_tensor = var_principle._get_metric_tensor(ansatz, parameters)
        evolution_grad = var_principle._get_evolution_grad(observable, ansatz, parameters)

        linear_solver = VarQteLinearSolver(
            metric_tensor,
            evolution_grad,
            CircuitSampler(backend),
            CircuitSampler(backend),
            CircuitSampler(backend),
        )
        ode_function_generator._lazy_init(linear_solver, None, None, param_dict)

        var_qte_ode_solver = VarQteOdeSolver(
            list(param_dict.values()),
            ode_function_generator,
        )

        result = var_qte_ode_solver._run(time)

        expected_result = [-0.30076755873631345, -0.8032811383782005, 1.1674108371914734e-15,
                           3.2293849116821145e-16, 2.541585055586039, 1.155475184255733,
                           -2.966331417968169e-16, 9.604292449638343e-17]

        # TODO check if values correct
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)


if __name__ == "__main__":
    unittest.main()
