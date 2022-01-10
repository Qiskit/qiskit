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

"""Test solver of ODEs."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np

from qiskit.algorithms.time_evolution.variational.solvers.ode.var_qte_ode_solver import (
    VarQteOdeSolver,
)
from qiskit.algorithms.time_evolution.variational.solvers.ode.ode_function_generator import (
    OdeFunctionGenerator,
)
from qiskit import Aer
from qiskit.algorithms.time_evolution.variational.variational_principles.imaginary.implementations.imaginary_mc_lachlan_variational_principle import (
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

        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, parameters)
        time = 1

        reg = "ridge"

        ode_function_generator = OdeFunctionGenerator(
            param_dict,
            var_principle,
            CircuitSampler(backend),
            CircuitSampler(backend),
            CircuitSampler(backend),
            reg,
            None,
        )

        var_qte_ode_solver = VarQteOdeSolver(
            list(param_dict.values()),
            ode_function_generator,
        )

        result = var_qte_ode_solver._run(time)

        expected_result = [
            -0.334644,
            -0.790279,
            -0.021695,
            -0.002949,
            2.503808,
            1.147496,
            -0.008201,
            -0.003358,
        ]

        # TODO check if values correct
        np.testing.assert_array_almost_equal(result, expected_result, decimal=4)


if __name__ == "__main__":
    unittest.main()
