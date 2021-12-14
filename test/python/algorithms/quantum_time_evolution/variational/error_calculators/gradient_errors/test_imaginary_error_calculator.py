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

"""Test imaginary gradient errors calculator."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np
from ddt import ddt

from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.gradient_errors.imaginary_error_calculator import (
    ImaginaryErrorCalculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.variational_principles.imaginary.implementations.imaginary_mc_lachlan_variational_principle import (
    ImaginaryMcLachlanVariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.var_qte_linear_solver import (
    VarQteLinearSolver,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import (
    SummedOp,
    X,
    Y,
    I,
    Z,
    StateFn,
    PauliExpectation,
    ComposedOp,
)


@ddt
class TestImaginaryErrorCalculator(QiskitAlgorithmsTestCase):
    """Test imaginary gradient errors calculator."""

    def test_calc_single_step_error(self):
        """Test calculating single step error."""
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

        operator = ~StateFn(observable) @ StateFn(ansatz)
        param_dict = {param: np.pi / 4 for param in parameters}
        state = operator[-1]

        backend = None
        circuit_sampler = None

        hamiltonian = operator.oplist[0].primitive * operator.oplist[0].coeff
        h_squared = hamiltonian ** 2
        h_squared = ComposedOp([~StateFn(h_squared.reduce()), state])
        h_squared = PauliExpectation().convert(h_squared)

        imaginary_error_calculator = ImaginaryErrorCalculator(
            h_squared,
            operator,
            circuit_sampler,
            circuit_sampler,
            backend=backend,
        )

        linear_solver = VarQteLinearSolver(
            circuit_sampler, circuit_sampler, circuit_sampler, backend=backend
        )
        var_principle = ImaginaryMcLachlanVariationalPrinciple()
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, parameters)

        ng_res, metric_res, grad_res = linear_solver._solve_sle(var_principle, param_dict)

        eps_squared, dtdt_state, regrad2 = imaginary_error_calculator._calc_single_step_error(
            ng_res, grad_res, metric_res, param_dict
        )
        # TODO verify if values correct
        eps_squared_expected = 0.7813473565426678
        dtdt_state_expected = 0.2604491188475557 + 0j
        regrad2_expected = 0.13022455942377797 + 0j
        np.testing.assert_almost_equal(eps_squared, eps_squared_expected, decimal=5)
        np.testing.assert_almost_equal(dtdt_state, dtdt_state_expected, decimal=5)
        np.testing.assert_almost_equal(regrad2, regrad2_expected, decimal=5)

    def test_calc_single_step_error_gradient(self):
        """Test calculating single step error gradient."""
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

        operator = ~StateFn(observable) @ StateFn(ansatz)
        param_dict = {param: np.pi / 4 for param in parameters}
        state = operator[-1]

        backend = None
        circuit_sampler = None

        hamiltonian = operator.oplist[0].primitive * operator.oplist[0].coeff
        h_squared = hamiltonian ** 2
        h_squared = ComposedOp([~StateFn(h_squared.reduce()), state])
        h_squared = PauliExpectation().convert(h_squared)

        imaginary_error_calculator = ImaginaryErrorCalculator(
            h_squared,
            operator,
            circuit_sampler,
            circuit_sampler,
            backend=backend,
        )

        linear_solver = VarQteLinearSolver(
            circuit_sampler, circuit_sampler, circuit_sampler, backend=backend
        )
        var_principle = ImaginaryMcLachlanVariationalPrinciple()
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, parameters)

        ng_res, metric_res, grad_res = linear_solver._solve_sle(var_principle, param_dict)

        eps_squared = imaginary_error_calculator._calc_single_step_error_gradient(
            ng_res, grad_res, metric_res
        )
        # TODO verify if values correct
        eps_squared_expected = [
            0.435052,
            0.014177,
            0.065261,
            -0.134481,
            0.166601,
            0.260416,
            -0.001525,
            -0.099443,
            -0.352939,
            -0.003129,
            0.046254,
            -0.043715,
        ]

        np.testing.assert_array_almost_equal(eps_squared, eps_squared_expected, decimal=5)


if __name__ == "__main__":
    unittest.main()
