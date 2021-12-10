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
import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np
from ddt import ddt

from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.gradient_errors.real_error_calculator import (
    RealErrorCalculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.real.implementations.real_mc_lachlan_variational_principle import (
    RealMcLachlanVariationalPrinciple,
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
    # TODO test fail due to non-negligible imaginary parts, check real imag principles if they work.
    def test_calc_single_step_error(self):
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

        imaginary_error_calculator = RealErrorCalculator(
            h_squared,
            operator,
            circuit_sampler,
            circuit_sampler,
            backend=backend,
        )
        linear_solver = VarQteLinearSolver(
            circuit_sampler, circuit_sampler, circuit_sampler, backend=backend
        )
        var_principle = RealMcLachlanVariationalPrinciple()
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, parameters)
        ng_res, metric_res, grad_res = linear_solver._solve_sle(var_principle, param_dict)

        eps_squared, dtdt_state, regrad2 = imaginary_error_calculator._calc_single_step_error(
            ng_res, grad_res, metric_res, param_dict
        )

        # TODO verify if values correct
        eps_squared_expected = 0.2604491188475558
        dtdt_state_expected = 0.2604491188475554 + 0j
        regrad2_expected = 0.13022455942377772 + 2.7259468087997858e-18j
        np.testing.assert_almost_equal(eps_squared, eps_squared_expected, decimal=5)
        np.testing.assert_almost_equal(dtdt_state, dtdt_state_expected, decimal=5)
        np.testing.assert_almost_equal(regrad2, regrad2_expected, decimal=5)

    def test_calc_single_step_error_gradient(self):
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

        imaginary_error_calculator = RealErrorCalculator(
            h_squared,
            operator,
            circuit_sampler,
            circuit_sampler,
            backend=backend,
        )
        linear_solver = VarQteLinearSolver(
            circuit_sampler, circuit_sampler, circuit_sampler, backend=backend
        )
        var_principle = RealMcLachlanVariationalPrinciple()

        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, parameters)
        ng_res, metric_res, grad_res = linear_solver._solve_sle(var_principle, param_dict)
        eps_squared = imaginary_error_calculator._calc_single_step_error_gradient(
            ng_res, grad_res, metric_res
        )
        # TODO verify if values correct
        eps_squared_expected = [
            4.859767e-02,
            1.793361e-02,
            2.194252e-02,
            4.033828e-03,
            4.480250e-02,
            1.881751e-02,
            1.640536e-02,
            6.966432e-02,
            7.532176e-04,
            4.804854e-05,
            9.225491e-04,
            1.333258e-02,
        ]

        np.testing.assert_array_almost_equal(eps_squared, eps_squared_expected, decimal=5)


if __name__ == "__main__":
    unittest.main()
