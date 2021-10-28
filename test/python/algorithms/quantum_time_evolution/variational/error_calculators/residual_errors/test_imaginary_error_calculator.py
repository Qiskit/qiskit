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

import numpy as np
from ddt import ddt

from qiskit import Aer
from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.gradient_errors.imaginary_error_calculator import (
    ImaginaryErrorCalculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.principles.imaginary.implementations.imaginary_mc_lachlan_variational_principle import (
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
    CircuitSampler,
    PauliExpectation,
    ComposedOp,
)
from test.python.algorithms import QiskitAlgorithmsTestCase


@ddt
class TestImaginaryErrorCalculator(QiskitAlgorithmsTestCase):
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

        backend = Aer.get_backend("statevector_simulator")

        h_squared_sampler = CircuitSampler(backend)
        exp_operator_sampler = CircuitSampler(backend)

        h = operator.oplist[0].primitive * operator.oplist[0].coeff
        h_squared = h ** 2
        h_squared = ComposedOp([~StateFn(h_squared.reduce()), state])
        h_squared = PauliExpectation().convert(h_squared)

        imaginary_error_calculator = ImaginaryErrorCalculator(
            h_squared,
            operator,
            h_squared_sampler,
            exp_operator_sampler,
            param_dict,
            backend=None,
        )
        backend = Aer.get_backend("qasm_simulator")
        linear_solver = VarQteLinearSolver(
            CircuitSampler(backend), CircuitSampler(backend), CircuitSampler(backend), backend=None
        )
        var_principle = ImaginaryMcLachlanVariationalPrinciple()
        regularization = "ridge"
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, param_dict, regularization)
        ng_res = linear_solver._solve_sle(var_principle, param_dict)
        grad_res, metric_res = linear_solver._solve_sle_for_error_bounds(var_principle, param_dict)

        eps_squared, dtdt_state, regrad2 = imaginary_error_calculator._calc_single_step_error(
            ng_res, grad_res, metric_res
        )
        # TODO verify if values correct
        eps_squared_expected = 1.033501133414088
        dtdt_state_expected = 0.2563172717800766
        regrad2_expected = 0.25836737139322774
        np.testing.assert_almost_equal(eps_squared, eps_squared_expected)
        np.testing.assert_almost_equal(dtdt_state, dtdt_state_expected)
        np.testing.assert_almost_equal(regrad2, regrad2_expected)

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

        backend = Aer.get_backend("statevector_simulator")

        h_squared_sampler = CircuitSampler(backend)
        exp_operator_sampler = CircuitSampler(backend)

        h = operator.oplist[0].primitive * operator.oplist[0].coeff
        h_squared = h ** 2
        h_squared = ComposedOp([~StateFn(h_squared.reduce()), state])
        h_squared = PauliExpectation().convert(h_squared)

        imaginary_error_calculator = ImaginaryErrorCalculator(
            h_squared,
            operator,
            h_squared_sampler,
            exp_operator_sampler,
            param_dict,
            backend=None,
        )
        backend = Aer.get_backend("qasm_simulator")
        linear_solver = VarQteLinearSolver(
            CircuitSampler(backend), CircuitSampler(backend), CircuitSampler(backend), backend=None
        )
        var_principle = ImaginaryMcLachlanVariationalPrinciple()
        regularization = "ridge"
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, param_dict, regularization)
        ng_res = linear_solver._solve_sle(var_principle, param_dict)
        grad_res, metric_res = linear_solver._solve_sle_for_error_bounds(var_principle, param_dict)

        eps_squared = imaginary_error_calculator._calc_single_step_error_gradient(
            ng_res, grad_res, metric_res
        )
        # TODO verify if values correct
        eps_squared_expected = [
            0.627225,
            0.021158,
            0.096593,
            -0.202396,
            0.240526,
            0.37758,
            -0.002541,
            -0.149731,
            -0.554199,
            -0.007761,
            0.069505,
            -0.066318,
        ]

        np.testing.assert_array_almost_equal(eps_squared, eps_squared_expected)


if __name__ == "__main__":
    unittest.main()
