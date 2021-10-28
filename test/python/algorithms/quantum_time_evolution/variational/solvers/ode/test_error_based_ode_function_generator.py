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
from numpy import array

from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.gradient_errors.imaginary_error_calculator import (
    ImaginaryErrorCalculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode.error_based_ode_function_generator import (
    ErrorBasedOdeFunctionGenerator,
)
from qiskit import Aer
from qiskit.algorithms.quantum_time_evolution.variational.principles.imaginary.implementations.imaginary_mc_lachlan_variational_principle import (
    ImaginaryMcLachlanVariationalPrinciple,
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
    ComposedOp,
    PauliExpectation,
)
from test.python.algorithms import QiskitAlgorithmsTestCase


class TestErrorBasedOdeFunctionGenerator(QiskitAlgorithmsTestCase):
    def test_error_based_ode_fun(self):
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
        backend = Aer.get_backend("qasm_simulator")
        state = operator[-1]

        h = operator.oplist[0].primitive * operator.oplist[0].coeff
        h_squared = h ** 2
        h_squared = ComposedOp([~StateFn(h_squared.reduce()), state])
        h_squared = PauliExpectation().convert(h_squared)

        error_calculator = ImaginaryErrorCalculator(
            h_squared, operator, CircuitSampler(backend), CircuitSampler(backend), param_dict
        )

        var_principle = ImaginaryMcLachlanVariationalPrinciple()
        regularization = "ridge"
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, param_dict, regularization)

        ode_function_generator = ErrorBasedOdeFunctionGenerator(
            error_calculator,
            param_dict,
            var_principle,
            CircuitSampler(backend),
            CircuitSampler(backend),
            CircuitSampler(backend),
        )

        qte_ode_function = ode_function_generator.var_qte_ode_function(1, param_dict.values())

        # TODO extract
        # TODO verify values if correct
        expected_qte_ode_function = array(
            [
                -0.20586541,
                0.71865927,
                0.1353771,
                0.99934292,
                -0.00631492,
                -0.07428645,
                -0.34469716,
                -0.3218795,
                0.81221089,
                0.65605788,
                -0.32451751,
                -0.24383516,
            ]
        )
        np.testing.assert_almost_equal(expected_qte_ode_function, qte_ode_function)


if __name__ == "__main__":
    unittest.main()
