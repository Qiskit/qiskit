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
from scipy.linalg import expm

from qiskit.algorithms.quantum_time_evolution.variational.calculators.distance_energy_calculator import (
    _inner_prod,
)
from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.gradient_errors.imaginary_error_calculator import (
    ImaginaryErrorCalculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode.ode_function_generator import (
    OdeFunctionGenerator,
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


class TestOdeFunctionGenerator(QiskitAlgorithmsTestCase):
    def test_var_qte_ode_function(self):
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
        if backend is not None:
            init_state = CircuitSampler(backend).convert(state, params=param_dict)
        else:
            init_state = state.assign_parameters(param_dict)
        init_state = init_state.eval().primitive.data

        h = operator.oplist[0].primitive * operator.oplist[0].coeff
        h_matrix = h.to_matrix(massive=True)
        h_norm = np.linalg.norm(h_matrix, np.infty)
        h_squared = h ** 2
        h_squared = ComposedOp([~StateFn(h_squared.reduce()), state])
        h_squared = PauliExpectation().convert(h_squared)

        error_calculator = ImaginaryErrorCalculator(
            h_squared, operator, CircuitSampler(backend), CircuitSampler(backend), param_dict
        )

        var_principle = ImaginaryMcLachlanVariationalPrinciple()
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, param_dict)
        time = 2

        target_state = np.dot(expm(-1 * h_matrix * time), init_state)
        # Normalization
        target_state /= np.sqrt(_inner_prod(target_state, target_state))

        ode_function_generator = OdeFunctionGenerator(
            error_calculator,
            param_dict,
            var_principle,
            state,
            target_state,
            h_matrix,
            h_norm,
            CircuitSampler(backend),
            CircuitSampler(backend),
            CircuitSampler(backend),
        )

        qte_ode_function = ode_function_generator.var_qte_ode_function(time, np.array([0, 0, 0]))
        expected_qte_ode_function = [
            -0.8842908,
            0.0441611,
            -0.21244606,
            0.2349356,
            -0.50246622,
            -0.6425113,
            0.12545623,
            0.07241851,
            1.01843757,
            0.3669189,
            0.10147791,
            0.18632604,
            0.75854565,
        ]

        np.testing.assert_array_almost_equal(expected_qte_ode_function, qte_ode_function)


if __name__ == "__main__":
    unittest.main()
