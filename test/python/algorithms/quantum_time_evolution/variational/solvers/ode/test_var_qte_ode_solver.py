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

from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode.var_qte_ode_solver import (
    VarQteOdeSolver,
)
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


class TestVarQteOdeSolver(QiskitAlgorithmsTestCase):

    # TODO runs very slowly
    def test_run_no_backend(self):

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

        operator = ~StateFn(observable) @ StateFn(ansatz)
        init_param_values = np.zeros(len(ansatz.ordered_parameters))
        for i in range(ansatz.num_qubits):
            init_param_values[-(ansatz.num_qubits + i + 1)] = np.pi / 2

        param_dict = dict(zip(parameters, init_param_values))

        backend = Aer.get_backend("statevector_simulator")
        state = operator[-1]
        init_state = CircuitSampler(backend).convert(state, params=param_dict)
        init_state = init_state.eval().primitive.data

        h = operator.oplist[0].primitive * operator.oplist[0].coeff
        h_matrix = h.to_matrix(massive=True)
        h_norm = np.linalg.norm(h_matrix, np.infty)
        h_squared = h ** 2
        h_squared = ComposedOp([~StateFn(h_squared.reduce()), state])
        h_squared = PauliExpectation().convert(h_squared)

        error_calculator = ImaginaryErrorCalculator(
            h_squared,
            operator,
            CircuitSampler(backend),
            CircuitSampler(backend),
            param_dict,
        )

        var_principle = ImaginaryMcLachlanVariationalPrinciple()
        regularization = "ridge"
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, param_dict, regularization)
        time = 1

        target_state = np.dot(expm(-1 * h_matrix * time), init_state)
        # Normalization
        target_state /= np.sqrt(_inner_prod(target_state, target_state))

        reg = "ridge"
        error_based_ode = True

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
            reg,
            CircuitSampler(backend),
            None,
            error_based_ode,
        )

        var_qte_ode_solver = VarQteOdeSolver(
            list(param_dict.values()),
            ode_function_generator,
        )

        result = var_qte_ode_solver._run(time)

        print(result)


if __name__ == "__main__":
    unittest.main()
