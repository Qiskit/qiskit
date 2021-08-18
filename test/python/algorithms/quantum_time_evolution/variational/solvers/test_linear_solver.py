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

from qiskit.algorithms.quantum_time_evolution.variational.principles.imaginary.implementations.imaginary_mc_lachlan_variational_principle import (
    ImaginaryMcLachlanVariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.linear_solver import LinearSolver
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z
from test.python.algorithms import QiskitAlgorithmsTestCase


class TestLinearSolver(QiskitAlgorithmsTestCase):
    def test_solve_sle_no_backend_not_faster(self):
        linear_solver = LinearSolver()

        # Define the Hamiltonian for the simulation
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
        # Define Ansatz
        # ansatz = RealAmplitudes(observable.num_qubits, reps=d)
        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        # Define a set of initial parameters
        parameters = ansatz.ordered_parameters

        var_principle = ImaginaryMcLachlanVariationalPrinciple(observable, ansatz, parameters)

        param_dict = {}
        for param in parameters:
            param_dict[param] = 2.0
        print(param_dict)
        sol = linear_solver._solve_sle(var_principle, param_dict)
        print(sol)


if __name__ == "__main__":
    unittest.main()
