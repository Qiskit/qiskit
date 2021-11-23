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
from qiskit.quantum_info import state_fidelity, Statevector
from qiskit.algorithms.quantum_time_evolution.variational.principles.real.implementations\
    .real_mc_lachlan_variational_principle import (
    RealMcLachlanVariationalPrinciple,
)
from qiskit import Aer
from qiskit.algorithms.quantum_time_evolution.variational.var_qrte import VarQrte
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import (
    SummedOp,
    X,
    Y,
    I,
    Z, StateFn,
)
from test.python.algorithms import QiskitAlgorithmsTestCase

np.random.seed = 11
from qiskit.utils import algorithm_globals
algorithm_globals.random_seed = 11


class TestVarQrte(QiskitAlgorithmsTestCase):
    # pass
    def test_run_d_1(self):
    # def test_run_d_1():
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

        parameters = ansatz.ordered_parameters
        init_param_values = np.zeros(len(ansatz.ordered_parameters))
        for i in range(len(ansatz.ordered_parameters)):
            init_param_values[i] = np.pi / 2
        init_param_values[0] = 1
        var_principle = RealMcLachlanVariationalPrinciple()

        param_dict = dict(zip(parameters, init_param_values))

        reg = 'ridge'
        backend = Aer.get_backend("statevector_simulator")

        var_qrte = VarQrte(
            var_principle, regularization=reg, backend=backend, error_based_ode=False
        )
        time = 0.1

        evolution_result = var_qrte.evolve(
            observable,
            time,
            ansatz,  # ansatz is a state in this case
            hamiltonian_value_dict=param_dict,
        )

        # values from the prototype
        thetas_expected = [-0.02417581,  1.12012969,  1.51326909,  1.66610599,  1.68460371,
                           1.50506159, 2.28006157,  1.1236262]

        parameter_values = evolution_result.data[0][0].params

        print(state_fidelity(Statevector(evolution_result), Statevector(ansatz.assign_parameters(dict(
            zip(ansatz.parameters, thetas_expected))))))
        print('Expected ', thetas_expected)
        print('Computed ', parameter_values)
        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=2)

#
if __name__ == "__main__":
    unittest.main()
    # test_run_d_1()