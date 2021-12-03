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
from qiskit.algorithms.quantum_time_evolution.variational.principles.real.implementations.real_mc_lachlan_variational_principle import (
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
    Z,
)
from test.python.algorithms import QiskitAlgorithmsTestCase

np.random.seed = 11
from qiskit.utils import algorithm_globals

algorithm_globals.random_seed = 11


class TestVarQrte(QiskitAlgorithmsTestCase):
    def test_run_d_1(self):
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

        reg = None
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
        thetas_expected = [
            0.88689233,
            1.53841938,
            1.57100495,
            1.58891647,
            1.59947863,
            1.57016298,
            1.63950734,
            1.53843156,
        ]

        parameter_values = evolution_result.data[0][0].params

        print(
            state_fidelity(
                Statevector(evolution_result),
                Statevector(
                    ansatz.assign_parameters(dict(zip(ansatz.parameters, thetas_expected)))
                ),
            )
        )
        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=3)

    def test_run_d_2(self):
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

        parameters = ansatz.ordered_parameters
        init_param_values = np.zeros(len(ansatz.ordered_parameters))
        for i in range(len(ansatz.ordered_parameters)):
            init_param_values[i] = np.pi / 4

        var_principle = RealMcLachlanVariationalPrinciple()

        param_dict = dict(zip(parameters, init_param_values))

        reg = None
        backend = Aer.get_backend("statevector_simulator")

        var_qite = VarQrte(
            var_principle, regularization=reg, backend=backend, error_based_ode=False
        )
        time = 1

        evolution_result = var_qite.evolve(
            observable,
            time,
            ansatz,  # ansatz is a state in this case
            hamiltonian_value_dict=param_dict,
        )

        # values from the prototype
        thetas_expected = [
            0.281316385345389,
            0.986871118474767,
            1.35534959612472,
            0.691840609009987,
            0.57358725779109,
            1.03073602828349,
            1.47364740090864,
            1.24473065474471,
            1.26862435890771,
            1.05396303596684,
            0.844504836078978,
            0.558976984077953,
        ]
        print(
            state_fidelity(
                Statevector(evolution_result),
                Statevector(
                    ansatz.assign_parameters(dict(zip(ansatz.parameters, thetas_expected)))
                ),
            )
        )
        parameter_values = evolution_result.data[0][0].params
        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=4)

    def test_run_d_1_error_based(self):
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
            init_param_values[i] = np.pi / 4
        var_principle = RealMcLachlanVariationalPrinciple()

        param_dict = dict(zip(parameters, init_param_values))

        reg = "ridge"
        backend = Aer.get_backend("statevector_simulator")

        var_qrte = VarQrte(var_principle, regularization=reg, backend=backend, error_based_ode=True)
        time = 0.1

        evolution_result = var_qrte.evolve(
            observable,
            time,
            ansatz,  # ansatz is a state in this case
            hamiltonian_value_dict=param_dict,
        )

        # values from the prototype
        thetas_expected = [
            0.63886479,
            1.56391727,
            0.96591303,
            1.57332808,
            1.15739773,
            0.96590066,
            1.15161309,
            -0.3293391,
        ]

        parameter_values = evolution_result.data[0][0].params
        print(
            state_fidelity(
                Statevector(evolution_result),
                Statevector(
                    ansatz.assign_parameters(dict(zip(ansatz.parameters, thetas_expected)))
                ),
            )
        )
        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=3)

    def test_run_d_1_error_based_t_006(self):
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
            init_param_values[i] = np.pi / 4
        var_principle = RealMcLachlanVariationalPrinciple()

        param_dict = dict(zip(parameters, init_param_values))

        reg = "ridge"
        backend = Aer.get_backend("statevector_simulator")

        var_qrte = VarQrte(var_principle, regularization=reg, backend=backend, error_based_ode=True)
        time = 0.06

        evolution_result = var_qrte.evolve(
            observable,
            time,
            ansatz,  # ansatz is a state in this case
            hamiltonian_value_dict=param_dict,
        )

        # values from the prototype
        thetas_expected = [
            0.650027700129024,
            1.56194984853426,
            0.926353536023022,
            1.57053579386368,
            1.15169824084686,
            0.924784887323687,
            1.14070867127809,
            -0.316445310366042,
        ]

        parameter_values = evolution_result.data[0][0].params
        print(
            state_fidelity(
                Statevector(evolution_result),
                Statevector(
                    ansatz.assign_parameters(dict(zip(ansatz.parameters, thetas_expected)))
                ),
            )
        )
        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=3)


if __name__ == "__main__":
    unittest.main()
