# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Variational Quantum Real Time Evolution algorithm."""

import unittest

from qiskit.algorithms import EvolutionProblem
from test.python.algorithms import QiskitAlgorithmsTestCase
import numpy as np

from qiskit.algorithms.evolvers.variational.solvers.ode.ode_function_generator import (
    OdeFunctionGenerator,
)
from qiskit.quantum_info import state_fidelity, Statevector
from qiskit.algorithms.evolvers.variational.variational_principles.real.implementations.real_mc_lachlan_variational_principle import (
    RealMcLachlanVariationalPrinciple,
)
from qiskit import Aer
from qiskit.algorithms.evolvers.variational.algorithms.var_qrte import VarQRTE
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import (
    SummedOp,
    X,
    Y,
    I,
    Z,
)


class TestVarQRTE(QiskitAlgorithmsTestCase):
    """Test Variational Quantum Real Time Evolution algorithm."""

    def setUp(self):
        super().setUp()
        np.random.seed(11)

    def test_run_d_1(self):
        """Test VarQRTE for d = 1 and t = 0.1."""
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

        backend = Aer.get_backend("statevector_simulator")

        ode_function = OdeFunctionGenerator()
        var_qrte = VarQRTE(var_principle, ode_function, backend=backend)
        time = 0.1

        evolution_problem = EvolutionProblem(
            observable, time, ansatz, hamiltonian_value_dict=param_dict
        )

        evolution_result = var_qrte.evolve(evolution_problem)

        evolved_state = evolution_result.evolved_state

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

        parameter_values = evolved_state.data[0][0].params

        print(
            state_fidelity(
                Statevector(evolved_state),
                Statevector(
                    ansatz.assign_parameters(dict(zip(ansatz.parameters, thetas_expected)))
                ),
            )
        )
        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=3)

    def test_run_d_2(self):
        """Test VarQRTE for d = 2 and t = 1."""
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

        backend = Aer.get_backend("statevector_simulator")

        ode_function = OdeFunctionGenerator()
        var_qrte = VarQRTE(var_principle, ode_function, backend=backend)
        time = 1

        evolution_problem = EvolutionProblem(
            observable, time, ansatz, hamiltonian_value_dict=param_dict
        )

        evolution_result = var_qrte.evolve(evolution_problem)

        evolved_state = evolution_result.evolved_state

        thetas_expected = [
            0.348407744196573,
            0.919404626262464,
            1.18189219371626,
            0.771011177789998,
            0.734384256533924,
            0.965289520781899,
            1.14441687204195,
            1.17231927568571,
            1.03014771379412,
            0.867266309056347,
            0.699606368428206,
            0.610788576398685,
        ]

        print(
            state_fidelity(
                Statevector(evolved_state),
                Statevector(
                    ansatz.assign_parameters(dict(zip(ansatz.parameters, thetas_expected)))
                ),
            )
        )
        parameter_values = evolved_state.data[0][0].params

        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=4)


if __name__ == "__main__":
    unittest.main()
