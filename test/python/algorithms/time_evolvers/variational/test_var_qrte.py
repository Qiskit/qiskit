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
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import data, ddt
import numpy as np

from qiskit.algorithms.gradients import LinCombQFI, DerivativeType, LinCombEstimatorGradient
from qiskit.primitives import Estimator
from qiskit.utils import algorithm_globals
from qiskit.algorithms.time_evolvers.variational.var_qrte import VarQRTE
from qiskit.quantum_info import SparsePauliOp, Pauli
from qiskit.algorithms import TimeEvolutionProblem
from qiskit.algorithms.time_evolvers.variational import (
    RealMcLachlanPrinciple,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import PauliSumOp


@ddt
class TestVarQRTE(QiskitAlgorithmsTestCase):
    """Test Variational Quantum Real Time Evolution algorithm."""

    def setUp(self):
        super().setUp()
        self.seed = 11
        np.random.seed(self.seed)

    def test_run_d_1_with_aux_ops(self):
        """Test VarQRTE for d = 1 and t = 0.1 with evaluating auxiliary operators and the Forward
        Euler solver."""
        observable = SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        )
        aux_ops = [Pauli("XX"), Pauli("YZ")]
        d = 1
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(len(parameters)):
            init_param_values[i] = np.pi / 2
        init_param_values[0] = 1

        time = 0.1

        evolution_problem = TimeEvolutionProblem(observable, time, aux_operators=aux_ops)

        thetas_expected = [
            0.886841151529636,
            1.53852629218265,
            1.57099556659882,
            1.5889216657174,
            1.5996487153364,
            1.57018939515742,
            1.63950719260698,
            1.53853696496673,
        ]

        thetas_expected_shots = [
            0.886975892820015,
            1.53822607733397,
            1.57058096749141,
            1.59023223608564,
            1.60105707043745,
            1.57018042397236,
            1.64010900210835,
            1.53959523034133,
        ]

        with self.subTest(msg="Test exact backend."):
            algorithm_globals.random_seed = self.seed
            estimator = Estimator()
            qfi = LinCombQFI(estimator)
            gradient = LinCombEstimatorGradient(estimator, derivative_type=DerivativeType.IMAG)
            var_principle = RealMcLachlanPrinciple(qfi, gradient)

            var_qite = VarQRTE(
                ansatz,
                var_principle,
                init_param_values,
                estimator,
                num_timesteps=25,
            )
            evolution_result = var_qite.evolve(evolution_problem)

            evolved_state = evolution_result.evolved_state
            aux_ops = evolution_result.aux_ops_evaluated

            parameter_values = evolved_state.data[0][0].params

            expected_aux_ops = [0.06836996703935797, 0.7711574493422457]

            for i, parameter_value in enumerate(parameter_values):
                np.testing.assert_almost_equal(
                    float(parameter_value), thetas_expected[i], decimal=2
                )

            np.testing.assert_array_almost_equal(
                [result[0] for result in aux_ops], expected_aux_ops
            )

        with self.subTest(msg="Test shot-based backend."):
            algorithm_globals.random_seed = self.seed

            estimator = Estimator(options={"shots": 4 * 4096, "seed": self.seed})
            qfi = LinCombQFI(estimator)
            gradient = LinCombEstimatorGradient(estimator, derivative_type=DerivativeType.IMAG)
            var_principle = RealMcLachlanPrinciple(qfi, gradient)

            var_qite = VarQRTE(
                ansatz,
                var_principle,
                init_param_values,
                estimator,
                num_timesteps=25,
            )
            evolution_result = var_qite.evolve(evolution_problem)

            evolved_state = evolution_result.evolved_state
            aux_ops = evolution_result.aux_ops_evaluated

            parameter_values = evolved_state.data[0][0].params

            expected_aux_ops = [
                0.06920924180526315,
                0.7779237744682032,
            ]

            for i, parameter_value in enumerate(parameter_values):
                np.testing.assert_almost_equal(
                    float(parameter_value), thetas_expected_shots[i], decimal=2
                )

            np.testing.assert_array_almost_equal(
                [result[0] for result in aux_ops], expected_aux_ops
            )

    @data(
        SparsePauliOp.from_list(
            [
                ("II", 0.2252),
                ("ZZ", 0.5716),
                ("IZ", 0.3435),
                ("ZI", -0.4347),
                ("YY", 0.091),
                ("XX", 0.091),
            ]
        ),
        PauliSumOp(
            SparsePauliOp.from_list(
                [
                    ("II", 0.2252),
                    ("ZZ", 0.5716),
                    ("IZ", 0.3435),
                    ("ZI", -0.4347),
                    ("YY", 0.091),
                    ("XX", 0.091),
                ]
            )
        ),
    )
    def test_run_d_2(self, observable):
        """Test VarQRTE for d = 2 and t = 1 with RK45 ODE solver."""
        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(len(parameters)):
            init_param_values[i] = np.pi / 4
        estimator = Estimator()
        qfi = LinCombQFI(estimator)
        gradient = LinCombEstimatorGradient(estimator, derivative_type=DerivativeType.IMAG)

        var_principle = RealMcLachlanPrinciple(qfi, gradient)

        param_dict = dict(zip(parameters, init_param_values))

        time = 1
        var_qrte = VarQRTE(
            ansatz,
            var_principle,
            param_dict,
            ode_solver="RK45",
            num_timesteps=25,
        )

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

        self._test_helper(observable, thetas_expected, time, var_qrte)

    def _test_helper(self, observable, thetas_expected, time, var_qrte):
        evolution_problem = TimeEvolutionProblem(observable, time)
        evolution_result = var_qrte.evolve(evolution_problem)
        evolved_state = evolution_result.evolved_state

        parameter_values = evolved_state.data[0][0].params

        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(float(parameter_value), thetas_expected[i], decimal=4)


if __name__ == "__main__":
    unittest.main()
