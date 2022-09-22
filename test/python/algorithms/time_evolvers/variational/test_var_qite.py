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

"""Test Variational Quantum Imaginary Time Evolution algorithm."""

import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import data, ddt
import numpy as np

from qiskit.algorithms.time_evolvers.variational.var_qite import VarQITE
from qiskit.quantum_info import SparsePauliOp, state_fidelity, Statevector, Pauli
from qiskit.test import slow_test
from qiskit.utils import algorithm_globals
from qiskit.algorithms import TimeEvolutionProblem
from qiskit.algorithms.time_evolvers.variational import (
    ImaginaryMcLachlanPrinciple,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import PauliSumOp


@ddt
class TestVarQITE(QiskitAlgorithmsTestCase):
    """Test Variational Quantum Imaginary Time Evolution algorithm."""

    def setUp(self):
        super().setUp()
        self.seed = 11
        np.random.seed(self.seed)

    # TODO test shot-based primitive setting
    # @slow_test
    # def test_run_d_1_with_aux_ops(self):
    #     """Test VarQITE for d = 1 and t = 1 with evaluating auxiliary operator and the Forward
    #     Euler solver.."""
    #
    #     observable = SparsePauliOp.from_list(
    #         [
    #             ("II", 0.2252),
    #             ("ZZ", 0.5716),
    #             ("IZ", 0.3435),
    #             ("ZI", -0.4347),
    #             ("YY", 0.091),
    #             ("XX", 0.091),
    #         ]
    #     )
    #
    #     aux_ops = [Pauli("XX"), Pauli("YZ")]
    #     d = 1
    #     ansatz = EfficientSU2(observable.num_qubits, reps=d)
    #
    #     parameters = list(ansatz.parameters)
    #     init_param_values = np.zeros(len(parameters))
    #     for i in range(len(parameters)):
    #         init_param_values[i] = np.pi / 2
    #     init_param_values[0] = 1
    #     var_principle = ImaginaryMcLachlanPrinciple()
    #
    #     param_dict = dict(zip(parameters, init_param_values))
    #
    #     time = 1
    #
    #     evolution_problem = TimeEvolutionProblem(observable, time, aux_operators=aux_ops)
    #
    #     thetas_expected_sv = [
    #         1.03612467538419,
    #         1.91891042963193,
    #         2.81129500883365,
    #         2.78938736703301,
    #         2.2215151699331,
    #         1.61953721158502,
    #         2.23490753161058,
    #         1.97145113701782,
    #     ]
    #
    #     thetas_expected_qasm = [
    #         1.03612467538419,
    #         1.91891042963193,
    #         2.81129500883365,
    #         2.78938736703301,
    #         2.2215151699331,
    #         1.61953721158502,
    #         2.23490753161058,
    #         1.97145113701782,
    #     ]
    #
    #     expected_aux_ops_evaluated_sv = [(-0.160899, 0.0), (0.26207, 0.0)]
    #     expected_aux_ops_evaluated_qasm = [
    #         (-0.1765, 0.015563),
    #         (0.2555, 0.015287),
    #     ]
    #
    #     for backend_name in self.backends_names:
    #         with self.subTest(msg=f"Test {backend_name} backend."):
    #             algorithm_globals.random_seed = self.seed
    #
    #             var_qite = VarQITE(
    #                 ansatz,
    #                 var_principle,
    #                 param_dict,
    #                 num_timesteps=25,
    #             )
    #             evolution_result = var_qite.evolve(evolution_problem)
    #
    #             evolved_state = evolution_result.evolved_state
    #             aux_ops = evolution_result.aux_ops_evaluated
    #
    #             parameter_values = evolved_state.data[0][0].params
    #
    #             if backend_name == "qi_qasm":
    #                 thetas_expected = thetas_expected_qasm
    #                 expected_aux_ops = expected_aux_ops_evaluated_qasm
    #             else:
    #                 thetas_expected = thetas_expected_sv
    #                 expected_aux_ops = expected_aux_ops_evaluated_sv
    #
    #             for i, parameter_value in enumerate(parameter_values):
    #                 np.testing.assert_almost_equal(
    #                     float(parameter_value), thetas_expected[i], decimal=3
    #                 )
    #
    #             np.testing.assert_array_almost_equal(aux_ops, expected_aux_ops)

    # @slow_test
    def test_run_d_1_t_7(self):
        """Test VarQITE for d = 1 and t = 7 with RK45 ODE solver."""

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

        d = 1
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(len(parameters)):
            init_param_values[i] = np.pi / 2
        init_param_values[0] = 1
        var_principle = ImaginaryMcLachlanPrinciple()

        time = 7
        var_qite = VarQITE(
            ansatz,
            var_principle,
            init_param_values,
            ode_solver="RK45",
            num_timesteps=25,
        )

        thetas_expected = [
            0.828917365718767,
            1.88481074798033,
            3.14111335991238,
            3.14125849601269,
            2.33768562678401,
            1.78670990729437,
            2.04214275514208,
            2.04009918594422,
        ]

        self._test_helper(ansatz, observable, thetas_expected, time, var_qite, 2)

    # @slow_test
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
        """Test VarQITE for d = 2 and t = 1 with RK45 ODE solver."""
        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        parameters = list(ansatz.parameters)
        init_param_values = np.zeros(len(parameters))
        for i in range(len(parameters)):
            init_param_values[i] = np.pi / 4

        var_principle = ImaginaryMcLachlanPrinciple()

        param_dict = dict(zip(parameters, init_param_values))

        time = 1
        var_qite = VarQITE(
            ansatz,
            var_principle,
            param_dict,
            ode_solver="RK45",
            num_timesteps=25,
        )

        thetas_expected = [
            1.29495364023786,
            1.08970061333559,
            0.667488228710748,
            0.500122687902944,
            1.4377736672043,
            1.22881086103085,
            0.729773048146251,
            1.01698854755226,
            0.050807780587492,
            0.294828474947149,
            0.839305697704923,
            0.663689581255428,
        ]

        self._test_helper(ansatz, observable, thetas_expected, time, var_qite, 4)

    def _test_helper(self, ansatz, observable, thetas_expected, time, var_qite, decimal):
        evolution_problem = TimeEvolutionProblem(observable, time)
        evolution_result = var_qite.evolve(evolution_problem)
        evolved_state = evolution_result.evolved_state

        parameter_values = evolved_state.data[0][0].params
        print(parameter_values)
        print(thetas_expected)
        print(
            state_fidelity(
                Statevector(evolved_state),
                Statevector(
                    ansatz.assign_parameters(dict(zip(list(ansatz.parameters), thetas_expected)))
                ),
            )
        )
        for i, parameter_value in enumerate(parameter_values):
            np.testing.assert_almost_equal(
                float(parameter_value), thetas_expected[i], decimal=decimal
            )


if __name__ == "__main__":
    unittest.main()
