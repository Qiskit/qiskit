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

from qiskit import Aer
from qiskit.algorithms.quantum_time_evolution.variational.principles.imaginary.implementations \
    .imaginary_mc_lachlan_variational_principle import (
    ImaginaryMcLachlanVariationalPrinciple,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.var_qte_linear_solver import (
    VarQteLinearSolver,
)
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z, CircuitSampler
from test.python.algorithms import QiskitAlgorithmsTestCase


class TestLinearSolver(QiskitAlgorithmsTestCase):
    # TODO use ddt
    def test_solve_sle_no_backend(self):
        backend = Aer.get_backend("qasm_simulator")
        linear_solver = VarQteLinearSolver(CircuitSampler(backend), CircuitSampler(backend),
                                           CircuitSampler(backend), backend=None)

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

        # TODO var_principle currently requires an error calculator but linear solver for example
        #  does not need it at all.
        var_principle = ImaginaryMcLachlanVariationalPrinciple(None)
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, parameters)

        param_dict = {}
        for param in parameters:
            param_dict[param] = 2.0

        nat_grad_res, grad_res, metric_res = linear_solver._solve_sle(var_principle, param_dict)
        expected_nat_grad_res = [
            1.41355699,
            -0.61865659,
            0.63638228,
            1.93359943,
            -0.57264552,
            -0.21515823,
            -0.47842443,
            -0.05039853,
            -0.82424299,
            -0.65752459,
            -0.77788125,
            0.32882854,
        ]
        expected_grad_res = [
            -0.15007366,
            0.03991101,
            -0.17085654,
            -0.3824469,
            0.0363512,
            -0.03996239,
            0.17481221,
            -0.137705,
            0.16585187,
            -0.03745316,
            0.01845176,
            -0.01845176,
        ]
        # TODO extract
        expected_metric_res = [
            [
                2.50000000e-01,
                5.06965122e-12,
                1.07454665e-11,
                -2.09039175e-11,
                3.93676205e-02,
                7.06187142e-12,
                -7.82176012e-02,
                6.80873114e-12,
                -1.57863658e-02,
                -1.26506909e-01,
                1.65106556e-01,
                6.46720186e-02,
            ],
            [
                5.06953669e-12,
                2.50000000e-01,
                1.27513345e-11,
                2.83192371e-12,
                -1.87956736e-01,
                4.32945474e-02,
                -7.82176012e-02,
                2.06705453e-01,
                1.10279385e-01,
                -6.77921508e-02,
                6.35456679e-02,
                4.31514128e-02,
            ],
            [
                1.07454614e-11,
                1.27513519e-11,
                2.06705453e-01,
                -7.56968865e-13,
                -3.25500073e-02,
                1.70908577e-01,
                -9.95654027e-02,
                3.57968761e-02,
                -4.86450288e-02,
                1.40641070e-02,
                6.33008294e-02,
                1.48775630e-01,
            ],
            [
                -2.09039684e-11,
                2.83197244e-12,
                -7.56899653e-13,
                2.06705453e-01,
                -3.25500073e-02,
                -3.57968761e-02,
                -1.35455826e-02,
                7.15937521e-02,
                -5.39040990e-02,
                6.44977407e-02,
                -7.56011691e-02,
                -5.92728045e-02,
            ],
            [
                3.93676205e-02,
                -1.87956736e-01,
                -3.25500073e-02,
                -3.25500073e-02,
                2.25528028e-01,
                -6.62806765e-02,
                3.36164862e-03,
                -1.34130624e-01,
                -8.37417224e-02,
                4.41731239e-02,
                4.08641785e-02,
                -8.59333896e-02,
            ],
            [
                7.06178356e-12,
                4.32945474e-02,
                1.70908577e-01,
                -3.57968761e-02,
                -6.62806765e-02,
                2.20402362e-01,
                -4.10524541e-03,
                2.33983997e-02,
                -1.03565134e-02,
                4.41080941e-02,
                1.03339479e-01,
                2.13391583e-01,
            ],
            [
                -7.82176012e-02,
                -7.82176012e-02,
                -9.95654027e-02,
                -1.35455826e-02,
                3.36164862e-03,
                -4.10524541e-03,
                2.49538219e-01,
                -1.48932393e-01,
                8.90510145e-03,
                1.01322686e-01,
                -1.04487731e-01,
                2.18178041e-02,
            ],
            [
                6.80859821e-12,
                2.06705453e-01,
                3.57968761e-02,
                7.15937521e-02,
                -1.34130624e-01,
                2.33983997e-02,
                -1.48932393e-01,
                2.31502405e-01,
                5.63609979e-02,
                -3.84044239e-02,
                5.79069506e-02,
                -9.75702362e-03,
            ],
            [
                -1.57863658e-02,
                1.10279385e-01,
                -4.86450288e-02,
                -5.39040990e-02,
                -8.37417224e-02,
                -1.03565134e-02,
                8.90510145e-03,
                5.63609979e-02,
                7.82716135e-02,
                -4.86716864e-02,
                8.69762008e-03,
                4.75918239e-03,
            ],
            [
                -1.26506909e-01,
                -6.77921508e-02,
                1.40641070e-02,
                6.44977407e-02,
                4.41731239e-02,
                4.41080941e-02,
                1.01322686e-01,
                -3.84044239e-02,
                -4.86716864e-02,
                1.70265321e-01,
                -6.09902757e-02,
                -1.09741255e-02,
            ],
            [
                1.65106556e-01,
                6.35456679e-02,
                6.33008294e-02,
                -7.56011691e-02,
                4.08641785e-02,
                1.03339479e-01,
                -1.04487731e-01,
                5.79069506e-02,
                8.69762008e-03,
                -6.09902757e-02,
                2.49559487e-01,
                1.14822259e-01,
            ],
            [
                6.46720186e-02,
                4.31514128e-02,
                1.48775630e-01,
                -5.92728045e-02,
                -8.59333896e-02,
                2.13391583e-01,
                2.18178041e-02,
                -9.75702362e-03,
                4.75918239e-03,
                -1.09741255e-02,
                1.14822259e-01,
                2.48489598e-01,
            ],
        ]

        np.testing.assert_array_almost_equal(nat_grad_res, expected_nat_grad_res)
        np.testing.assert_array_almost_equal(grad_res, expected_grad_res)
        np.testing.assert_array_almost_equal(metric_res, expected_metric_res)

    # # TODO causes bad eigenvalue with backend
    # def test_solve_sle_with_backend(self):
    #     backend = Aer.get_backend("qasm_simulator")
    #     linear_solver = VarQteLinearSolver(CircuitSampler(backend), CircuitSampler(backend),
    #                                        CircuitSampler(backend), backend=backend)
    #
    #     # Define the Hamiltonian for the simulation
    #     observable = SummedOp(
    #         [
    #             0.2252 * (I ^ I),
    #             0.5716 * (Z ^ Z),
    #             0.3435 * (I ^ Z),
    #             -0.4347 * (Z ^ I),
    #             0.091 * (Y ^ Y),
    #             0.091 * (X ^ X),
    #         ]
    #     ).reduce()
    #     # Define Ansatz
    #     # ansatz = RealAmplitudes(observable.num_qubits, reps=d)
    #     d = 2
    #     ansatz = EfficientSU2(observable.num_qubits, reps=d)
    #
    #     # Define a set of initial parameters
    #     parameters = ansatz.ordered_parameters
    #
    #     # TODO var_principle currently requires an error calculator but linear solver for example
    #     #  does not need it at all.
    #     var_principle = ImaginaryMcLachlanVariationalPrinciple(None)
    #     # for the purpose of the test we invoke lazy_init
    #     var_principle._lazy_init(observable, ansatz, parameters)
    #
    #     param_dict = {}
    #     for param in parameters:
    #         param_dict[param] = 2.0
    #
    #     nat_grad_res, grad_res, metric_res = linear_solver._solve_sle(var_principle, param_dict)
    #     expected_nat_grad_res = [
    #         1.41355699,
    #         -0.61865659,
    #         0.63638228,
    #         1.93359943,
    #         -0.57264552,
    #         -0.21515823,
    #         -0.47842443,
    #         -0.05039853,
    #         -0.82424299,
    #         -0.65752459,
    #         -0.77788125,
    #         0.32882854,
    #     ]
    #     expected_grad_res = [
    #         -0.15007366,
    #         0.03991101,
    #         -0.17085654,
    #         -0.3824469,
    #         0.0363512,
    #         -0.03996239,
    #         0.17481221,
    #         -0.137705,
    #         0.16585187,
    #         -0.03745316,
    #         0.01845176,
    #         -0.01845176,
    #     ]
    #     # TODO extract
    #     expected_metric_res = [
    #         [
    #             2.50000000e-01,
    #             5.06965122e-12,
    #             1.07454665e-11,
    #             -2.09039175e-11,
    #             3.93676205e-02,
    #             7.06187142e-12,
    #             -7.82176012e-02,
    #             6.80873114e-12,
    #             -1.57863658e-02,
    #             -1.26506909e-01,
    #             1.65106556e-01,
    #             6.46720186e-02,
    #         ],
    #         [
    #             5.06953669e-12,
    #             2.50000000e-01,
    #             1.27513345e-11,
    #             2.83192371e-12,
    #             -1.87956736e-01,
    #             4.32945474e-02,
    #             -7.82176012e-02,
    #             2.06705453e-01,
    #             1.10279385e-01,
    #             -6.77921508e-02,
    #             6.35456679e-02,
    #             4.31514128e-02,
    #         ],
    #         [
    #             1.07454614e-11,
    #             1.27513519e-11,
    #             2.06705453e-01,
    #             -7.56968865e-13,
    #             -3.25500073e-02,
    #             1.70908577e-01,
    #             -9.95654027e-02,
    #             3.57968761e-02,
    #             -4.86450288e-02,
    #             1.40641070e-02,
    #             6.33008294e-02,
    #             1.48775630e-01,
    #         ],
    #         [
    #             -2.09039684e-11,
    #             2.83197244e-12,
    #             -7.56899653e-13,
    #             2.06705453e-01,
    #             -3.25500073e-02,
    #             -3.57968761e-02,
    #             -1.35455826e-02,
    #             7.15937521e-02,
    #             -5.39040990e-02,
    #             6.44977407e-02,
    #             -7.56011691e-02,
    #             -5.92728045e-02,
    #         ],
    #         [
    #             3.93676205e-02,
    #             -1.87956736e-01,
    #             -3.25500073e-02,
    #             -3.25500073e-02,
    #             2.25528028e-01,
    #             -6.62806765e-02,
    #             3.36164862e-03,
    #             -1.34130624e-01,
    #             -8.37417224e-02,
    #             4.41731239e-02,
    #             4.08641785e-02,
    #             -8.59333896e-02,
    #         ],
    #         [
    #             7.06178356e-12,
    #             4.32945474e-02,
    #             1.70908577e-01,
    #             -3.57968761e-02,
    #             -6.62806765e-02,
    #             2.20402362e-01,
    #             -4.10524541e-03,
    #             2.33983997e-02,
    #             -1.03565134e-02,
    #             4.41080941e-02,
    #             1.03339479e-01,
    #             2.13391583e-01,
    #         ],
    #         [
    #             -7.82176012e-02,
    #             -7.82176012e-02,
    #             -9.95654027e-02,
    #             -1.35455826e-02,
    #             3.36164862e-03,
    #             -4.10524541e-03,
    #             2.49538219e-01,
    #             -1.48932393e-01,
    #             8.90510145e-03,
    #             1.01322686e-01,
    #             -1.04487731e-01,
    #             2.18178041e-02,
    #         ],
    #         [
    #             6.80859821e-12,
    #             2.06705453e-01,
    #             3.57968761e-02,
    #             7.15937521e-02,
    #             -1.34130624e-01,
    #             2.33983997e-02,
    #             -1.48932393e-01,
    #             2.31502405e-01,
    #             5.63609979e-02,
    #             -3.84044239e-02,
    #             5.79069506e-02,
    #             -9.75702362e-03,
    #         ],
    #         [
    #             -1.57863658e-02,
    #             1.10279385e-01,
    #             -4.86450288e-02,
    #             -5.39040990e-02,
    #             -8.37417224e-02,
    #             -1.03565134e-02,
    #             8.90510145e-03,
    #             5.63609979e-02,
    #             7.82716135e-02,
    #             -4.86716864e-02,
    #             8.69762008e-03,
    #             4.75918239e-03,
    #         ],
    #         [
    #             -1.26506909e-01,
    #             -6.77921508e-02,
    #             1.40641070e-02,
    #             6.44977407e-02,
    #             4.41731239e-02,
    #             4.41080941e-02,
    #             1.01322686e-01,
    #             -3.84044239e-02,
    #             -4.86716864e-02,
    #             1.70265321e-01,
    #             -6.09902757e-02,
    #             -1.09741255e-02,
    #         ],
    #         [
    #             1.65106556e-01,
    #             6.35456679e-02,
    #             6.33008294e-02,
    #             -7.56011691e-02,
    #             4.08641785e-02,
    #             1.03339479e-01,
    #             -1.04487731e-01,
    #             5.79069506e-02,
    #             8.69762008e-03,
    #             -6.09902757e-02,
    #             2.49559487e-01,
    #             1.14822259e-01,
    #         ],
    #         [
    #             6.46720186e-02,
    #             4.31514128e-02,
    #             1.48775630e-01,
    #             -5.92728045e-02,
    #             -8.59333896e-02,
    #             2.13391583e-01,
    #             2.18178041e-02,
    #             -9.75702362e-03,
    #             4.75918239e-03,
    #             -1.09741255e-02,
    #             1.14822259e-01,
    #             2.48489598e-01,
    #         ],
    #     ]
    #
    #     np.testing.assert_array_almost_equal(nat_grad_res, expected_nat_grad_res)
    #     np.testing.assert_array_almost_equal(grad_res, expected_grad_res)
    #     np.testing.assert_array_almost_equal(metric_res, expected_metric_res)


if __name__ == "__main__":
    unittest.main()
