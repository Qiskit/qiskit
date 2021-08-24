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
    # TODO use ddt
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

        # TODO var_principle currently requires an error calculator but linear solver for example
        #  does not need it at all.
        var_principle = ImaginaryMcLachlanVariationalPrinciple(observable, ansatz, parameters, None)

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

    def test_solve_sle_no_backend_faster(self):
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

        var_principle = ImaginaryMcLachlanVariationalPrinciple(observable, ansatz, parameters, None)

        param_dict = {}
        for param in parameters:
            param_dict[param] = 2.0

        nat_grad_res, grad_res, metric_res = linear_solver._solve_sle(
            var_principle, param_dict, faster=True
        )
        expected_nat_grad_res = [
            -0.34488717500197774,
            0.14588908124193561,
            -0.2210216075104149,
            -0.735875462872678,
            0.11486454689840815,
            -0.010299995704711518,
            0.15535228548184593,
            -0.11579257778498733,
            0.30178768002808387,
            -0.02024265717219398,
            0.15169629430111192,
            -0.06229051510060435,
        ]
        expected_grad_res = [
            -0.15007365667106218,
            0.03991100545511253,
            -0.17085654187610033,
            -0.382446898849228,
            0.03635119512659436,
            -0.03996239236507216,
            0.17481220621775215,
            -0.13770500491970517,
            0.16585187396458728,
            -0.0374531649935598,
            0.01845175514132781,
            -0.018451755141327755,
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
                0.03936762,
                -0.18795674,
                -0.03255001,
                -0.03255001,
                0.22552803,
                -0.06628068,
                0.00336165,
                -0.13413062,
                -0.08374172,
                0.04417312,
                0.04086418,
                -0.08593339,
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
                -0.0782176,
                -0.0782176,
                -0.0995654,
                -0.01354558,
                0.00336165,
                -0.00410525,
                0.24953822,
                -0.14893239,
                0.0089051,
                0.10132269,
                -0.10448773,
                0.0218178,
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
                -0.01578637,
                0.11027939,
                -0.04864503,
                -0.0539041,
                -0.08374172,
                -0.01035651,
                0.0089051,
                0.056361,
                0.07827161,
                -0.04867169,
                0.00869762,
                0.00475918,
            ],
            [
                -0.12650691,
                -0.06779215,
                0.01406411,
                0.06449774,
                0.04417312,
                0.04410809,
                0.10132269,
                -0.03840442,
                -0.04867169,
                0.17026532,
                -0.06099028,
                -0.01097413,
            ],
            [
                0.16510656,
                0.06354567,
                0.06330083,
                -0.07560117,
                0.04086418,
                0.10333948,
                -0.10448773,
                0.05790695,
                0.00869762,
                -0.06099028,
                0.24955949,
                0.11482226,
            ],
            [
                0.06467202,
                0.04315141,
                0.14877563,
                -0.0592728,
                -0.08593339,
                0.21339158,
                0.0218178,
                -0.00975702,
                0.00475918,
                -0.01097413,
                0.11482226,
                0.2484896,
            ],
        ]

        np.testing.assert_array_almost_equal(nat_grad_res, expected_nat_grad_res)
        np.testing.assert_array_almost_equal(grad_res, expected_grad_res)
        np.testing.assert_array_almost_equal(metric_res, expected_metric_res)


if __name__ == "__main__":
    unittest.main()
