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
from qiskit.algorithms.quantum_time_evolution.variational.principles.imaginary.implementations.imaginary_mc_lachlan_variational_principle import (
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
        backend = Aer.get_backend("statevector_simulator")
        linear_solver = VarQteLinearSolver(
            CircuitSampler(backend),
            CircuitSampler(backend),
            CircuitSampler(backend),
            backend=None,
        )

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
        param_dict = {}
        init_param_values = np.zeros(len(ansatz.ordered_parameters))
        for i in range(ansatz.num_qubits):
            init_param_values[-(ansatz.num_qubits + i + 1)] = np.pi / 2

        print(init_param_values)

        param_dict = dict(zip(parameters, init_param_values))

        var_principle = ImaginaryMcLachlanVariationalPrinciple()
        regularization = "ridge"
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, param_dict, regularization)

        nat_grad_res, grad_res, metric_res = linear_solver._solve_sle(var_principle, param_dict)
        # print(nat_grad_res)
        # print("nat_grad_res")
        # print(grad_res)
        # print("grad_res")
        # print(metric_res)
        # print("metric_res")

        # TODO verify all values below if correct
        expected_nat_grad_res = [
            3.438843e-01,
            -2.881146e-01,
            6.615561e-16,
            6.318103e-16,
            -9.425985e-01,
            -2.881146e-01,
            6.424706e-16,
            6.424706e-16,
            3.438843e-01,
            -2.881146e-01,
            3.464226e-03,
            3.464226e-03,
        ]
        expected_grad_res = [
            3.435e-01,
            -4.347e-01,
            -0.000e00,
            -0.000e00,
            -4.806e-01,
            -4.347e-01,
            -0.000e00,
            -0.000e00,
            3.435e-01,
            -4.347e-01,
            4.900e-17,
            -1.110e-16,
        ]
        # TODO extract
        expected_metric_res = [
            [
                2.50000000e-01,
                5.20093357e-18,
                -3.35902978e-18,
                1.86019655e-17,
                3.08989820e-19,
                1.62673464e-18,
                3.65194539e-18,
                1.70843275e-18,
                2.50000000e-01,
                7.53845584e-18,
                3.71390522e-17,
                -3.88242861e-18,
            ],
            [
                -2.44043303e-18,
                2.50000000e-01,
                9.49713925e-18,
                -1.93558798e-17,
                -7.59716016e-17,
                2.50000000e-01,
                -1.37379415e-18,
                4.07290740e-18,
                -7.97312577e-18,
                2.50000000e-01,
                1.24267272e-17,
                9.62477424e-17,
            ],
            [
                1.38750000e-17,
                1.68162736e-17,
                1.00000000e-10,
                2.68678230e-27,
                5.74148023e-18,
                1.68162736e-17,
                -1.76233561e-27,
                1.07015248e-26,
                1.38750000e-17,
                1.68162736e-17,
                1.22212870e-16,
                -7.76899169e-17,
            ],
            [
                9.48696076e-28,
                9.00931884e-18,
                -2.75966025e-27,
                1.00000000e-10,
                -3.13020095e-18,
                9.00931886e-18,
                -8.65845248e-28,
                -1.29583786e-27,
                -1.27199658e-28,
                9.00931887e-18,
                2.31165442e-17,
                -6.17038916e-17,
            ],
            [
                -3.19251920e-18,
                1.27361799e-17,
                -6.33975892e-18,
                -4.68773457e-19,
                2.50000000e-01,
                1.95646321e-17,
                7.66124586e-18,
                -3.25409625e-17,
                3.19251920e-18,
                6.38473223e-17,
                1.97870528e-17,
                1.29276939e-16,
            ],
            [
                -2.44043303e-18,
                2.50000000e-01,
                9.49713926e-18,
                -1.93558798e-17,
                -2.98696861e-17,
                2.50000000e-01,
                -1.37379416e-18,
                4.07290740e-18,
                -7.97312577e-18,
                2.50000000e-01,
                3.03554551e-18,
                5.75311897e-17,
            ],
            [
                3.10108007e-28,
                -4.69285617e-20,
                2.63404030e-27,
                1.27187501e-27,
                4.99306912e-18,
                -4.69285664e-20,
                1.00000000e-10,
                7.81063317e-28,
                -4.10903581e-28,
                -4.69285654e-20,
                2.79631793e-17,
                -6.06781522e-17,
            ],
            [
                -2.91983861e-27,
                2.39666271e-28,
                -2.01110019e-26,
                5.32072816e-28,
                -7.74154336e-18,
                -9.50083293e-28,
                -3.77605176e-27,
                1.00000000e-10,
                8.96857314e-28,
                2.46370766e-28,
                1.91756625e-17,
                -6.22304063e-17,
            ],
            [
                2.50000000e-01,
                -1.98212845e-18,
                -3.35902978e-18,
                1.86019655e-17,
                6.62990408e-18,
                -5.55632739e-18,
                3.65194539e-18,
                1.70843275e-18,
                2.50000000e-01,
                3.55393817e-19,
                1.84402255e-17,
                2.72308795e-18,
            ],
            [
                -2.44043303e-18,
                2.50000000e-01,
                9.49713926e-18,
                -1.93558798e-17,
                -3.42594832e-17,
                2.50000000e-01,
                -1.37379415e-18,
                4.07290740e-18,
                -7.97312577e-18,
                2.50000000e-01,
                1.36268505e-17,
                1.45448809e-16,
            ],
            [
                7.72276349e-17,
                1.60734049e-17,
                1.13383560e-16,
                2.30617572e-17,
                1.58897395e-17,
                1.94622276e-17,
                3.21205273e-17,
                1.06757231e-17,
                -1.46026349e-17,
                -6.40785700e-18,
                2.50000000e-01,
                -8.71660992e-17,
            ],
            [
                -2.72749598e-18,
                6.59572196e-17,
                -7.18239534e-17,
                -6.20295754e-17,
                1.02535223e-16,
                4.77979690e-17,
                -6.27695016e-17,
                -6.02807869e-17,
                9.77495976e-19,
                1.08287739e-16,
                -6.13448081e-17,
                2.50000000e-01,
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
    #     regularization = 'ridge'
    #     # for the purpose of the test we invoke lazy_init
    #     var_principle._lazy_init(observable, ansatz, param_dict, regularization)
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
