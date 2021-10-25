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
from numpy import array

from qiskit.algorithms.quantum_time_evolution.variational.error_calculators.gradient_errors.imaginary_error_calculator import (
    ImaginaryErrorCalculator,
)
from qiskit.algorithms.quantum_time_evolution.variational.solvers.ode.error_based_ode_function_generator import (
    ErrorBaseOdeFunctionGenerator,
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


class TestErrorBasedOdeFunctionGenerator(QiskitAlgorithmsTestCase):
    def test_error_based_ode_fun(self):
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

        # Define a set of initial parameters
        parameters = ansatz.ordered_parameters

        operator = ~StateFn(observable) @ StateFn(ansatz)
        param_dict = {param: np.pi / 4 for param in parameters}
        backend = Aer.get_backend("qasm_simulator")
        state = operator[-1]

        h = operator.oplist[0].primitive * operator.oplist[0].coeff
        h_squared = h ** 2
        h_squared = ComposedOp([~StateFn(h_squared.reduce()), state])
        h_squared = PauliExpectation().convert(h_squared)

        error_calculator = ImaginaryErrorCalculator(
            h_squared, operator, CircuitSampler(backend), CircuitSampler(backend), param_dict
        )

        var_principle = ImaginaryMcLachlanVariationalPrinciple()
        regularization = "ridge"
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, param_dict, regularization)

        ode_function_generator = ErrorBaseOdeFunctionGenerator(
            error_calculator,
            param_dict,
            var_principle,
            CircuitSampler(backend),
            CircuitSampler(backend),
            CircuitSampler(backend),
        )

        qte_ode_function = ode_function_generator.error_based_ode_fun(1, param_dict.values())

        # TODO extract
        # TODO verify values if correct
        expected_qte_ode_function = (
            array(
                [
                    -0.20586541,
                    0.71865927,
                    0.1353771,
                    0.99934292,
                    -0.00631492,
                    -0.07428645,
                    -0.34469716,
                    -0.3218795,
                    0.81221089,
                    0.65605788,
                    -0.32451751,
                    -0.24383516,
                ]
            ),
            array(
                [
                    0.38617868,
                    0.01405535,
                    0.06385049,
                    -0.13620629,
                    0.15180743,
                    0.23783937,
                    -0.00240605,
                    -0.09977052,
                    -0.40357722,
                    -0.01045385,
                    0.04578581,
                    -0.04578581,
                ]
            ),
            array(
                [
                    [
                        2.50000000e-01,
                        1.20731505e-11,
                        9.23734973e-12,
                        -1.41196120e-11,
                        8.83883476e-02,
                        7.48163411e-12,
                        6.25000000e-02,
                        8.16538615e-12,
                        -1.41735435e-01,
                        3.12500000e-02,
                        1.00222087e-01,
                        -3.12500000e-02,
                    ],
                    [
                        1.20731213e-11,
                        2.50000000e-01,
                        7.08568147e-12,
                        9.18928324e-12,
                        -8.83883477e-02,
                        1.25000000e-01,
                        6.25000000e-02,
                        1.25000000e-01,
                        -8.45970869e-02,
                        7.54441738e-02,
                        1.48207521e-01,
                        2.00444174e-01,
                    ],
                    [
                        9.23738336e-12,
                        7.08569569e-12,
                        1.25000000e-01,
                        1.13699429e-12,
                        -4.41941738e-02,
                        6.25000000e-02,
                        1.19638348e-01,
                        6.25000000e-02,
                        -5.14514565e-02,
                        6.89720869e-02,
                        1.04933262e-02,
                        -6.89720869e-02,
                    ],
                    [
                        -1.41196209e-11,
                        9.18928475e-12,
                        1.13703340e-12,
                        1.25000000e-01,
                        -4.41941738e-02,
                        -6.25000000e-02,
                        3.12500000e-02,
                        1.25000000e-01,
                        5.14514565e-02,
                        -6.89720869e-02,
                        7.81250000e-03,
                        1.94162607e-02,
                    ],
                    [
                        8.83883476e-02,
                        -8.83883477e-02,
                        -4.41941738e-02,
                        -4.41941738e-02,
                        2.34375000e-01,
                        -1.10485435e-01,
                        -2.02014565e-02,
                        -4.41941738e-02,
                        1.49547935e-02,
                        -2.24896848e-02,
                        -1.42172278e-03,
                        -1.23822206e-01,
                    ],
                    [
                        7.48165147e-12,
                        1.25000000e-01,
                        6.25000000e-02,
                        -6.25000000e-02,
                        -1.10485435e-01,
                        2.18750000e-01,
                        -2.68082617e-03,
                        -2.63374080e-13,
                        -1.57197815e-01,
                        2.53331304e-02,
                        9.82311964e-03,
                        1.06138957e-01,
                    ],
                    [
                        6.25000000e-02,
                        6.25000000e-02,
                        1.19638348e-01,
                        3.12500000e-02,
                        -2.02014565e-02,
                        -2.68082617e-03,
                        2.23881674e-01,
                        1.37944174e-01,
                        -3.78033966e-02,
                        1.58423239e-01,
                        1.34535646e-01,
                        -5.49651086e-02,
                    ],
                    [
                        8.16540703e-12,
                        1.25000000e-01,
                        6.25000000e-02,
                        1.25000000e-01,
                        -4.41941738e-02,
                        -2.63360930e-13,
                        1.37944174e-01,
                        2.50000000e-01,
                        -1.75163714e-12,
                        1.95295670e-11,
                        9.75412607e-02,
                        5.71383476e-02,
                    ],
                    [
                        -1.41735435e-01,
                        -8.45970869e-02,
                        -5.14514565e-02,
                        5.14514565e-02,
                        1.49547935e-02,
                        -1.57197815e-01,
                        -3.78033966e-02,
                        -1.75163755e-12,
                        1.95283753e-01,
                        -3.82941440e-02,
                        -6.11392595e-02,
                        -4.51588288e-02,
                    ],
                    [
                        3.12500000e-02,
                        7.54441738e-02,
                        6.89720869e-02,
                        -6.89720869e-02,
                        -2.24896848e-02,
                        2.53331304e-02,
                        1.58423239e-01,
                        1.95295392e-11,
                        -3.82941440e-02,
                        2.17629701e-01,
                        1.32431810e-01,
                        -1.91961467e-02,
                    ],
                    [
                        1.00222087e-01,
                        1.48207521e-01,
                        1.04933262e-02,
                        7.81250000e-03,
                        -1.42172278e-03,
                        9.82311964e-03,
                        1.34535646e-01,
                        9.75412607e-02,
                        -6.11392595e-02,
                        1.32431810e-01,
                        1.81683746e-01,
                        7.28902444e-02,
                    ],
                    [
                        -3.12500000e-02,
                        2.00444174e-01,
                        -6.89720869e-02,
                        1.94162607e-02,
                        -1.23822206e-01,
                        1.06138957e-01,
                        -5.49651086e-02,
                        5.71383476e-02,
                        -4.51588288e-02,
                        -1.91961467e-02,
                        7.28902444e-02,
                        2.38616353e-01,
                    ],
                ]
            ),
        )

        expected_qte_ode_function_1 = expected_qte_ode_function[0]
        expected_qte_ode_function_2 = expected_qte_ode_function[1]
        expected_qte_ode_function_3 = expected_qte_ode_function[2]

        np.testing.assert_almost_equal(expected_qte_ode_function_1, qte_ode_function[0])
        np.testing.assert_almost_equal(expected_qte_ode_function_2, qte_ode_function[1])
        np.testing.assert_almost_equal(expected_qte_ode_function_3, qte_ode_function[2])


if __name__ == "__main__":
    unittest.main()
