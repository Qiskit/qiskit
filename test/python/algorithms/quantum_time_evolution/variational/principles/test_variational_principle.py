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
from qiskit.circuit.library import EfficientSU2
from qiskit.opflow import SummedOp, X, Y, I, Z
from test.python.algorithms import QiskitAlgorithmsTestCase


class TestVariationalPrinciple(QiskitAlgorithmsTestCase):
    def test_lazy_init(self):
        observable = SummedOp(
            [
                0.2252 * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        )

        d = 2
        ansatz = EfficientSU2(observable.num_qubits, reps=d)

        # Define a set of initial parameters
        parameters = ansatz.ordered_parameters
        param_dict = {param: np.pi / 4 for param in parameters}
        var_principle = ImaginaryMcLachlanVariationalPrinciple()
        # for the purpose of the test we invoke lazy_init
        var_principle._lazy_init(observable, ansatz, param_dict)

        np.testing.assert_equal(var_principle._hamiltonian, observable)
        np.testing.assert_equal(var_principle._ansatz, ansatz)
        np.testing.assert_equal(var_principle._param_dict, param_dict)

        assert var_principle._operator is not None
        assert var_principle._metric_tensor is not None
        assert var_principle._evolution_grad is not None

    def test_op_real_part(self):
        observable = SummedOp(
            [
                0.2252j * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        )

        real_part = ImaginaryMcLachlanVariationalPrinciple.op_real_part(observable).reduce()

        expected_real_part = SummedOp(
            [
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091 * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        )

        np.testing.assert_equal(real_part.to_matrix(), expected_real_part.to_matrix())

    def test_op_imag_part(self):
        observable = SummedOp(
            [
                0.2252j * (I ^ I),
                0.5716 * (Z ^ Z),
                0.3435 * (I ^ Z),
                -0.4347 * (Z ^ I),
                0.091j * (Y ^ Y),
                0.091 * (X ^ X),
            ]
        )

        imag_part = ImaginaryMcLachlanVariationalPrinciple.op_imag_part(observable).reduce()

        expected_imag_part = SummedOp(
            [
                0.2252 * (I ^ I),
                0.091 * (Y ^ Y),
            ]
        )

        np.testing.assert_equal(imag_part.to_matrix(), expected_imag_part.to_matrix())


if __name__ == "__main__":
    unittest.main()
