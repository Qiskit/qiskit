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
""" Test Trotter Qrte. """
import unittest

import numpy as np
from numpy.testing import assert_raises

from qiskit import QiskitError
from qiskit.algorithms.quantum_time_evolution.builders.implementations.trotterizations \
    .trotter_mode_enum import (
    TrotterModeEnum,
)
from qiskit.algorithms.quantum_time_evolution.real.implementations.trotterization.trotter_qrte \
    import (
    TrotterQrte,
)
from qiskit.quantum_info import Statevector
from qiskit.utils import algorithm_globals
from test.python.opflow import QiskitOpflowTestCase
from qiskit.circuit import Parameter
from qiskit.opflow import (
    X,
    Z,
    Zero,
    VectorStateFn,
    StateFn,
    MatrixOp, I, PauliSumOp, Y,
)


class TestTrotterQrte(QiskitOpflowTestCase):
    """Trotter Qrte tests."""

    def test_trotter_qrte_trotter(self):
        """Test for trotter qrte."""
        operator = X + Z
        mode = TrotterModeEnum.TROTTER
        trotter_qrte = TrotterQrte(mode)
        initial_state = Zero
        evolved_state = trotter_qrte.evolve(operator, 1, initial_state)
        expected_evolved_state = VectorStateFn(
            Statevector([0.29192658 - 0.45464871j, -0.70807342 - 0.45464871j], dims=(2,))
        )

        np.testing.assert_equal(evolved_state, expected_evolved_state)

    def test_trotter_qrte_trotter_2(self):
        """Test for trotter qrte."""
        operator = X + Z
        mode = TrotterModeEnum.TROTTER
        trotter_qrte = TrotterQrte(mode)
        initial_state = StateFn([1, 0])
        evolved_state = trotter_qrte.evolve(operator, 1, initial_state)
        expected_evolved_state = VectorStateFn(
            Statevector([0.29192658 - 0.45464871j, -0.70807342 - 0.45464871j], dims=(2,))
        )

        np.testing.assert_equal(evolved_state, expected_evolved_state)

    def test_trotter_qrte_trotter_observable(self):
        """Test for trotter qrte with an observable."""
        operator = X + Z
        mode = TrotterModeEnum.TROTTER
        trotter_qrte = TrotterQrte(mode)
        observable = X
        evolved_observable = trotter_qrte.evolve(operator, 1, observable=observable)
        evolved_observable = evolved_observable.to_matrix_op()
        expected_evolved_observable = MatrixOp(
            [
                [2.99928030e-17 - 4.67110231e-17j, -4.16146837e-01 + 9.09297427e-01j],
                [-4.16146837e-01 - 9.09297427e-01j, 0.00000000e00 + 0.00000000e00j],
            ]
        )
        np.testing.assert_equal(evolved_observable, expected_evolved_observable)

    def test_trotter_qrte_suzuki(self):
        """Test for trotter qrte with Suzuki."""
        operator = X + Z
        mode = TrotterModeEnum.SUZUKI
        trotter_qrte = TrotterQrte(mode)
        initial_state = Zero
        evolved_state = trotter_qrte.evolve(operator, 1, initial_state)
        expected_evolved_state = VectorStateFn(
            Statevector([0.29192658 - 0.45464871j, 0.0 - 0.84147098j], dims=(2,))
        )

        np.testing.assert_equal(evolved_state, expected_evolved_state)

    def test_trotter_qrte_qdrift(self):
        """Test for trotter qrte with QDrift."""
        algorithm_globals.random_seed = 0
        operator = X + Z
        mode = TrotterModeEnum.QDRIFT
        trotter_qrte = TrotterQrte(mode)
        initial_state = Zero
        evolved_state = trotter_qrte.evolve(operator, 1, initial_state)
        expected_evolved_state = VectorStateFn(
            Statevector([0.23071786 - 0.69436148j, -0.4646314 - 0.49874749j], dims=(2,))
        )

        np.testing.assert_equal(evolved_state, expected_evolved_state)

    def test_trotter_qrte_trotter_binding(self):
        """Test for trotter qrte with binding."""
        t_param = Parameter("t")
        operator = X * t_param + Z
        hamiltonian_value_dict = {t_param: 1}
        mode = TrotterModeEnum.TROTTER
        trotter_qrte = TrotterQrte(mode)
        initial_state = Zero
        evolved_state = trotter_qrte.evolve(
            operator,
            1,
            initial_state,
            t_param=t_param,
            hamiltonian_value_dict=hamiltonian_value_dict,
        )
        expected_evolved_state = VectorStateFn(
            Statevector([0.29192658 - 0.45464871j, -0.70807342 - 0.45464871j], dims=(2,))
        )

        np.testing.assert_equal(evolved_state, expected_evolved_state)

    def test_trotter_qrte_trotter_binding_missing_dict(self):
        """Test for trotter qrte with binding and missing dictionary.."""
        t_param = Parameter("t")
        operator = X * t_param + Z
        mode = TrotterModeEnum.TROTTER
        trotter_qrte = TrotterQrte(mode)
        initial_state = Zero
        with assert_raises(ValueError):
            _ = trotter_qrte.evolve(operator, 1, initial_state, t_param=t_param)

    def test_trotter_qrte_trotter_binding_missing_param(self):
        """Test for trotter qrte with binding and missing param."""
        t_param = Parameter("t")
        operator = X * t_param + Z
        mode = TrotterModeEnum.TROTTER
        trotter_qrte = TrotterQrte(mode)
        initial_state = Zero
        with assert_raises(ValueError):
            _ = trotter_qrte.evolve(operator, 1, initial_state)

    def test_trotter_qrte_gradient_summed_op_qdrift(self):
        """Test for trotter qrte gradient with SummedOp and QDrift."""
        theta1 = Parameter('theta1')
        theta2 = Parameter('theta2')
        operator = theta1 * (I ^ Z ^ I) + theta2 * (Z ^ I ^ I)
        mode = TrotterModeEnum.QDRIFT
        trotter_qrte = TrotterQrte(mode)
        initial_state = Zero
        time = 5
        gradient_object = None
        observable = Z^Y^Z
        value_dict = {theta1: 2, theta2: 3}
        gradient = trotter_qrte.gradient(operator, time, initial_state, gradient_object,
                                         observable, hamiltonian_value_dict=value_dict,
                                         gradient_params=[theta1, theta2])

        print(gradient)

    def test_trotter_qrte_gradient_summed_op_qdrift_no_param(self):
        """Test for trotter qrte gradient with SummedOp and QDrift; missing param."""
        theta1 = Parameter('theta1')
        operator = theta1 * (I ^ Z ^ I) + 5 * (Z ^ I ^ I)
        mode = TrotterModeEnum.QDRIFT
        trotter_qrte = TrotterQrte(mode)
        initial_state = Zero
        time = 5
        gradient_object = None
        observable = Z
        value_dict = {theta1: 2}
        with assert_raises(ValueError):
            _ = trotter_qrte.gradient(operator, time, initial_state, gradient_object,
                                      observable, hamiltonian_value_dict=value_dict,
                                      gradient_params=[theta1])

    def test_trotter_qrte_gradient_summed_op_qdrift_t_param_grad(self):
        """Test for trotter qrte gradient with SummedOp and QDrift; missing param."""
        t_param = Parameter('t_param')
        theta1 = Parameter('theta1')
        operator = t_param * (I ^ Z ^ I) + theta1 * (Z ^ I ^ I)
        mode = TrotterModeEnum.QDRIFT
        trotter_qrte = TrotterQrte(mode)
        initial_state = Zero
        time = 5
        gradient_object = None
        observable = Z
        value_dict = {theta1: 2, t_param: 5}
        gradient = trotter_qrte.gradient(operator, time, initial_state, gradient_object,
                                         observable, t_param=t_param,
                                         hamiltonian_value_dict=value_dict,
                                         gradient_params=[t_param])

        print(gradient)


if __name__ == "__main__":
    unittest.main()
