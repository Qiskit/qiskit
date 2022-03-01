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

from test.python.opflow import QiskitOpflowTestCase
from ddt import ddt, data
import numpy as np
from numpy.testing import assert_raises
from scipy.linalg import expm

from qiskit.algorithms.quantum_time_evolution.real.implementations.\
    trotterization.trotter_qrte import (
    TrotterQrte,
)
from qiskit.quantum_info import Statevector
from qiskit.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit.opflow import (
    X,
    Z,
    Zero,
    VectorStateFn,
    StateFn,
    MatrixOp,
    I,
    Y,
)
from qiskit.synthesis import SuzukiTrotter, QDrift


@ddt
class TestTrotterQrte(QiskitOpflowTestCase):
    """Trotter Qrte tests."""

    def test_trotter_qrte_trotter(self):
        """Test for trotter qrte."""
        operator = X + Z
        # LieTrotter with 1 rep
        trotter_qrte = TrotterQrte()
        initial_state = Zero
        evolved_state = trotter_qrte.evolve(operator, 1, initial_state)
        # Calculate the expected state
        expected_state = (
            expm(-1j * Z.to_matrix()) @ expm(-1j * X.to_matrix()) @ initial_state.to_matrix()
        )
        expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2,)))

        np.testing.assert_equal(evolved_state, expected_evolved_state)

    @data((X ^ Y) + (Y ^ X), (Z ^ Z) + (Z ^ I) + (I ^ Z), Y ^ Y)
    def test_trotter_qrte_trotter_2(self, op):
        """Test for trotter qrte."""
        # LieTrotter with 1 rep
        trotter_qrte = TrotterQrte()
        initial_state = StateFn([1, 0, 0, 0])

        # Calculate the expected state
        expected_state = initial_state.to_matrix()
        expected_state = expm(-1j * op.to_matrix()) @ expected_state
        expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2, 2)))
        evolved_state = trotter_qrte.evolve(op, 1, initial_state)
        np.testing.assert_equal(evolved_state, expected_evolved_state)

    def test_trotter_qrte_trotter_observable(self):
        """Test for trotter qrte with an observable."""
        operator = X + Z
        # LieTrotter with 1 rep
        trotter_qrte = TrotterQrte()
        observable = X
        evolved_observable = trotter_qrte.evolve(operator, 1, observable=observable)
        evolved_observable = evolved_observable.to_matrix_op()
        # Calculate the expected operator
        expected_op = expm(-1j * Z.to_matrix()) @ expm(-1j * X.to_matrix())
        expected_evolved_observable = MatrixOp(
            expected_op.conj().T @ observable.to_matrix() @ expected_op
        )

        np.testing.assert_equal(evolved_observable, expected_evolved_observable)

    def test_trotter_qrte_suzuki(self):
        """Test for trotter qrte with Suzuki."""
        operator = X + Z
        # 2nd order Suzuki with 1 rep
        trotter_qrte = TrotterQrte(SuzukiTrotter())
        initial_state = Zero
        evolved_state = trotter_qrte.evolve(operator, 1, initial_state)
        # Calculate the expected state
        expected_state = (
            expm(-1j * X.to_matrix() * 0.5)
            @ expm(-1j * Z.to_matrix())
            @ expm(-1j * X.to_matrix() * 0.5)
            @ initial_state.to_matrix()
        )
        expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2,)))

        np.testing.assert_equal(evolved_state, expected_evolved_state)

    def test_trotter_qrte_qdrift(self):
        """Test for trotter qrte with QDrift."""
        algorithm_globals.random_seed = 0
        operator = X + Z
        # QDrift with one repetition
        trotter_qrte = TrotterQrte(QDrift())
        initial_state = Zero
        evolved_state = trotter_qrte.evolve(operator, 1, initial_state)
        sampled_ops = [Z, X, X, X, Z, Z, Z, Z]
        evo_time = 0.25
        # Calculate the expected state
        expected_state = initial_state.to_matrix()
        for op in sampled_ops:
            expected_state = expm(-1j * op.to_matrix() * evo_time) @ expected_state
        expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2,)))

        np.testing.assert_equal(evolved_state, expected_evolved_state)

    def test_trotter_qrte_trotter_binding_missing_dict(self):
        """Test for trotter qrte with binding and missing dictionary.."""
        t_param = Parameter("t")
        operator = X * t_param + Z
        trotter_qrte = TrotterQrte()
        initial_state = Zero
        with assert_raises(ValueError):
            _ = trotter_qrte.evolve(operator, 1, initial_state, t_param=t_param)

    def test_trotter_qrte_trotter_binding_missing_param(self):
        """Test for trotter qrte with binding and missing param."""
        t_param = Parameter("t")
        operator = X * t_param + Z
        trotter_qrte = TrotterQrte()
        initial_state = Zero
        with assert_raises(ValueError):
            _ = trotter_qrte.evolve(operator, 1, initial_state)


if __name__ == "__main__":
    unittest.main()
