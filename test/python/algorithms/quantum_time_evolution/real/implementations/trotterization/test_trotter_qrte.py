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

from qiskit import BasicAer
from qiskit.algorithms import EvolutionProblem
from qiskit.algorithms.evolvers.real.trotterization.trotter_qrte import (
    TrotterQrte,
)
from qiskit.quantum_info import Statevector
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.circuit import Parameter
from qiskit.opflow import (
    X,
    Z,
    Zero,
    VectorStateFn,
    StateFn,
    I,
    Y,
    MatrixExpectation,
    SummedOp,
)
from qiskit.synthesis import SuzukiTrotter, QDrift


@ddt
class TestTrotterQrte(QiskitOpflowTestCase):
    """Trotter Qrte tests."""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        shots = 1
        backend = BasicAer.get_backend("statevector_simulator")
        self.quantum_instance = QuantumInstance(
            backend=backend,
            shots=shots,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )

    def test_trotter_qrte_trotter(self):
        """Test for trotter qrte."""
        operator = SummedOp([X, Z])
        # LieTrotter with 1 rep

        trotter_qrte = TrotterQrte(quantum_instance=self.quantum_instance)
        initial_state = Zero
        evolution_problem = EvolutionProblem(operator, 1, initial_state)
        evolution_result = trotter_qrte.evolve(evolution_problem)
        # Calculate the expected state
        expected_state = (
            expm(-1j * Z.to_matrix()) @ expm(-1j * X.to_matrix()) @ initial_state.to_matrix()
        )
        expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2,)))

        np.testing.assert_equal(evolution_result.evolved_state, expected_evolved_state)

    def test_trotter_qrte_trotter_aux_ops(self):
        """Test for trotter qrte."""
        operator = SummedOp([X, Z])
        # LieTrotter with 1 rep
        aux_ops = [X, Y]
        expectation = MatrixExpectation()
        trotter_qrte = TrotterQrte(quantum_instance=self.quantum_instance, expectation=expectation)
        initial_state = Zero
        evolution_problem = EvolutionProblem(operator, 1, initial_state, aux_ops)
        evolution_result = trotter_qrte.evolve(evolution_problem)
        # Calculate the expected state
        expected_state = (
            expm(-1j * Z.to_matrix()) @ expm(-1j * X.to_matrix()) @ initial_state.to_matrix()
        )
        expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2,)))
        expected_aux_ops_evaluated = [(0.8268218104318058, 0.0), (0.3784012476539641, 0.0)]

        np.testing.assert_equal(evolution_result.evolved_state, expected_evolved_state)
        np.testing.assert_array_almost_equal(
            evolution_result.aux_ops_evaluated, expected_aux_ops_evaluated
        )

    @data(SummedOp([(X ^ Y), (Y ^ X)]), SummedOp([(Z ^ Z), (Z ^ I), (I ^ Z)]), Y ^ Y)
    def test_trotter_qrte_trotter_2(self, operator):
        """Test for trotter qrte."""
        # LieTrotter with 1 rep
        trotter_qrte = TrotterQrte(quantum_instance=self.quantum_instance)
        initial_state = StateFn([1, 0, 0, 0])

        # Calculate the expected state
        expected_state = initial_state.to_matrix()
        expected_state = expm(-1j * operator.to_matrix()) @ expected_state
        expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2, 2)))

        evolution_problem = EvolutionProblem(operator, 1, initial_state)
        evolution_result = trotter_qrte.evolve(evolution_problem)
        np.testing.assert_equal(evolution_result.evolved_state, expected_evolved_state)

    def test_trotter_qrte_suzuki(self):
        """Test for trotter qrte with Suzuki."""
        operator = SummedOp([X, Z])
        # 2nd order Suzuki with 1 rep
        trotter_qrte = TrotterQrte(
            quantum_instance=self.quantum_instance, product_formula=SuzukiTrotter()
        )
        initial_state = Zero
        evolution_problem = EvolutionProblem(operator, 1, initial_state)
        evolution_result = trotter_qrte.evolve(evolution_problem)
        # Calculate the expected state
        expected_state = (
            expm(-1j * X.to_matrix() * 0.5)
            @ expm(-1j * Z.to_matrix())
            @ expm(-1j * X.to_matrix() * 0.5)
            @ initial_state.to_matrix()
        )
        expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2,)))

        np.testing.assert_equal(evolution_result.evolved_state, expected_evolved_state)

    def test_trotter_qrte_qdrift(self):
        """Test for trotter qrte with QDrift."""
        algorithm_globals.random_seed = 0
        operator = SummedOp([X, Z])
        # QDrift with one repetition
        trotter_qrte = TrotterQrte(quantum_instance=self.quantum_instance, product_formula=QDrift())
        initial_state = Zero
        evolution_problem = EvolutionProblem(operator, 1, initial_state)
        evolution_result = trotter_qrte.evolve(evolution_problem)
        sampled_ops = [Z, X, X, X, Z, Z, Z, Z]
        evo_time = 0.25
        # Calculate the expected state
        expected_state = initial_state.to_matrix()
        for op in sampled_ops:
            expected_state = expm(-1j * op.to_matrix() * evo_time) @ expected_state
        expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2,)))

        np.testing.assert_equal(evolution_result.evolved_state, expected_evolved_state)

    def test_trotter_qrte_trotter_binding_missing_dict(self):
        """Test for trotter qrte with binding and missing dictionary.."""
        t_param = Parameter("t")
        operator = X * t_param + Z
        trotter_qrte = TrotterQrte(quantum_instance=self.quantum_instance)
        initial_state = Zero
        evolution_problem = EvolutionProblem(operator, 1, initial_state, t_param=t_param)
        with assert_raises(ValueError):
            _ = trotter_qrte.evolve(evolution_problem)

    def test_trotter_qrte_trotter_binding_missing_param(self):
        """Test for trotter qrte with binding and missing param."""
        t_param = Parameter("t")
        operator = X * t_param + Z
        trotter_qrte = TrotterQrte(quantum_instance=self.quantum_instance)
        initial_state = Zero
        evolution_problem = EvolutionProblem(operator, 1, initial_state)
        with assert_raises(ValueError):
            _ = trotter_qrte.evolve(evolution_problem)


if __name__ == "__main__":
    unittest.main()
