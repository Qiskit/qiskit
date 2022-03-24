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

""" Test TrotterQRTE. """

import unittest

from test.python.opflow import QiskitOpflowTestCase
from ddt import ddt, data
import numpy as np
from numpy.testing import assert_raises
from scipy.linalg import expm

from qiskit import BasicAer, QuantumCircuit
from qiskit.algorithms import EvolutionProblem
from qiskit.algorithms.evolvers.real.trotterization.trotter_qrte import (
    TrotterQRTE,
)
from qiskit.quantum_info import Statevector, SparsePauliOp, Pauli, PauliTable
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
class TestTrotterQRTE(QiskitOpflowTestCase):
    """TrotterQRTE tests."""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed
        backend = BasicAer.get_backend("statevector_simulator")
        backend_qasm = BasicAer.get_backend("qasm_simulator")  # TODO add to tests
        self.quantum_instance = QuantumInstance(
            backend=backend,
            shots=1,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        self.quantum_instance_qasm = QuantumInstance(
            backend=backend_qasm,
            shots=4000,
            seed_simulator=self.seed,
            seed_transpiler=self.seed,
        )
        self.backends_dict = {
            "qi_sv": self.quantum_instance,
            "qi_qasm": self.quantum_instance_qasm,
            "b_sv": backend,
            "None": None,
        }

    @data("qi_sv", "b_sv", "None")
    def test_trotter_qrte_trotter_single_qubit(self, quantum_instanc):
        """Test for default TrotterQRTE on a single qubit."""
        operator = SummedOp([X, Z])
        # LieTrotter with 1 rep
        quantum_instance = self.backends_dict[quantum_instanc]

        trotter_qrte = TrotterQRTE(quantum_instance=quantum_instance)
        initial_state = Zero
        evolution_problem = EvolutionProblem(operator, 1, initial_state)
        evolution_result = trotter_qrte.evolve(evolution_problem)
        # Calculate the expected state
        expected_state = (
            expm(-1j * Z.to_matrix()) @ expm(-1j * X.to_matrix()) @ initial_state.to_matrix()
        )
        expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2,)))

        np.testing.assert_equal(evolution_result.evolved_state, expected_evolved_state)

    @data("qi_sv", "b_sv")
    def test_trotter_qrte_trotter_single_qubit_aux_ops(self, quantum_instance):
        """Test for default TrotterQRTE on a single qubit with auxiliary operators."""
        operator = SummedOp([X, Z])
        # LieTrotter with 1 rep
        aux_ops = [X, Y]
        expectation = MatrixExpectation()
        quantum_instance = self.backends_dict[quantum_instance]

        trotter_qrte = TrotterQRTE(quantum_instance=quantum_instance, expectation=expectation)
        initial_state = Zero
        time = 3
        evolution_problem = EvolutionProblem(operator, time, initial_state, aux_ops)
        evolution_result = trotter_qrte.evolve(evolution_problem)
        # Calculate the expected state
        expected_state = (
            expm(-time * 1j * Z.to_matrix())
            @ expm(-time * 1j * X.to_matrix())
            @ initial_state.to_matrix()
        )
        expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2,)))
        expected_aux_ops_evaluated = [(0.078073, 0.0), (0.268286, 0.0)]

        np.testing.assert_equal(evolution_result.evolved_state, expected_evolved_state)
        np.testing.assert_array_almost_equal(
            evolution_result.aux_ops_evaluated, expected_aux_ops_evaluated
        )

    @data(
        SummedOp([(X ^ Y), (Y ^ X)]),
        (Z ^ Z) + (Z ^ I) + (I ^ Z),
        Y ^ Y,
        SparsePauliOp(Pauli("XI")),
        SparsePauliOp(PauliTable.from_labels(["XX", "ZZ"])),
    )
    def test_trotter_qrte_trotter_two_qubits(self, operator):
        """Test for TrotterQRTE on two qubits with various types of a Hamiltonian."""
        # LieTrotter with 1 rep
        trotter_qrte = TrotterQRTE(quantum_instance=self.quantum_instance)
        initial_state = StateFn([1, 0, 0, 0])
        # Calculate the expected state
        expected_state = initial_state.to_matrix()
        expected_state = expm(-1j * operator.to_matrix()) @ expected_state
        expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2, 2)))

        evolution_problem = EvolutionProblem(operator, 1, initial_state)
        evolution_result = trotter_qrte.evolve(evolution_problem)
        np.testing.assert_equal(evolution_result.evolved_state, expected_evolved_state)

    @data("qi_sv", "b_sv", "None")
    def test_trotter_qrte_suzuki(self, quantum_instance):
        """Test for TrotterQRTE with Suzuki."""
        operator = X + Z
        # 2nd order Suzuki with 1 rep
        quantum_instance = self.backends_dict[quantum_instance]
        trotter_qrte = TrotterQRTE(
            quantum_instance=quantum_instance, product_formula=SuzukiTrotter()
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

    @data("qi_sv", "b_sv", "None")
    def test_trotter_qrte_qdrift_fractional_time(self, quantum_instance):
        """Test for TrotterQRTE with QDrift."""
        algorithm_globals.random_seed = 0
        operator = SummedOp([X, Z])
        quantum_instance = self.backends_dict[quantum_instance]
        # QDrift with one repetition
        trotter_qrte = TrotterQRTE(quantum_instance=quantum_instance, product_formula=QDrift())
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

    @data("qi_sv", "b_sv", "None")
    def test_trotter_qrte_qdrift_circuit(self, quantum_instance):
        """Test for TrotterQRTE with QDrift."""
        algorithm_globals.random_seed = 0
        operator = SummedOp([X, Z])
        quantum_instance = self.backends_dict[quantum_instance]
        # QDrift with one repetition
        trotter_qrte = TrotterQRTE(quantum_instance=quantum_instance, product_formula=QDrift())
        initial_state = QuantumCircuit(1)
        initial_state.append(X, [0])
        evolution_problem = EvolutionProblem(operator, 1, initial_state)
        evolution_result = trotter_qrte.evolve(evolution_problem)
        sampled_ops = [Z, X, X, X, Z, Z, Z, Z]
        evo_time = 0.25
        # Calculate the expected state
        expected_state = StateFn(initial_state).to_matrix()
        for op in sampled_ops:
            expected_state = expm(-1j * op.to_matrix() * evo_time) @ expected_state
        expected_evolved_state = VectorStateFn(Statevector(expected_state, dims=(2,)))

        np.testing.assert_equal(evolution_result.evolved_state, expected_evolved_state)

    def test_trotter_qrte_trotter_binding_missing_dict(self):
        """Test for TrotterQRTE with binding and missing dictionary.."""
        t_param = Parameter("t")
        operator = X * t_param + Z
        trotter_qrte = TrotterQRTE(quantum_instance=self.quantum_instance)
        initial_state = Zero
        with assert_raises(ValueError):
            evolution_problem = EvolutionProblem(operator, 1, initial_state, t_param=t_param)
            _ = trotter_qrte.evolve(evolution_problem)

    def test_trotter_qrte_trotter_binding_missing_param(self):
        """Test for TrotterQRTE with binding and missing param."""
        t_param = Parameter("t")
        operator = X * t_param + Z
        trotter_qrte = TrotterQRTE(quantum_instance=self.quantum_instance)
        initial_state = Zero
        with assert_raises(ValueError):
            evolution_problem = EvolutionProblem(operator, 1, initial_state)
            _ = trotter_qrte.evolve(evolution_problem)


if __name__ == "__main__":
    unittest.main()
