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

""" Test TrotterQRTE. """

import unittest

from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import ddt, data, unpack
import numpy as np
from numpy.testing import assert_raises

from qiskit.algorithms.time_evolvers.time_evolution_problem import TimeEvolutionProblem
from qiskit.algorithms.time_evolvers.trotterization.trotter_qrte import TrotterQRTE
from qiskit.primitives import Estimator
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate
from qiskit.quantum_info import Statevector, Pauli, SparsePauliOp
from qiskit.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit.opflow import PauliSumOp, X, MatrixOp
from qiskit.synthesis import SuzukiTrotter, QDrift


@ddt
class TestTrotterQRTE(QiskitAlgorithmsTestCase):
    """TrotterQRTE tests."""

    def setUp(self):
        super().setUp()
        self.seed = 50
        algorithm_globals.random_seed = self.seed

    @data(
        (
            None,
            Statevector([0.29192658 - 0.45464871j, 0.70807342 - 0.45464871j]),
        ),
        (
            SuzukiTrotter(),
            Statevector([0.29192658 - 0.84147098j, 0.0 - 0.45464871j]),
        ),
    )
    @unpack
    def test_trotter_qrte_trotter_single_qubit(self, product_formula, expected_state):
        """Test for default TrotterQRTE on a single qubit."""
        operator = PauliSumOp(SparsePauliOp([Pauli("X"), Pauli("Z")]))
        initial_state = QuantumCircuit(1)
        time = 1
        evolution_problem = TimeEvolutionProblem(operator, time, initial_state)

        trotter_qrte = TrotterQRTE(product_formula=product_formula)
        evolution_result_state_circuit = trotter_qrte.evolve(evolution_problem).evolved_state

        np.testing.assert_array_almost_equal(
            Statevector.from_instruction(evolution_result_state_circuit).data, expected_state.data
        )

    def test_trotter_qrte_trotter_single_qubit_aux_ops(self):
        """Test for default TrotterQRTE on a single qubit with auxiliary operators."""
        operator = PauliSumOp(SparsePauliOp([Pauli("X"), Pauli("Z")]))
        # LieTrotter with 1 rep
        aux_ops = [Pauli("X"), Pauli("Y")]

        initial_state = QuantumCircuit(1)
        time = 3
        evolution_problem = TimeEvolutionProblem(operator, time, initial_state, aux_ops)
        estimator = Estimator()

        expected_evolved_state = Statevector([0.98008514 + 0.13970775j, 0.01991486 + 0.13970775j])

        algorithm_globals.random_seed = 0
        trotter_qrte = TrotterQRTE(estimator=estimator)
        evolution_result = trotter_qrte.evolve(evolution_problem)

        np.testing.assert_array_almost_equal(
            Statevector.from_instruction(evolution_result.evolved_state).data,
            expected_evolved_state.data,
        )

        aux_ops_result = evolution_result.aux_ops_evaluated
        expected_aux_ops_result = [
            (0.078073, {"variance": 0, "shots": 0}),
            (0.268286, {"variance": 0, "shots": 0}),
        ]

        means = [element[0] for element in aux_ops_result]
        expected_means = [element[0] for element in expected_aux_ops_result]
        np.testing.assert_array_almost_equal(means, expected_means)

        vars_and_shots = [element[1] for element in aux_ops_result]
        expected_vars_and_shots = [element[1] for element in expected_aux_ops_result]

        for computed, expected in zip(vars_and_shots, expected_vars_and_shots):
            self.assertAlmostEqual(computed.pop("variance", 0), expected["variance"], 2)
            self.assertEqual(computed.pop("shots", 0), expected["shots"])

    @data(
        (
            PauliSumOp(SparsePauliOp([Pauli("XY"), Pauli("YX")])),
            Statevector([-0.41614684 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.90929743 + 0.0j]),
        ),
        (
            PauliSumOp(SparsePauliOp([Pauli("ZZ"), Pauli("ZI"), Pauli("IZ")])),
            Statevector([-0.9899925 - 0.14112001j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j]),
        ),
        (
            Pauli("YY"),
            Statevector([0.54030231 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.84147098j]),
        ),
    )
    @unpack
    def test_trotter_qrte_trotter_two_qubits(self, operator, expected_state):
        """Test for TrotterQRTE on two qubits with various types of a Hamiltonian."""
        # LieTrotter with 1 rep
        initial_state = QuantumCircuit(2)
        evolution_problem = TimeEvolutionProblem(operator, 1, initial_state)

        trotter_qrte = TrotterQRTE()
        evolution_result = trotter_qrte.evolve(evolution_problem)

        np.testing.assert_array_almost_equal(
            Statevector.from_instruction(evolution_result.evolved_state).data, expected_state.data
        )

    @data(
        (QuantumCircuit(1), Statevector([0.23071786 - 0.69436148j, 0.4646314 - 0.49874749j])),
        (
            QuantumCircuit(1).compose(ZGate(), [0]),
            Statevector([0.23071786 - 0.69436148j, 0.4646314 - 0.49874749j]),
        ),
    )
    @unpack
    def test_trotter_qrte_qdrift(self, initial_state, expected_state):
        """Test for TrotterQRTE with QDrift."""
        operator = PauliSumOp(SparsePauliOp([Pauli("X"), Pauli("Z")]))
        time = 1
        evolution_problem = TimeEvolutionProblem(operator, time, initial_state)

        algorithm_globals.random_seed = 0
        trotter_qrte = TrotterQRTE(product_formula=QDrift())
        evolution_result = trotter_qrte.evolve(evolution_problem)

        np.testing.assert_array_almost_equal(
            Statevector.from_instruction(evolution_result.evolved_state).data, expected_state.data
        )

    @data((Parameter("t"), {}), (None, {Parameter("x"): 2}), (None, None))
    @unpack
    def test_trotter_qrte_trotter_param_errors(self, t_param, param_value_dict):
        """Test TrotterQRTE with raising errors for parameters."""
        operator = Parameter("t") * PauliSumOp(SparsePauliOp([Pauli("X")])) + PauliSumOp(
            SparsePauliOp([Pauli("Z")])
        )
        initial_state = QuantumCircuit(1)
        self._run_error_test(initial_state, operator, None, None, t_param, param_value_dict)

    @data(([Pauli("X"), Pauli("Y")], None))
    @unpack
    def test_trotter_qrte_trotter_aux_ops_errors(self, aux_ops, estimator):
        """Test TrotterQRTE with raising errors."""
        operator = PauliSumOp(SparsePauliOp([Pauli("X")])) + PauliSumOp(SparsePauliOp([Pauli("Z")]))
        initial_state = QuantumCircuit(1)
        self._run_error_test(initial_state, operator, aux_ops, estimator, None, None)

    @data(
        (SparsePauliOp([Pauli("X"), Pauli("Z")]), QuantumCircuit(1)),
        (X, QuantumCircuit(1)),
        (MatrixOp([[1, 1], [0, 1]]), QuantumCircuit(1)),
        (PauliSumOp(SparsePauliOp([Pauli("X")])) + PauliSumOp(SparsePauliOp([Pauli("Z")])), None),
    )
    @unpack
    def test_trotter_qrte_trotter_hamiltonian_errors(self, operator, initial_state):
        """Test TrotterQRTE with raising errors for evolution problem content."""
        self._run_error_test(initial_state, operator, None, None, None, None)

    @staticmethod
    def _run_error_test(initial_state, operator, aux_ops, estimator, t_param, param_value_dict):
        time = 1
        algorithm_globals.random_seed = 0
        trotter_qrte = TrotterQRTE(estimator=estimator)
        with assert_raises(ValueError):
            evolution_problem = TimeEvolutionProblem(
                operator,
                time,
                initial_state,
                aux_ops,
                t_param=t_param,
                param_value_map=param_value_dict,
            )
            _ = trotter_qrte.evolve(evolution_problem)


if __name__ == "__main__":
    unittest.main()
