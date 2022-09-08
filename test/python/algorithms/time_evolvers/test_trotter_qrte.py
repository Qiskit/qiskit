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

from qiskit.algorithms.time_evolvers.evolution_problem import EvolutionProblem
from qiskit.algorithms.time_evolvers.trotterization.trotter_qrte import TrotterQRTE
from qiskit.primitives import Estimator
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate
from qiskit.quantum_info import Statevector, Pauli
from qiskit.utils import algorithm_globals
from qiskit.circuit import Parameter
from qiskit.opflow import (
    X,
    Z,
    Zero,
    VectorStateFn,
    StateFn,
    I,
    Y,
)
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
            VectorStateFn(
                Statevector([0.29192658 - 0.45464871j, 0.70807342 - 0.45464871j], dims=(2,))
            ),
        ),
        (
            SuzukiTrotter(),
            VectorStateFn(Statevector([0.29192658 - 0.84147098j, 0.0 - 0.45464871j], dims=(2,))),
        ),
    )
    @unpack
    def test_trotter_qrte_trotter_single_qubit(self, product_formula, expected_state):
        """Test for default TrotterQRTE on a single qubit."""
        operator = X + Z
        initial_state = StateFn([1, 0])
        time = 1
        evolution_problem = EvolutionProblem(operator, time, initial_state)

        trotter_qrte = TrotterQRTE(product_formula=product_formula)
        evolution_result_state_circuit = trotter_qrte.evolve(evolution_problem).evolved_state

        np.testing.assert_equal(evolution_result_state_circuit.eval(), expected_state)

    def test_trotter_qrte_trotter_single_qubit_aux_ops(self):
        """Test for default TrotterQRTE on a single qubit with auxiliary operators."""
        operator = X + Z
        # LieTrotter with 1 rep
        aux_ops = [Pauli("X"), Pauli("Y")]

        initial_state = Zero
        time = 3
        evolution_problem = EvolutionProblem(operator, time, initial_state, aux_ops)
        estimator = Estimator()

        expected_evolved_state = VectorStateFn(
            Statevector([0.98008514 + 0.13970775j, 0.01991486 + 0.13970775j], dims=(2,))
        )
        expected_aux_ops_evaluated = [(0.078073, 0.0), (0.268286, 0.0)]

        algorithm_globals.random_seed = 0
        trotter_qrte = TrotterQRTE(estimator=estimator)
        evolution_result = trotter_qrte.evolve(evolution_problem)

        np.testing.assert_equal(evolution_result.evolved_state.eval(), expected_evolved_state)

        np.testing.assert_array_almost_equal(
            evolution_result.aux_ops_evaluated, expected_aux_ops_evaluated
        )

    @data(
        (
            (X ^ Y) + (Y ^ X),
            VectorStateFn(
                Statevector(
                    [-0.41614684 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.90929743 + 0.0j], dims=(2, 2)
                )
            ),
        ),
        (
            (Z ^ Z) + (Z ^ I) + (I ^ Z),
            VectorStateFn(
                Statevector(
                    [-0.9899925 - 0.14112001j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j], dims=(2, 2)
                )
            ),
        ),
        (
            Pauli("YY"),
            VectorStateFn(
                Statevector(
                    [0.54030231 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.84147098j], dims=(2, 2)
                )
            ),
        ),
    )
    @unpack
    def test_trotter_qrte_trotter_two_qubits(self, operator, expected_state):
        """Test for TrotterQRTE on two qubits with various types of a Hamiltonian."""
        # LieTrotter with 1 rep
        initial_state = QuantumCircuit(2)
        evolution_problem = EvolutionProblem(operator, 1, initial_state)

        trotter_qrte = TrotterQRTE()
        evolution_result = trotter_qrte.evolve(evolution_problem)
        np.testing.assert_equal(evolution_result.evolved_state.eval(), expected_state)

    @data(
        (
            Zero,
            VectorStateFn(
                Statevector([0.23071786 - 0.69436148j, 0.4646314 - 0.49874749j], dims=(2,))
            ),
        ),
        (
            QuantumCircuit(1).compose(ZGate(), [0]),
            VectorStateFn(
                Statevector([0.23071786 - 0.69436148j, 0.4646314 - 0.49874749j], dims=(2,))
            ),
        ),
    )
    @unpack
    def test_trotter_qrte_qdrift(self, initial_state, expected_state):
        """Test for TrotterQRTE with QDrift."""
        operator = X + Z
        time = 1
        evolution_problem = EvolutionProblem(operator, time, initial_state)

        algorithm_globals.random_seed = 0
        trotter_qrte = TrotterQRTE(product_formula=QDrift())
        evolution_result = trotter_qrte.evolve(evolution_problem)
        np.testing.assert_equal(evolution_result.evolved_state.eval(), expected_state)

    @data((Parameter("t"), {}), (None, {Parameter("x"): 2}), (None, None))
    @unpack
    def test_trotter_qrte_trotter_errors(self, t_param, param_value_dict):
        """Test TrotterQRTE with raising errors."""
        operator = X * Parameter("t") + Z
        initial_state = Zero
        time = 1
        algorithm_globals.random_seed = 0
        trotter_qrte = TrotterQRTE()
        with assert_raises(ValueError):
            evolution_problem = EvolutionProblem(
                operator,
                time,
                initial_state,
                t_param=t_param,
                param_value_dict=param_value_dict,
            )
            _ = trotter_qrte.evolve(evolution_problem)


if __name__ == "__main__":
    unittest.main()
