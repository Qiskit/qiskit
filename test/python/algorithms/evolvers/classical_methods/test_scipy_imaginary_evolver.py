# This code is part of Qiskit.
#
# (C) Copyright IBM 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Classical Real Evolver."""
import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import data, ddt, unpack
import numpy as np
import scipy.sparse as sp
from qiskit.algorithms.evolvers.evolution_problem import EvolutionProblem
from qiskit.opflow import X, Zero, Plus, Minus, PauliSumOp
from qiskit.opflow import StateFn, OperatorBase
from qiskit import QuantumCircuit
from qiskit.algorithms.evolvers.classical_methods import ScipyImaginaryEvolver


@ddt
class TestClassicalRealEvolver(QiskitAlgorithmsTestCase):
    """Test Classical Real Evolver."""

    def create_hamiltonian_lattice(self, num_sites: int) -> PauliSumOp:
        """Creates an ising hamiltonian on a lattice."""
        j_const = 0.1
        g_const = -1.0

        zz_op = ["I" * i + "ZZ" + "I" * (num_sites - i - 2) for i in range(num_sites - 1)]
        x_op = ["I" * i + "X" + "I" * (num_sites - i - 1) for i in range(num_sites)]
        return PauliSumOp.from_list(
            list(zip(zz_op, len(zz_op) * [j_const])) + list(zip(x_op, len(x_op) * [g_const]))
        )

    @data(
        [Zero, 100, X, Minus, "first"],
        [Zero, 100, -X, Plus, "second"],
        [
            Plus ^ Plus ^ Plus ^ Plus ^ Plus,
            100,
            "lattice",
            Plus ^ Plus ^ Plus ^ Plus ^ Plus,
            "second",
        ],
    )
    @unpack
    def test_evolve(
        self,
        initial_state: StateFn,
        tau: float,
        hamiltonian: OperatorBase,
        expected_state: StateFn,
        order: str,
    ):
        """Initializes a classical imaginary evolver and evolves a state to find the ground state.
        It compares the solution with the first eigenstate of the hamiltonian.
        """

        if hamiltonian == "lattice":
            hamiltonian = self.create_hamiltonian_lattice(5)
            expected_state_matrix = np.absolute(sp.linalg.eigsh(hamiltonian.to_spmatrix())[1][:, 0])
        else:
            expected_state_matrix = expected_state.to_matrix()

        evolution_problem = EvolutionProblem(hamiltonian, tau, initial_state)
        classic_evolver = ScipyImaginaryEvolver(
            timesteps=100, threshold=None, order=order
        )
        result = classic_evolver.evolve(evolution_problem)

        with self.subTest("Amplitudes"):
            np.testing.assert_allclose(
                np.absolute(result.evolved_state.to_matrix()),
                np.absolute(expected_state_matrix),
                atol=1e-5,
                rtol=0,
            )

        with self.subTest("Phases"):
            np.testing.assert_allclose(
                np.angle(result.evolved_state.to_matrix()),
                np.angle(expected_state_matrix),
                atol=1e-10,
                rtol=0,
            )

    def test_quantum_circuit_initial_state(self):
        """Tests if the system can be evolved with a quantum circuit as an initial state."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, range(1, 3))

        evolution_problem = EvolutionProblem(hamiltonian=X ^ X ^ X, time=1.0, initial_state=qc)
        classic_evolver = ScipyImaginaryEvolver(timesteps=5, threshold=None)
        classic_evolver.evolve(evolution_problem)

    def test_error_time_dependency(self):
        """Tests if an error is raised for time dependent hamiltonian."""
        evolution_problem = EvolutionProblem(
            hamiltonian=X ^ X ^ X, time=1.0, initial_state=Zero,t_param = 0
        )
        classic_evolver = ScipyImaginaryEvolver(timesteps=5, threshold=None)
        with self.assertRaises(ValueError):
            classic_evolver.evolve(evolution_problem)


if __name__ == "__main__":
    unittest.main()
