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
from qiskit.algorithms.time_evolvers.time_evolution_problem import TimeEvolutionProblem
from qiskit.opflow import (
    I,
    X,
    Z,
    Zero,
    Plus,
    Minus,
    PauliSumOp,
    DictStateFn,
    OperatorBase,
    VectorStateFn,
)
from qiskit.quantum_info.states.statevector import Statevector

from qiskit import QuantumCircuit
from qiskit.algorithms import SciPyImaginaryEvolver


@ddt
class TestSciPyImaginaryEvolver(QiskitAlgorithmsTestCase):
    """Test SciPy Imaginary Evolver."""

    def create_hamiltonian_lattice(self, num_sites: int) -> PauliSumOp:
        """Creates an Ising Hamiltonian on a lattice."""
        j_const = 0.1
        g_const = -1.0

        zz_op = ["I" * i + "ZZ" + "I" * (num_sites - i - 2) for i in range(num_sites - 1)]
        x_op = ["I" * i + "X" + "I" * (num_sites - i - 1) for i in range(num_sites)]
        return PauliSumOp.from_list(
            list(zip(zz_op, len(zz_op) * [j_const])) + list(zip(x_op, len(x_op) * [g_const]))
        )

    @data(
        (Zero, 100, X, Minus),
        (Zero, 100, -X, Plus),
    )
    @unpack
    def test_evolve(
        self,
        initial_state: DictStateFn,
        tau: float,
        hamiltonian: OperatorBase,
        expected_state: DictStateFn,
    ):
        """Initializes a classical imaginary evolver and evolves a state to find the ground state.
        It compares the solution with the first eigenstate of the hamiltonian.
        """
        expected_state_matrix = expected_state.to_matrix()

        evolution_problem = TimeEvolutionProblem(hamiltonian, tau, initial_state)
        classic_evolver = SciPyImaginaryEvolver(steps=300)
        result = classic_evolver.evolve(evolution_problem)

        with self.subTest("Amplitudes"):
            np.testing.assert_allclose(
                np.absolute(result.evolved_state.to_matrix()),
                np.absolute(expected_state_matrix),
                atol=1e-10,
                rtol=0,
            )

        with self.subTest("Phases"):
            np.testing.assert_allclose(
                np.angle(result.evolved_state.to_matrix()),
                np.angle(expected_state_matrix),
                atol=1e-10,
                rtol=0,
            )

    @data(
        (
            Zero ^ 5,
            (X ^ (I ^ 4))
            + (I ^ X ^ (I ^ 3))
            + ((I ^ 2) ^ X ^ (I ^ 2))
            + ((I ^ 3) ^ X ^ I)
            + ((I ^ 4) ^ X),
            5,
        ),
        (Zero, X, 1),
    )
    @unpack
    def test_observables(self, initial_state: DictStateFn, hamiltonian: OperatorBase, nqubits: int):
        """Tests if the observables are properly evaluated at each timestep."""

        time_ev = 5.0
        observables = {"Energy": hamiltonian, "Z": Z ^ nqubits}
        evolution_problem = TimeEvolutionProblem(
            hamiltonian, time_ev, initial_state, aux_operators=observables
        )

        classic_evolver = SciPyImaginaryEvolver(steps=300)
        result = classic_evolver.evolve(evolution_problem)

        z_mean, z_std = result.observables["Z"]

        time_vector = result.observables["time"]
        expected_z = 1 / (np.cosh(time_vector) ** 2 + np.sinh(time_vector) ** 2)
        expected_z_std = np.zeros_like(expected_z)

        np.testing.assert_allclose(z_mean, expected_z**nqubits, atol=1e-10, rtol=0)
        np.testing.assert_allclose(z_std, expected_z_std, atol=1e-10, rtol=0)

    def test_quantum_circuit_initial_state(self):
        """Tests if the system can be evolved with a quantum circuit as an initial state."""
        qc = QuantumCircuit(3)

        qc.h(0)
        qc.cx(0, range(1, 3))

        evolution_problem = TimeEvolutionProblem(hamiltonian=X ^ X ^ X, time=1.0, initial_state=qc)
        classic_evolver = SciPyImaginaryEvolver(steps=5)
        result = classic_evolver.evolve(evolution_problem)
        self.assertEqual(result.evolved_state, VectorStateFn(Statevector(qc)))

    def test_error_time_dependency(self):
        """Tests if an error is raised for a time dependent Hamiltonian."""
        evolution_problem = TimeEvolutionProblem(
            hamiltonian=X ^ X ^ X, time=1.0, initial_state=Zero, t_param=0
        )
        classic_evolver = SciPyImaginaryEvolver(steps=5)
        with self.assertRaises(ValueError):
            classic_evolver.evolve(evolution_problem)

    def test_no_time_steps(self):
        """Tests if the evolver handles some edge cases related to the number of timesteps."""
        evolution_problem = TimeEvolutionProblem(hamiltonian=X, time=1.0, initial_state=Zero)

        with self.subTest("0 timesteps"):
            with self.assertRaises(ValueError):
                classic_evolver = SciPyImaginaryEvolver(steps=0)
                classic_evolver.evolve(evolution_problem)

        with self.subTest("1 timestep"):
            classic_evolver = SciPyImaginaryEvolver(steps=1)
            classic_evolver.evolve(evolution_problem)

        with self.subTest("Negative timesteps"):
            with self.assertRaises(ValueError):
                classic_evolver = SciPyImaginaryEvolver(steps=-5)
                classic_evolver.evolve(evolution_problem)


if __name__ == "__main__":
    unittest.main()
