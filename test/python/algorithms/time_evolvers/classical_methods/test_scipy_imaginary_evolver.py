# This code is part of Qiskit.
#
# (C) Copyright IBM 2022, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Test Classical Imaginary Evolver."""
import unittest
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import data, ddt, unpack
import numpy as np
from qiskit.algorithms.time_evolvers.time_evolution_problem import TimeEvolutionProblem

from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info import SparsePauliOp

from qiskit import QuantumCircuit
from qiskit.algorithms import SciPyImaginaryEvolver

from qiskit.opflow import PauliSumOp


@ddt
class TestSciPyImaginaryEvolver(QiskitAlgorithmsTestCase):
    """Test SciPy Imaginary Evolver."""

    def create_hamiltonian_lattice(self, num_sites: int) -> SparsePauliOp:
        """Creates an Ising Hamiltonian on a lattice."""
        j_const = 0.1
        g_const = -1.0

        zz_op = ["I" * i + "ZZ" + "I" * (num_sites - i - 2) for i in range(num_sites - 1)]
        x_op = ["I" * i + "X" + "I" * (num_sites - i - 1) for i in range(num_sites)]
        return SparsePauliOp(zz_op) * j_const + SparsePauliOp(x_op) * g_const

    @data(
        (Statevector.from_label("0"), 100, SparsePauliOp("X"), Statevector.from_label("-")),
        (Statevector.from_label("0"), 100, SparsePauliOp("-X"), Statevector.from_label("+")),
    )
    @unpack
    def test_evolve(
        self,
        initial_state: Statevector,
        tau: float,
        hamiltonian: SparsePauliOp,
        expected_state: Statevector,
    ):
        """Initializes a classical imaginary evolver and evolves a state to find the ground state.
        It compares the solution with the first eigenstate of the hamiltonian.
        """
        expected_state_matrix = expected_state.data

        evolution_problem = TimeEvolutionProblem(hamiltonian, tau, initial_state)
        classic_evolver = SciPyImaginaryEvolver(num_timesteps=300)
        result = classic_evolver.evolve(evolution_problem)

        with self.subTest("Amplitudes"):
            np.testing.assert_allclose(
                np.absolute(result.evolved_state.data),
                np.absolute(expected_state_matrix),
                atol=1e-10,
                rtol=0,
            )

        with self.subTest("Phases"):
            np.testing.assert_allclose(
                np.angle(result.evolved_state.data),
                np.angle(expected_state_matrix),
                atol=1e-10,
                rtol=0,
            )

    @data(
        (
            Statevector.from_label("0" * 5),
            SparsePauliOp.from_sparse_list([("X", [i], 1) for i in range(5)], num_qubits=5),
            5,
        ),
        (Statevector.from_label("0"), SparsePauliOp("X"), 1),
    )
    @unpack
    def test_observables(
        self, initial_state: Statevector, hamiltonian: SparsePauliOp, nqubits: int
    ):
        """Tests if the observables are properly evaluated at each timestep."""

        time_ev = 5.0
        observables = {"Energy": hamiltonian, "Z": SparsePauliOp("Z" * nqubits)}
        evolution_problem = TimeEvolutionProblem(
            hamiltonian, time_ev, initial_state, aux_operators=observables
        )

        classic_evolver = SciPyImaginaryEvolver(num_timesteps=300)
        result = classic_evolver.evolve(evolution_problem)

        z_mean, z_std = result.observables["Z"]

        time_vector = result.times
        expected_z = 1 / (np.cosh(time_vector) ** 2 + np.sinh(time_vector) ** 2)
        expected_z_std = np.zeros_like(expected_z)

        np.testing.assert_allclose(z_mean, expected_z**nqubits, atol=1e-10, rtol=0)
        np.testing.assert_allclose(z_std, expected_z_std, atol=1e-10, rtol=0)

    def test_quantum_circuit_initial_state(self):
        """Tests if the system can be evolved with a quantum circuit as an initial state."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, range(1, 3))

        evolution_problem = TimeEvolutionProblem(
            hamiltonian=SparsePauliOp("X" * 3), time=1.0, initial_state=qc
        )
        classic_evolver = SciPyImaginaryEvolver(num_timesteps=5)
        result = classic_evolver.evolve(evolution_problem)
        self.assertEqual(result.evolved_state, Statevector(qc))

    def test_paulisumop_hamiltonian(self):
        """Tests if the hamiltonian can be a PauliSumOp"""
        with self.assertWarns(DeprecationWarning):
            hamiltonian = PauliSumOp.from_list(
                [
                    ("XI", 1),
                    ("IX", 1),
                ]
            )
            observable = PauliSumOp.from_list([("ZZ", 1)])
        evolution_problem = TimeEvolutionProblem(
            hamiltonian=hamiltonian,
            time=1.0,
            initial_state=Statevector.from_label("00"),
            aux_operators={"ZZ": observable},
        )
        classic_evolver = SciPyImaginaryEvolver(num_timesteps=5)
        result = classic_evolver.evolve(evolution_problem)
        expected = 1 / (np.cosh(1.0) ** 2 + np.sinh(1.0) ** 2)
        np.testing.assert_almost_equal(result.aux_ops_evaluated["ZZ"][0], expected**2)

    def test_error_time_dependency(self):
        """Tests if an error is raised for a time dependent Hamiltonian."""
        evolution_problem = TimeEvolutionProblem(
            hamiltonian=SparsePauliOp("X" * 3),
            time=1.0,
            initial_state=Statevector.from_label("0" * 3),
            t_param=0,
        )
        classic_evolver = SciPyImaginaryEvolver(num_timesteps=5)
        with self.assertRaises(ValueError):
            classic_evolver.evolve(evolution_problem)

    def test_no_time_steps(self):
        """Tests if the evolver handles some edge cases related to the number of timesteps."""
        evolution_problem = TimeEvolutionProblem(
            hamiltonian=SparsePauliOp("X"),
            time=1.0,
            initial_state=Statevector.from_label("0"),
            aux_operators={"Energy": SparsePauliOp("X")},
        )

        with self.subTest("0 timesteps"):
            with self.assertRaises(ValueError):
                classic_evolver = SciPyImaginaryEvolver(num_timesteps=0)
                classic_evolver.evolve(evolution_problem)

        with self.subTest("1 timestep"):
            classic_evolver = SciPyImaginaryEvolver(num_timesteps=1)
            result = classic_evolver.evolve(evolution_problem)
            np.testing.assert_equal(result.times, np.array([0.0, 1.0]))

        with self.subTest("Negative timesteps"):
            with self.assertRaises(ValueError):
                classic_evolver = SciPyImaginaryEvolver(num_timesteps=-5)
                classic_evolver.evolve(evolution_problem)


if __name__ == "__main__":
    unittest.main()
