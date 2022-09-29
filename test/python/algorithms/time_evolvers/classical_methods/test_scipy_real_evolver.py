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
from qiskit.opflow import StateFn, OperatorBase
from qiskit import QuantumCircuit
from qiskit.algorithms import SciPyRealEvolver
from qiskit.algorithms import TimeEvolutionProblem
from qiskit.opflow import Y, Z, One, X, Zero


@ddt
class TestClassicalRealEvolver(QiskitAlgorithmsTestCase):
    """Test Classical Real Evolver."""

    @data(
        [One, np.pi / 2, X, -1.0j * Zero],
        [One ^ Zero, np.pi / 2, ((X ^ X) + (Y ^ Y)) / 2, -1.0j * Zero ^ One],
        [
            One ^ Zero,
            np.pi / 4,
            ((X ^ X) + (Y ^ Y)) / 2,
            ((One ^ Zero) - 1.0j * (Zero ^ One)) / np.sqrt(2),
        ],
        [Zero ^ 12, np.pi / 2, X ^ 12, -1.0j * (One ^ 12)],
    )
    @unpack
    def test_evolve(
        self,
        initial_state: StateFn,
        time_ev: float,
        hamiltonian: OperatorBase,
        expected_state: StateFn,
    ):
        """Initializes a classical real evolver and evolves a state."""
        evolution_problem = TimeEvolutionProblem(hamiltonian, time_ev, initial_state)
        classic_evolver = SciPyRealEvolver(steps=1)
        result = classic_evolver.evolve(evolution_problem)

        with self.subTest("Amplitudes"):
            np.testing.assert_allclose(
                np.absolute(result.evolved_state.to_matrix()),
                np.absolute(expected_state.to_matrix()),
                atol=1e-10,
                rtol=0,
            )

        with self.subTest("Phases"):
            np.testing.assert_allclose(
                np.angle(result.evolved_state.to_matrix()),
                np.angle(expected_state.to_matrix()),
                atol=1e-10,
                rtol=0,
            )

    def test_observables(self):
        """Tests if the observables are properly evaluated at each timestep."""

        initial_state = Zero
        time_ev = 10.0
        hamiltonian = X
        observables = {"Energy": X, "Z": Z}
        evolution_problem = TimeEvolutionProblem(
            hamiltonian, time_ev, initial_state, aux_operators=observables
        )
        classic_evolver = SciPyRealEvolver(steps=10)
        result = classic_evolver.evolve(evolution_problem)

        z_mean, z_std = result.observables["Z"]

        timesteps = z_mean.shape[0]
        time_vector = np.linspace(0, time_ev, timesteps)
        expected_z = 1 - 2 * (np.sin(time_vector)) ** 2
        expected_z_std = np.zeros_like(expected_z)

        np.testing.assert_allclose(z_mean, expected_z, atol=1e-10, rtol=0)
        np.testing.assert_allclose(z_std, expected_z_std, atol=1e-10, rtol=0)

    def test_quantum_circuit_initial_state(self):
        """Tests if the system can be evolved with a quantum circuit as an initial state."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, range(1, 3))

        evolution_problem = TimeEvolutionProblem(
            hamiltonian=X ^ X ^ X, time=2 * np.pi, initial_state=qc
        )
        classic_evolver = SciPyRealEvolver(steps=500)
        result = classic_evolver.evolve(evolution_problem)
        np.testing.assert_almost_equal(
            result.evolved_state.to_matrix(),
            np.array([1, 0, 0, 0, 0, 0, 0, 1]) / np.sqrt(2),
            decimal=10,
        )

    def test_error_time_dependency(self):
        """Tests if an error is raised for time dependent hamiltonian."""
        evolution_problem = TimeEvolutionProblem(
            hamiltonian=X ^ X ^ X, time=1.0, initial_state=Zero, t_param=0
        )
        classic_evolver = SciPyRealEvolver(steps=5)
        with self.assertRaises(ValueError):
            classic_evolver.evolve(evolution_problem)

    def test_no_time_steps(self):
        """Tests if the evolver handles some edge cases related to the number of timesteps."""
        evolution_problem = TimeEvolutionProblem(
            hamiltonian=X, time=1.0, initial_state=Zero, aux_operators={"Energy": X}
        )

        with self.subTest("0 timesteps"):
            with self.assertRaises(ValueError):
                classic_evolver = SciPyRealEvolver(steps=0)
                classic_evolver.evolve(evolution_problem)

        with self.subTest("1 timestep"):
            classic_evolver = SciPyRealEvolver(steps=1)
            result = classic_evolver.evolve(evolution_problem)
            np.testing.assert_equal(result.observables["time"], np.array([0.0, 1.0]))

        with self.subTest("Negative timesteps"):
            with self.assertRaises(ValueError):
                classic_evolver = SciPyRealEvolver(steps=-5)
                classic_evolver.evolve(evolution_problem)


if __name__ == "__main__":
    unittest.main()
