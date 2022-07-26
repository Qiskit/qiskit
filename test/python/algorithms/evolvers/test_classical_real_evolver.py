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
from typing import List
from ddt import data, ddt, unpack
import numpy as np
from qiskit.opflow import StateFn, OperatorBase
from qiskit import QuantumCircuit
from qiskit.algorithms.evolvers import NumericalIntegrationRealEvolver
from qiskit.algorithms.evolvers.evolution_problem import EvolutionProblem
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
    )
    @unpack
    def test_evolve(
        self, initial_state: StateFn, time_ev: float, hamiltonian: OperatorBase, expected_state: StateFn
    ):
        """Initializes a classical real evolver and evolves a state."""
        evolution_problem = EvolutionProblem(hamiltonian, time_ev, initial_state)
        classic_evolver = NumericalIntegrationRealEvolver(timesteps=30, threshold=None)
        result = classic_evolver.evolve(evolution_problem)

        with self.subTest("Amplitudes"):
            np.testing.assert_allclose(
                np.absolute(result.evolved_state.to_matrix()),
                np.absolute(expected_state.to_matrix()),
                atol=1e-3,
                rtol=0,
            )

        with self.subTest("Phases"):
            np.testing.assert_allclose(
                np.angle(result.evolved_state.to_matrix()),
                np.angle(expected_state.to_matrix()),
                atol=1e-20,
                rtol=0,
            )

    @data(
        [Zero, 1.0, X, {"Energy": X, "Polarity": Z}],
    )
    @unpack
    def test_observables(
        self,
        initial_state: StateFn,
        time_ev: float,
        hamiltonian: OperatorBase,
        observalbes: List[OperatorBase],
    ):
        """Tests if the observables are properly evaluated at each timestep."""
        evolution_problem = EvolutionProblem(
            hamiltonian, time_ev, initial_state, aux_operators=observalbes
        )
        classic_evolver = NumericalIntegrationRealEvolver(timesteps=10, threshold=None)
        result = classic_evolver.evolve(evolution_problem)
        print(result.aux_ops_evaluated)

    def test_quantum_circuit_initial_state(self):
        """Tests if the system can be evolved with a quantum circuit as an initial state."""
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, range(1, 3))

        evolution_problem = EvolutionProblem(hamiltonian=X ^ X ^ X, time=1.0, initial_state=qc)
        classic_evolver = NumericalIntegrationRealEvolver(timesteps=5, threshold=None)
        classic_evolver.evolve(evolution_problem)


if __name__ == "__main__":
    unittest.main()
