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
import numpy as np
from typing import List
from qiskit.algorithms.evolvers import NumericalIntegrationRealEvolver
from test.python.algorithms import QiskitAlgorithmsTestCase
from qiskit.opflow import StateFn, OperatorBase
from ddt import data, ddt, unpack
from qiskit.algorithms.evolvers.evolution_problem import EvolutionProblem
from qiskit.opflow import Y, Z, I, One, X, Zero, Plus, DictStateFn
import time


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
        self, initial_state: StateFn, t: float, hamiltonian: OperatorBase, expected_state: StateFn
    ):
        """Initializes a classical real evolver and evolves a state."""
        evolution_problem = EvolutionProblem(hamiltonian, t, initial_state)
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
    [Zero, 1.0, X, {"Energy":X, "Polarity":Z}],

    )
    @unpack
    def test_observables(
        self, initial_state: StateFn, t: float, hamiltonian: OperatorBase, observalbes: List[OperatorBase]
    ):
        evolution_problem = EvolutionProblem(hamiltonian, t, initial_state,aux_operators=observalbes)
        classic_evolver = NumericalIntegrationRealEvolver(timesteps=30, threshold=None)
        result = classic_evolver.evolve(evolution_problem)
        print(result.aux_ops_evaluated)

if __name__ == "__main__":
    unittest.main()
