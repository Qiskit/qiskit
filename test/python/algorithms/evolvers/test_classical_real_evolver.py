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
from qiskit.algorithms.evolvers.classical_real_evolver import ClassicalRealEvolver
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import data, ddt, unpack
from qiskit.algorithms.evolvers.evolution_problem import EvolutionProblem
from qiskit.opflow import Y, Z, I, One, X, Zero
import time


@ddt
class TestClassicalRealEvolver(QiskitAlgorithmsTestCase):
    """Test Classical Real Evolver."""

    @data(
        [One, np.pi / 2, X, Zero, -1.0j],
        [One ^ Zero, np.pi / 2, ((X ^ X) + (Y ^ Y)) / 2, Zero ^ One, -1.0j],
    )
    @unpack
    def test_evolve(self, initial_state, time, hamiltonian, expected_state, phase):
        evolution_problem = EvolutionProblem(hamiltonian, time, initial_state)
        classic_evolver = ClassicalRealEvolver(timesteps=None, threshold = 1e-5)
        # recomended_steps = classic_evolver.minimal_number_steps(1.0,time, 1e-4)
        # print(recomended_steps)
        # classic_evolver.timesteps = recomended_steps
        result = classic_evolver.evolve(evolution_problem)

        with self.subTest("Amplitudes"):
            np.testing.assert_allclose(
                np.absolute(result.evolved_state.to_matrix()),
                np.absolute(phase * expected_state.to_matrix()),
                atol=1e-7,
                rtol=0,
            )
        with self.subTest("Phases"):
            np.testing.assert_allclose(
                np.angle(result.evolved_state.to_matrix()),
                np.angle(phase * expected_state.to_matrix()),
                atol=1e-7,
                rtol=0,
            )


if __name__ == "__main__":
    unittest.main()
