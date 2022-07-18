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
from math import trunc
import unittest
import numpy as np
from qiskit.algorithms.evolvers.classical_real_evolver import ClassicalRealEvolver
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import data, ddt, unpack
from numpy.testing import assert_raises

from qiskit.quantum_info import Statevector
from qiskit.algorithms.evolvers.evolution_problem import EvolutionProblem
from qiskit.circuit import Parameter
from qiskit.opflow import Y, Z, One, X, Zero, VectorStateFn, StateFn, SummedOp


@ddt
class TestEvolutionProblem(QiskitAlgorithmsTestCase):
    """Test Classical Real Evolver."""

    # def test_dummy(self):
    #     hamiltonian = Y
    #     time = 2.5
    #     initial_state = One

    #     evo_problem = EvolutionProblem(hamiltonian, time, initial_state)

    @data(
        [One, np.pi / 2, X, Zero, -1.0j],
    )
    @unpack
    def test_step(self, initial_state, time, hamiltonian, expected_state, phase):
        evolution_problem = EvolutionProblem(
            hamiltonian, time, initial_state, truncation_threshold=1e-3
        )
        classic_evolver = ClassicalRealEvolver()
        result = classic_evolver.evolve(evolution_problem)
        np.testing.assert_almost_equal(
            result.evolved_state.to_matrix(), phase * expected_state.to_matrix(), decimal=1
        )


if __name__ == "__main__":
    unittest.main()
