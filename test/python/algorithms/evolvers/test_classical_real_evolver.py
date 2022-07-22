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
import scipy.sparse as sp
from qiskit.algorithms.evolvers.classical_real_evolver import ClassicalRealEvolver
from test.python.algorithms import QiskitAlgorithmsTestCase
from ddt import data, ddt, unpack
from numpy.testing import assert_raises

from qiskit.quantum_info import Statevector
from qiskit.algorithms.evolvers.evolution_problem import EvolutionProblem
from qiskit.circuit import Parameter
from qiskit.opflow import Y, Z, I, One, X, Zero, VectorStateFn, StateFn, SummedOp, PauliSumOp
import time


@ddt
class TestClassicalRealEvolver(QiskitAlgorithmsTestCase):
    """Test Classical Real Evolver."""

    @data(
        [One, np.pi / 2, X, Zero, -1.0j],
        [One ^ Zero, np.pi/2 , ((X ^ X) + (Y^Y)) / 2, Zero ^ One, -1.0j],
    )

    @unpack
    def test_evolve(self, initial_state, time, hamiltonian, expected_state, phase):
        evolution_problem = EvolutionProblem(
            hamiltonian, time, initial_state, truncation_threshold=1e-8
        )
        classic_evolver = ClassicalRealEvolver()
        result = classic_evolver.evolve(evolution_problem)
        np.testing.assert_almost_equal(
            result.evolved_state.to_matrix(), phase * expected_state.to_matrix(), decimal=5
        )

    def test_ising_quick(self):
        num_sites = 25
        J = 0.1
        g = -1.0
        t = 1.0

        zz = ["I" * i + "ZZ" + "I" * (num_sites - i - 2) for i in range(num_sites - 1)]
        x = ["I" * i + "X" + "I" * (num_sites - i - 1) for i in range(num_sites)]
        hamiltonian = PauliSumOp.from_list(list(zip(zz, len(zz) * [J])) + list(zip(x, len(x) * [g])))
        initial_state = np.zeros(2 ** num_sites, dtype=np.complex128)
        initial_state[0] = 1.0
        initial_state = VectorStateFn(initial_state)

        evolution_problem = EvolutionProblem(hamiltonian,t,initial_state,truncation_threshold=1e-5)
        classic_evolver = ClassicalRealEvolver()

        start_time_s = time.time()
        state = evolution_problem.initial_state.to_matrix(massive=True).transpose()
        start_time = time.time()
        print(f"Time to create the state: {start_time - start_time_s:.3}")

        hamiltonian = hamiltonian.primitive.to_matrix(sparse=True)

        end_time = time.time()

        print("Time to convert to matrix: ", end_time - start_time)

        ntimesteps = 10
        timestep = evolution_problem.time / ntimesteps

        start_time = time.time()
        idnty = sp.identity(
            hamiltonian.shape[0], format="csr"
        )  # What would be the best format for this?
        end_time = time.time()
        print("Identity matrix creation time: ", end_time - start_time)

        lhs_operator = idnty + 1j * timestep / 2 * hamiltonian
        rhs_operator = idnty - 1j * timestep / 2 * hamiltonian


        end_time_s = time.time()
        print(f"It takes {end_time_s- start_time_s :.3} seconds to initialize the system.")

        start_time = time.time()
        classic_evolver._step(state, lhs_operator, rhs_operator)
        end_time = time.time()
        print(f"It takes {end_time- start_time :.3} seconds to make a step.")


    def test_ising(self):
        num_sites = 18
        J = 0.1
        g = -1.0
        time = 1.0

        zz = ["I" * i + "ZZ" + "I" * (num_sites - i - 2) for i in range(num_sites - 1)]
        x = ["I" * i + "X" + "I" * (num_sites - i - 1) for i in range(num_sites)]
        hamiltonian = PauliSumOp.from_list(list(zip(zz, len(zz) * [J])) + list(zip(x, len(x) * [g])))
        initial_state = np.zeros(2 ** num_sites, dtype=np.complex128)
        initial_state[0] = 1.0
        initial_state = VectorStateFn(initial_state)

        evolution_problem = EvolutionProblem(hamiltonian,time,initial_state,truncation_threshold=1e-5)
        classic_evolver = ClassicalRealEvolver()
        result = classic_evolver.evolve(evolution_problem)
        print(result.evolved_state)

        # #Verify with exact solution.
        # sp_hamiltonian = hamiltonian.to_spmatrix()
        # print(sp_hamiltonian.shape)
        # evol_operator = sp.linalg.expm(-1j * time * sp_hamiltonian)
        # evol_state = evol_operator.dot( initial_state.to_matrix())
        # np.testing.assert_allclose(result.evolved_state.to_matrix(), evol_state, rtol = 1e-5, atol = 0)


if __name__ == "__main__":
    unittest.main()
