# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from qiskit.algorithms.gibbs_state_preparation.gibbs_state import GibbsState
from qiskit.algorithms.gibbs_state_preparation.gibbs_state_builder import GibbsStateBuilder
from qiskit.opflow import OperatorBase


class QiteGibbsStateBuilder(GibbsStateBuilder):
    def __init__(self, qite_algorithm):
        self._qite_algorithm = qite_algorithm

    def build(self, problem_hamiltonian: OperatorBase, temperature: float) -> GibbsState:
        time = 1 / (self.BOLTZMANN_CONSTANT * temperature)
        gibbs_state_function = self._qite_algorithm.evolve(
            hamiltonian=problem_hamiltonian, time=time
        )
        return GibbsState(
            gibbs_state_function=gibbs_state_function,
            hamiltonian=problem_hamiltonian,
            temperature=temperature,
        )
