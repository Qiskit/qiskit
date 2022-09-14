# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2022.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Class for holding time evolution result."""
from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.algorithms.list_or_dict import ListOrDict
from ..algorithm_result import AlgorithmResult


class TimeEvolutionResult(AlgorithmResult):
    """
    Class for holding time evolution result.

    Attributes:
        evolved_state (QuantumCircuit): An evolved quantum state.
        aux_ops_evaluated (ListOrDict[tuple[complex, complex]] | None): Optional list of
            observables for which expected values on an evolved state are calculated. These values
            are in fact tuples formatted as (mean, standard deviation).
    """

    def __init__(
        self,
        evolved_state: QuantumCircuit,
        aux_ops_evaluated: ListOrDict[tuple[complex, complex]] | None = None,
    ):
        """
        Args:
            evolved_state: An evolved quantum state.
            aux_ops_evaluated: Optional list of observables for which expected values on an evolved
                state are calculated. These values are in fact tuples formatted as (mean, standard
                deviation).
        """

        self.evolved_state = evolved_state
        self.aux_ops_evaluated = aux_ops_evaluated
