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

"""Evolution synthesis."""

from typing import List, Union
from abc import ABC, abstractmethod
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


class EvolutionSynthesis(ABC):
    """Interface for evolution synthesis algorithms.

    TODO do we need this or do we prefer plain functions?
    """

    @abstractmethod
    def synthesize(
        self, operators: Union[SparsePauliOp, List[SparsePauliOp]], time: float
    ) -> QuantumCircuit:
        """Synthesize the evolution of a (list of) operator(s) to a circuit.

        If a single operator, the summands are assumed not to commute.
        If a list, the list elements are *not* assumed to commute pairwisely, however summands in
        a single operator are assumed to commute.

        Args:
            operators: List of operators to evolve.
            time: Evolution time.

        Returns:
            A circuit implementing the evolution.
        """
        raise NotImplementedError
