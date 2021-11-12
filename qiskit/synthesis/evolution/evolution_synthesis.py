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

from abc import ABC, abstractmethod
from typing import Union
from qiskit.circuit import ParameterExpression, QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


class EvolutionSynthesis(ABC):
    """Interface for evolution synthesis algorithms."""

    def synthesize(self, evolution):
        """Synthesize an ``qiskit.circuit.library.PauliEvolutionGate``.

        Args:
            evolution (PauliEvolutionGate): The evolution gate to synthesize.

        Returns:
            QuantumCircuit: A circuit implementing the evolution.
        """
        definition = QuantumCircuit(evolution.num_qubits)
        for time, operator in zip(evolution.time, evolution.operator):
            definition.compose(self._evolve_operator(operator, time), inplace=True)

        return definition

    @abstractmethod
    def _evolve_operator(self, operator: SparsePauliOp, time: Union[float, ParameterExpression]):
        """Evolve a single operator for a given time.

        Args:
            operator: The operator to evolve.
            time: The time for which to evolve the operator.

        Returns:
            A circuit implementing `exp(-i time operator)`.
        """
        raise NotImplementedError()
