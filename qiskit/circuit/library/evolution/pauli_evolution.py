# This code is part of Qiskit.
from qiskit.quantum_info import Pauli
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.gate import Gate
from typing import Union, Optional
print(evo_gate.definition)
#
print(evo_gate.definition)
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A gate to implement time-evolution of a single Pauli string."""


class PauliEvolutionGate(Gate):
    """Time-evolution of a single Pauli string."""

    def __init__(self,
                 pauli: Pauli,
                 time: Union[float, ParameterExpression] = 1.0,
                 label: Optional[str] = None,
                 ) -> None:
        """
        Args:
            operator: The Pauli to evolve.
        """
        super().__init__(
            name='PauliEvolution',
            num_qubits=pauli.num_qubits,
            params=[],
            label=label
        )

        self.time = time
        self.pauli = pauli

    def _define(self):
        """Unroll with a default implementation."""
        if self.definition is not None:
            return

        # TODO move logic here
        from qiskit.opflow import PauliTrotterEvolution, PauliOp
        pop = PauliOp(self.pauli)
        definition = PauliTrotterEvolution().convert(pop).to_circuit_op().primitive
        self.definition = definition

    def inverse(self) -> "PauliEvolutionGate":
        # TODO label
        return PauliEvolutionGate(pauli=self.pauli, time=-self.time)
