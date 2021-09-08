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

"""A gate to implement time-evolution of operators."""

from typing import Union, Optional, List

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterExpression

from qiskit.quantum_info import SparsePauliOp, Operator, Pauli

from .evolution_synthesis import EvolutionSynthesis


class EvolutionGate(Gate):
    """Time-evolution of an operator."""

    def __init__(
        self,
        operator: Union[SparsePauliOp, "PauliSumOp", List[SparsePauliOp], List["PauliSumOp"]],
        time: Union[float, ParameterExpression] = 1.0,
        label: Optional[str] = None,
        synthesis: Optional[EvolutionSynthesis] = None,
    ) -> None:
        """
        Args:
            operator: The operator to evolve. Can be provided as list of non-commuting operators
                where the elements are sums of commuting operators.
            time: The evolution time.
            label: A label for the gate to display in visualizations.
            synthesis: A synthesis strategy. If None, the default synthesis is exponentially
                expensive matrix calculation, exponentiation and synthesis.
        """
        operator = _cast_to_sparse_pauli_op(operator)

        num_qubits = operator[0].num_qubits if isinstance(operator, list) else operator.num_qubits
        super().__init__(name="EvolutionGate", num_qubits=num_qubits, params=[], label=label)

        self.time = time
        self.operator = operator
        self.synthesis = synthesis

    def _define(self):
        """Unroll, where the default synthesis is matrix based."""
        if isinstance(self.operator, Pauli):
            from .pauli_evolution import PauliEvolutionGate

            self.definition = PauliEvolutionGate(self.operator, self.time).definition
        elif self.synthesis is None:
            self.definition = self._matrix_synthesis()
        else:
            self.definition = self.synthesis.synthesize(self.operator, self.time)

    def _matrix_synthesis(self):
        if isinstance(self.time, ParameterExpression):
            raise ValueError("Cannot define evolution with unbound time parameter.")

        if isinstance(self.operator, list):
            operator = sum(self.operator)
        else:
            operator = self.operator

        definition = QuantumCircuit(self.num_qubits)
        definition.hamiltonian(Operator(operator).data, self.time, definition.qubits)
        return definition

    def inverse(self) -> "EvolutionGate":
        # TODO label
        return EvolutionGate(operator=self.operator, time=-self.time)


def _cast_to_sparse_pauli_op(operator):
    from qiskit.opflow import PauliSumOp, PauliOp

    if isinstance(operator, (PauliSumOp, PauliOp)):
        # TODO warning: coefficients are discarded?
        operator = operator.primitive
    elif isinstance(operator, list):
        for i, op in enumerate(operator):
            if isinstance(op, (PauliSumOp, PauliOp)):
                # TODO warning: coefficients are discarded?
                operator[i] = op.primitive
    return operator
