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

from typing import Union, Optional

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.parameterexpression import ParameterExpression

from qiskit.quantum_info import Operator, Pauli, SparsePauliOp

from .evolution_synthesis import EvolutionSynthesis


class EvolutionGate(Gate):
    """Time-evolution of an operator."""

    def __init__(
        self,
        operator,
        time: Union[float, ParameterExpression] = 1.0,
        label: Optional[str] = None,
        synthesis: Optional[EvolutionSynthesis] = None,
    ) -> None:
        """
        Args:
            operator (Union[SparsePauliOp, PauliSumOp, List[SparsePauliOp], List[PauliSumOp]]):
                The operator to evolve. Can be provided as list of non-commuting operators
                where the elements are sums of commuting operators.
            time: The evolution time.
            label: A label for the gate to display in visualizations.
            synthesis: A synthesis strategy. If None, the default synthesis is exponentially
                expensive matrix calculation, exponentiation and synthesis.
        """
        if isinstance(operator, list):
            operator = [_to_sparse_pauli_op(op) for op in operator]
        else:
            operator = _to_sparse_pauli_op(operator)

        num_qubits = operator[0].num_qubits if isinstance(operator, list) else operator.num_qubits
        super().__init__(name="EvolutionGate", num_qubits=num_qubits, params=[time], label=label)

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
        return EvolutionGate(operator=self.operator, time=-self.time)


def _pauliop_to_sparsepauli(operator):
    return SparsePauliOp(operator.primitive, operator.coeff)


def _to_sparse_pauli_op(operator):
    """Cast the operator to a SparsePauliOp.

    For Opflow objects, return a global coefficient that must be multiplied to the evolution time.
    Since this coefficient might contain unbound parameters it cannot be absorbed into the
    coefficients of the SparsePauliOp.
    """
    # pylint: disable=cyclic-import
    from qiskit.opflow import PauliSumOp, PauliOp

    if isinstance(operator, PauliSumOp):
        sparse_pauli = operator.primitive
        sparse_pauli._coeffs *= operator.coeff
        return sparse_pauli
    if isinstance(operator, PauliOp):
        sparse_pauli = SparsePauliOp(operator.primitive)
        sparse_pauli._coeffs *= operator.coeff
        return sparse_pauli
    if isinstance(operator, Pauli):
        return SparsePauliOp(operator)
    if isinstance(operator, SparsePauliOp):
        return operator

    raise ValueError(f"Unsupported operator type for evolution: {type(operator)}.")
