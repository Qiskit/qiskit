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
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.synthesis import EvolutionSynthesis, LieTrotter
from qiskit.quantum_info import Pauli, SparsePauliOp


class EvolutionGate(Gate):
    r"""Time-evolution of an operator.

    For an operator :math:`H` consisting of Pauli terms and (real) evolution time :math:`t`
    this gate implements

    .. math::

        U(t) = e^{-itH}.

    The evolution gates are related to the Pauli rotation gates by a factor of 2. For example
    the time evolution of the Pauli :math:`X` operator is connected to the Pauli :math:`X` rotation
    :math:`R_X` by

    .. math::

        U(t) = e^{-itX} = R_X(2t).

    This gate serves as definition of the evolution and can be synthesized into a circuit using
    different algorithms.
    """

    def __init__(
        self,
        operator,
        time: Union[float, ParameterExpression] = 1.0,
        label: Optional[str] = None,
        synthesis: Optional[EvolutionSynthesis] = None,
    ) -> None:
        """
        Args:
            operator (Union[Pauli, PauliOp, SparsePauliOp, PauliSumOp, List[SparsePauliOp],
                List[PauliSumOp]]): The operator to evolve. Can be provided as list of non-commuting
                operators where the elements are sums of commuting operators.
                For example: ``[XY + YX, ZZ + ZI + IZ, YY]``.
            time: The evolution time.
            label: A label for the gate to display in visualizations.
            synthesis: A synthesis strategy. If None, the default synthesis is the Lie-Trotter
                product formula with a single repetition.
        """
        if isinstance(operator, list):
            operator = [_to_sparse_pauli_op(op) for op in operator]
            name = f"exp(-it {[' + '.join(op.paulis.to_labels()) for op in operator]})"
        else:
            operator = _to_sparse_pauli_op(operator)
            name = f"exp(-it {' + '.join(operator.paulis.to_labels())})"

        if synthesis is None:
            synthesis = LieTrotter()

        num_qubits = operator[0].num_qubits if isinstance(operator, list) else operator.num_qubits
        super().__init__(name=name, num_qubits=num_qubits, params=[time], label=label)

        self.time = time
        self.operator = operator
        self.synthesis = synthesis

    def _define(self):
        """Unroll, where the default synthesis is matrix based."""
        self.definition = self.synthesis.synthesize(self.operator, self.time)

    def inverse(self) -> "EvolutionGate":
        return EvolutionGate(operator=self.operator, time=-self.time, synthesis=self.synthesis)


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
