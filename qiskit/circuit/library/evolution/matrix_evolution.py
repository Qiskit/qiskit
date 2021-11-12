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
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.synthesis import MatrixExponential
from qiskit.quantum_info import Operator


class MatrixEvolutionGate(Gate):
    r"""Time-evolution of an operator given as hermitian matrix.

    For an operator :math:`H` and a (real) evolution time :math:`t`this gate implements

    .. math::

        U(t) = e^{-itH}.

    .. note::

        Specifying a :math:`n`-qubit operation using a :math:`2^n \times 2^n` matrix is
        inefficient and should not be used for large numbers of qubits.

    """

    def __init__(
        self,
        operator,
        time: Union[float, ParameterExpression, List[Union[float, ParameterExpression]]] = 1.0,
        label: Optional[str] = None,
    ) -> None:
        """
        Args:
            operator (Operator | np.ndarray | list):
                The operator to evolve. Can also be provided as list of operators, in which
                case all of them will be evolved in a product.
            time: The evolution time. Can also be a list if the operators are provided as list.
            label: A label for the gate to display in visualizations.
            synthesis: A synthesis strategy. If None, the default synthesis is the Lie-Trotter
                product formula with a single repetition.
        """
        if not isinstance(operator, list):
            operator = [operator]

        if not isinstance(time, list):
            time = [time] * len(operator)

        operator = [_to_quantum_info_operator(op) for op in operator]
        name = f"exp(-i {[' + '.join(op.paulis.to_labels()) for op in operator]})"
        num_qubits = operator[0].num_qubits

        super().__init__(name=name, num_qubits=num_qubits, params=time, label=label)

        self.time = time
        self.operator = operator
        self.synthesis = MatrixExponential()

    def _define(self):
        """Unroll, where the default synthesis is matrix based."""
        self.definition = self.synthesis.synthesize(self)

    def inverse(self) -> "PauliEvolutionGate":
        inv_time = [-time for time in self.time]
        return MatrixEvolutionGate(operator=self.operator, time=inv_time)


def _to_quantum_info_operator(operator):
    """Cast the operator to a qiskit.quantum_info.Operator."""

    if not isinstance(operator, Operator):
        operator = Operator(operator)

    return operator
