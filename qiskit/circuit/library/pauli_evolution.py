# This code is part of Qiskit.
#
# (C) Copyright IBM 2021, 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A gate to implement time-evolution of operators."""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
import scipy as sc

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import ParameterValueType
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.quantum_info import Pauli, SparsePauliOp, SparseObservable
import qiskit.quantum_info

if TYPE_CHECKING:
    from qiskit.synthesis.evolution import EvolutionSynthesis


class PauliEvolutionGate(Gate):
    r"""Time-evolution of an operator consisting of Paulis.

    For an Hermitian operator :math:`H` consisting of Pauli terms and (real) evolution time :math:`t`
    this gate represents the unitary

    .. math::

        U(t) = e^{-itH}.

    The evolution gates are related to the Pauli rotation gates by a factor of 2. For example
    the time evolution of the Pauli :math:`X` operator is connected to the Pauli :math:`X` rotation
    :math:`R_X` by

    .. math::

        U(t) = e^{-itX} = R_X(2t).

    Compilation:

    This gate represents the exact evolution :math:`U(t)`. Implementing this operation exactly,
    however, generally requires an exponential number of gates. The compiler therefore typically
    implements an *approximation* of the unitary :math:`U(t)`, e.g. using a product formula such
    as defined by :class:`.LieTrotter`. By passing the ``synthesis`` argument, you can specify
    which method the compiler should use, see :mod:`qiskit.synthesis` for the available options.

    Note that the order in which the approximation and methods like :meth:`control` and
    :meth:`power` are called matters. Changing the order can lead to different unitaries.

    Examples:

    .. plot::
       :include-source:
       :nofigs:

        from qiskit.circuit import QuantumCircuit
        from qiskit.circuit.library import PauliEvolutionGate
        from qiskit.quantum_info import SparsePauliOp

        X = SparsePauliOp("X")
        Z = SparsePauliOp("Z")
        I = SparsePauliOp("I")

        # build the evolution gate
        operator = (Z ^ Z) - 0.1 * (X ^ I)
        evo = PauliEvolutionGate(operator, time=0.2)

        # plug it into a circuit
        circuit = QuantumCircuit(2)
        circuit.append(evo, range(2))
        print(circuit.draw())

    The above will print (note that the ``-0.1`` coefficient is not printed!):

    .. code-block:: text

             ┌──────────────────────────┐
        q_0: ┤0                         ├
             │  exp(-it (ZZ + XI))(0.2) │
        q_1: ┤1                         ├
             └──────────────────────────┘


    References:

    [1] G. Li et al. Paulihedral: A Generalized Block-Wise Compiler Optimization
    Framework For Quantum Simulation Kernels (2021).
    `arXiv:2109.03371 <https://arxiv.org/abs/2109.03371>`__
    """

    def __init__(
        self,
        operator: (
            qiskit.quantum_info.Pauli
            | SparsePauliOp
            | SparseObservable
            | list[qiskit.quantum_info.Pauli | SparsePauliOp | SparseObservable]
        ),
        time: ParameterValueType = 1.0,
        label: str | None = None,
        synthesis: EvolutionSynthesis | None = None,
    ) -> None:
        """
        Args:
            operator: The operator to evolve. Can also be provided as list of non-commuting
                operators where the elements are sums of commuting operators.
                For example: ``[XY + YX, ZZ + ZI + IZ, YY]``.
            time: The evolution time.
            label: A label for the gate to display in visualizations. Per default, the label is
                set to ``exp(-it <operators>)`` where ``<operators>`` is the sum of the Paulis.
                Note that the label does not include any coefficients of the Paulis. See the
                class docstring for an example.
            synthesis: A synthesis strategy. If None, the default synthesis is the Lie-Trotter
                product formula with a single repetition.
        """
        if isinstance(operator, list):
            operator = [_to_sparse_op(op) for op in operator]
        else:
            operator = _to_sparse_op(operator)

        if label is None:
            label = _get_default_label(operator)

        if isinstance(operator, list):
            if len(operator) == 0:
                raise ValueError("The argument 'operator' cannot be an empty list.")
            num_qubits = operator[0].num_qubits
            for op in operator[1:]:
                if op.num_qubits != num_qubits:
                    raise ValueError(
                        "When represented as a list of operators, all of these operators "
                        "must have the same number of qubits."
                    )
        else:
            num_qubits = operator.num_qubits

        super().__init__(name="PauliEvolution", num_qubits=num_qubits, params=[time], label=label)
        self.operator = operator

        if synthesis is None:
            # pylint: disable=cyclic-import
            from qiskit.synthesis.evolution import LieTrotter

            synthesis = LieTrotter()

        self.synthesis = synthesis

    @property
    def time(self) -> ParameterValueType:
        """Return the evolution time as stored in the gate parameters.

        Returns:
            The evolution time.
        """
        return self.params[0]

    @time.setter
    def time(self, time: ParameterValueType) -> None:
        """Set the evolution time.

        Args:
            time: The evolution time.
        """
        self.params = [time]

    def to_matrix(self) -> np.ndarray:
        """Return the matrix :math:`e^{-it H}` as ``numpy.ndarray``.

        Returns:
            The matrix this gate represents.

        Raises:
            ValueError: If the ``time`` parameters is not numeric.
        """
        # check the parameter is numeric, otherwise raise an error
        if isinstance(self.time, ParameterExpression):
            try:
                time = self.time.numeric()
            except TypeError as exc:
                raise ValueError(
                    f"Cannot compute matrix with non-numeric parameter: {self.time}"
                ) from exc
        else:
            time = self.time

        # sum up all commuting terms if the operators are given as list
        if isinstance(self.operator, list):
            operator = sum(self.operator[1:], start=self.operator[0])
        else:
            operator = self.operator

        # SparseObservable does not have a to_matrix method yet
        if isinstance(operator, SparseObservable):
            operator = SparsePauliOp.from_sparse_observable(operator)

        # we use a sparse matrix representation for the exponentiation, as operators
        # are typically sparse (and if they aren't the whole thing is inefficient anyways)
        spmatrix = operator.to_matrix(sparse=True)

        exp = sc.sparse.linalg.expm(-1j * time * spmatrix)

        # return as dense matrix, since that's what the interface dictates
        return exp.toarray()

    # pylint: disable=unused-argument
    def inverse(self, annotated: bool = False):
        """Return the inverse, which is obtained by flipping the sign of the evolution time."""
        return PauliEvolutionGate(self.operator, -self.time, synthesis=self.synthesis)

    # pylint: disable=unused-argument
    def power(self, exponent: float, annotated: bool = False) -> Gate:
        """Raise this gate to the power of ``exponent``.

        The outcome represents :math:`e^{-i tp H}` where :math:`p` equals ``exponent``.

        Args:
            exponent: The power to raise the gate to.
            annotated: Not applicable to this class. Usually, when this is ``True`` we return an
                :class:`.AnnotatedOperation` with a power modifier set instead of a concrete
                :class:`.Gate`. However, we can efficiently represent powers of Pauli evolutions
                as :class:`.PauliEvolutionGate`, which is used here.

        Returns:
            An operation implementing ``gate^exponent``.
        """
        return PauliEvolutionGate(self.operator, self.time * exponent, synthesis=self.synthesis)

    def _return_repeat(self, exponent: float) -> PauliEvolutionGate:
        return self.power(exponent)  # same implementation

    # pylint: disable=unused-argument
    def control(
        self,
        num_ctrl_qubits: int = 1,
        label: str | None = None,
        ctrl_state: int | str | None = None,
        annotated: bool | None = None,
    ) -> Gate:
        r"""Return the controlled version of itself.

        The outcome is the specified controlled version of :math:`e^{-itH}`.
        The returned gate represents :math:`e^{-it H_C}`, where :math:`H_C` is the original
        operator :math:`H`, tensored with :math:`|0\rangle\langle 0|` and
        :math:`|1\rangle\langle 1|` projectors (depending on the control state).

        Args:
            num_ctrl_qubits: Number of controls to add to gate (default: ``1``).
            label: Optional gate label. Ignored if implemented as an annotated
                operation.
            ctrl_state: The control state in decimal or as a bitstring
                (e.g. ``"111"``). If ``None``, use ``2**num_ctrl_qubits - 1``.
            annotated: Not applicable to this class. Usually, when this is ``True`` we return an
                :class:`.AnnotatedOperation` with a control modifier set instead of a concrete
                :class:`.Gate`. However, we can efficiently represent controlled Pauli evolutions
                as :class:`.PauliEvolutionGate`, which is used here.

        Returns:
            Controlled version of the given operation.
        """
        if ctrl_state is None:
            ctrl_state = "1" * num_ctrl_qubits
        elif isinstance(ctrl_state, int):
            ctrl_state = bin(ctrl_state)[2:].zfill(num_ctrl_qubits)
        else:
            if len(ctrl_state) != num_ctrl_qubits:
                raise ValueError(
                    f"Length of ctrl_state ({len(ctrl_state)}) must match "
                    f"num_ctrl_qubits ({num_ctrl_qubits})"
                )

        # Implementing the controlled version of an evolution,
        #   |0><0| \otimes 1 + |1><1| \otimes exp(it H),
        # equals the evolution of the Hamiltonian extended by the |1><1| projector,
        #   exp(it |1><1| \otimes H).
        # For open controls, the control states are flipped.
        # We use the projector formalism here, which will result in a
        # circuit that only controls the central Pauli rotation. For example, calling
        # PauliEvolutionGate(Z).control(2) will produce PauliEvolutionGate(11Z).
        control_op = SparseObservable(ctrl_state)

        def extend_op(op):
            if isinstance(op, SparsePauliOp):
                op = SparseObservable.from_sparse_pauli_op(op)

            return op ^ control_op

        if isinstance(self.operator, list):
            operator = [extend_op(op) for op in self.operator]
        else:
            operator = extend_op(self.operator)

        return PauliEvolutionGate(operator, self.time, label, synthesis=self.synthesis)

    def _define(self):
        """Unroll, where the default synthesis is matrix based."""
        self.definition = self.synthesis.synthesize(self)

    def validate_parameter(self, parameter: ParameterValueType) -> ParameterValueType:
        """Gate parameters should be int, float, or ParameterExpression"""
        if isinstance(parameter, int):
            parameter = float(parameter)

        return super().validate_parameter(parameter)


def _to_sparse_op(
    operator: Pauli | SparsePauliOp | SparseObservable,
) -> SparsePauliOp | SparseObservable:
    """Cast the operator to a SparsePauliOp."""

    if isinstance(operator, Pauli):
        sparse = SparsePauliOp(operator)
    elif isinstance(operator, (SparseObservable, SparsePauliOp)):
        sparse = operator
    else:
        raise ValueError(f"Unsupported operator type for evolution: {type(operator)}.")

    if any(np.iscomplex(sparse.coeffs)):
        raise ValueError("Operator contains complex coefficients, which are not supported.")
    if any(isinstance(coeff, ParameterExpression) for coeff in sparse.coeffs):
        raise ValueError("Operator contains ParameterExpression, which are not supported.")

    return sparse


def _operator_label(operator):
    if isinstance(operator, SparseObservable):
        if len(operator) == 1:
            return operator[0].bit_labels()[::-1]
        return "(" + " + ".join(term.bit_labels()[::-1] for term in operator) + ")"

    # else: is a SparsePauliOp
    if len(operator.paulis) == 1:
        return operator.paulis.to_labels()[0]
    return "(" + " + ".join(operator.paulis.to_labels()) + ")"


def _get_default_label(operator):
    if isinstance(operator, list):
        return f"exp(-it ({[_operator_label(op) for op in operator]}))"
    return f"exp(-it {_operator_label(operator)})"
