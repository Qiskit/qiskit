# This code is part of Qiskit.
#
# (C) Copyright IBM 2020, 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""MatrixOp Class"""

from typing import Dict, List, Optional, Set, Union, cast, get_type_hints
import numpy as np
from scipy.sparse import spmatrix

from qiskit import QuantumCircuit
from qiskit.circuit import Instruction, ParameterExpression
from qiskit.extensions.hamiltonian_gate import HamiltonianGate
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.list_ops.tensored_op import TensoredOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.circuit_op import CircuitOp
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.quantum_info import Operator, Statevector
from qiskit.utils import arithmetic
from qiskit.utils.deprecation import deprecate_func


class MatrixOp(PrimitiveOp):
    """Deprecated: Class for Operators represented by matrices,
    backed by Terra's ``Operator`` module."""

    primitive: Operator

    @deprecate_func(
        since="0.24.0",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(
        self,
        primitive: Union[list, np.ndarray, spmatrix, Operator],
        coeff: Union[complex, ParameterExpression] = 1.0,
    ) -> None:
        """
        Args:
            primitive: The matrix-like object which defines the behavior of the underlying function.
            coeff: A coefficient multiplying the primitive

        Raises:
            TypeError: invalid parameters.
            ValueError: invalid parameters.
        """
        primitive_orig = primitive
        if isinstance(primitive, spmatrix):
            primitive = primitive.toarray()

        if isinstance(primitive, (list, np.ndarray)):
            primitive = Operator(primitive)

        if not isinstance(primitive, Operator):
            type_hints = get_type_hints(MatrixOp.__init__).get("primitive")
            valid_cls = [cls.__name__ for cls in type_hints.__args__]
            raise TypeError(
                f"MatrixOp can only be instantiated with {valid_cls}, "
                f"not '{primitive_orig.__class__.__name__}'"
            )

        if primitive.input_dims() != primitive.output_dims():
            raise ValueError("Cannot handle non-square matrices yet.")

        super().__init__(primitive, coeff=coeff)

    def primitive_strings(self) -> Set[str]:
        return {"Matrix"}

    @property
    def num_qubits(self) -> int:
        return len(self.primitive.input_dims())

    def add(self, other: OperatorBase) -> Union["MatrixOp", SummedOp]:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                "Sum over operators with different numbers of qubits, {} and {}, is not well "
                "defined".format(self.num_qubits, other.num_qubits)
            )

        if isinstance(other, MatrixOp) and self.primitive == other.primitive:
            return MatrixOp(self.primitive, coeff=self.coeff + other.coeff)

        # Terra's Operator cannot handle ParameterExpressions
        if (
            isinstance(other, MatrixOp)
            and not isinstance(self.coeff, ParameterExpression)
            and not isinstance(other.coeff, ParameterExpression)
        ):
            return MatrixOp((self.coeff * self.primitive) + (other.coeff * other.primitive))

        # Covers Paulis, Circuits, and all else.
        return SummedOp([self, other])

    def adjoint(self) -> "MatrixOp":
        return MatrixOp(self.primitive.adjoint(), coeff=self.coeff.conjugate())

    def equals(self, other: OperatorBase) -> bool:
        if not isinstance(other, MatrixOp):
            return False
        if isinstance(self.coeff, ParameterExpression) ^ isinstance(
            other.coeff, ParameterExpression
        ):
            return False
        if isinstance(self.coeff, ParameterExpression) and isinstance(
            other.coeff, ParameterExpression
        ):
            return self.coeff == other.coeff and self.primitive == other.primitive
        return self.coeff * self.primitive == other.coeff * other.primitive

    def _expand_dim(self, num_qubits: int) -> "MatrixOp":
        identity = np.identity(2**num_qubits, dtype=complex)
        return MatrixOp(self.primitive.tensor(Operator(identity)), coeff=self.coeff)

    def tensor(self, other: OperatorBase) -> Union["MatrixOp", TensoredOp]:
        if isinstance(other, MatrixOp):
            return MatrixOp(self.primitive.tensor(other.primitive), coeff=self.coeff * other.coeff)

        return TensoredOp([self, other])

    def compose(
        self, other: OperatorBase, permutation: Optional[List[int]] = None, front: bool = False
    ) -> OperatorBase:

        new_self, other = self._expand_shorter_operator_and_permute(other, permutation)
        new_self = cast(MatrixOp, new_self)

        if front:
            return other.compose(new_self)
        if isinstance(other, MatrixOp):
            return MatrixOp(
                new_self.primitive.compose(other.primitive, front=True),
                coeff=new_self.coeff * other.coeff,
            )

        return super(MatrixOp, new_self).compose(other)

    def permute(self, permutation: Optional[List[int]] = None) -> OperatorBase:
        """Creates a new MatrixOp that acts on the permuted qubits.

        Args:
            permutation: A list defining where each qubit should be permuted. The qubit at index
                j should be permuted to position permutation[j].

        Returns:
            A new MatrixOp representing the permuted operator.

        Raises:
            OpflowError: if indices do not define a new index for each qubit.
        """
        new_self = self
        new_matrix_size = max(permutation) + 1

        if self.num_qubits != len(permutation):
            raise OpflowError("New index must be defined for each qubit of the operator.")
        if self.num_qubits < new_matrix_size:
            # pad the operator with identities
            new_self = self._expand_dim(new_matrix_size - self.num_qubits)
        qc = QuantumCircuit(new_matrix_size)

        # extend the indices to match the size of the new matrix
        permutation = (
            list(filter(lambda x: x not in permutation, range(new_matrix_size))) + permutation
        )

        # decompose permutation into sequence of transpositions
        transpositions = arithmetic.transpositions(permutation)
        for trans in transpositions:
            qc.swap(trans[0], trans[1])
        matrix = CircuitOp(qc).to_matrix()
        return MatrixOp(matrix.transpose()) @ new_self @ MatrixOp(matrix)

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        return self.primitive.data * self.coeff

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return prim_str
        else:
            return f"{self.coeff} * {prim_str}"

    def eval(
        self,
        front: Optional[
            Union[str, Dict[str, complex], np.ndarray, OperatorBase, Statevector]
        ] = None,
    ) -> Union[OperatorBase, complex]:
        # For other ops' eval we return self.to_matrix_op() here, but that's unnecessary here.
        if front is None:
            return self

        # pylint: disable=cyclic-import
        from ..list_ops import ListOp
        from ..state_fns import StateFn, VectorStateFn, OperatorStateFn

        new_front = None

        # For now, always do this. If it's not performant, we can be more granular.
        if not isinstance(front, OperatorBase):
            front = StateFn(front, is_measurement=False)

        if isinstance(front, ListOp) and front.distributive:
            new_front = front.combo_fn(
                [self.eval(front.coeff * front_elem) for front_elem in front.oplist]
            )

        elif isinstance(front, OperatorStateFn):
            new_front = OperatorStateFn(self.adjoint().compose(front.to_matrix_op()).compose(self))

        elif isinstance(front, OperatorBase):
            new_front = VectorStateFn(self.to_matrix() @ front.to_matrix())

        return new_front

    def exp_i(self) -> OperatorBase:
        """Return a ``CircuitOp`` equivalent to e^-iH for this operator H"""
        return CircuitOp(HamiltonianGate(self.primitive, time=self.coeff))

    # Op Conversions

    def to_matrix_op(self, massive: bool = False) -> "MatrixOp":
        return self

    def to_instruction(self) -> Instruction:
        return (self.coeff * self.primitive).to_instruction()
