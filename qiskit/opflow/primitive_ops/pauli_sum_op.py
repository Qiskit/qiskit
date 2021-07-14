# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""PauliSumOp Class """

from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, Union, cast

import numpy as np
from scipy.sparse import spmatrix

from qiskit.circuit import Instruction, ParameterExpression
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.list_ops.tensored_op import TensoredOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.pauli_op import PauliOp
from qiskit.opflow.primitive_ops.primitive_op import PrimitiveOp
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit.quantum_info.operators.custom_iterator import CustomIterator
from qiskit.quantum_info.operators.symplectic.pauli_table import PauliTable


class PauliSumOp(PrimitiveOp):
    """Class for Operators backend by Terra's ``SparsePauliOp`` class."""

    primitive: SparsePauliOp

    def __init__(
        self,
        primitive: SparsePauliOp,
        coeff: Union[complex, ParameterExpression] = 1.0,
        grouping_type: str = "None",
    ) -> None:
        """
        Args:
            primitive: The SparsePauliOp which defines the behavior of the underlying function.
            coeff: A coefficient multiplying the primitive.
            grouping_type: The type of grouping. If None, the operator is not grouped.

        Raises:
            TypeError: invalid parameters.
        """
        if not isinstance(primitive, SparsePauliOp):
            raise TypeError(
                f"PauliSumOp can only be instantiated with SparsePauliOp, not {type(primitive)}"
            )

        super().__init__(primitive, coeff=coeff)
        self._grouping_type = grouping_type

    def primitive_strings(self) -> Set[str]:
        return {"SparsePauliOp"}

    @property
    def grouping_type(self) -> str:
        """
        Returns: Type of Grouping
        """
        return self._grouping_type

    @property
    def num_qubits(self) -> int:
        return self.primitive.num_qubits

    @property
    def coeffs(self):
        """Return the Pauli coefficients."""
        return self.coeff * self.primitive.coeffs

    @property
    def settings(self) -> Dict:
        """Return operator settings."""
        data = super().settings
        data.update({"grouping_type": self._grouping_type})
        return data

    def matrix_iter(self, sparse=False):
        """Return a matrix representation iterator.

        This is a lazy iterator that converts each term in the PauliSumOp
        into a matrix as it is used. To convert to a single matrix use the
        :meth:`to_matrix` method.

        Args:
            sparse (bool): optionally return sparse CSR matrices if True,
                           otherwise return Numpy array matrices
                           (Default: False)

        Returns:
            MatrixIterator: matrix iterator object for the PauliTable.
        """

        class MatrixIterator(CustomIterator):
            """Matrix representation iteration and item access."""

            def __repr__(self):
                return f"<PauliSumOp_matrix_iterator at {hex(id(self))}>"

            def __getitem__(self, key):
                sumopcoeff = self.obj.coeff * self.obj.primitive.coeffs[key]
                mat = PauliTable._to_matrix(self.obj.primitive.table.array[key], sparse=sparse)
                return sumopcoeff * mat

        return MatrixIterator(self)

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                f"Sum of operators with different numbers of qubits, {self.num_qubits} and "
                f"{other.num_qubits}, is not well defined"
            )

        if (
            isinstance(other, PauliSumOp)
            and not isinstance(self.coeff, ParameterExpression)
            and not isinstance(other.coeff, ParameterExpression)
        ):
            return PauliSumOp(self.coeff * self.primitive + other.coeff * other.primitive, coeff=1)

        if (
            isinstance(other, PauliOp)
            and not isinstance(self.coeff, ParameterExpression)
            and not isinstance(other.coeff, ParameterExpression)
        ):
            return PauliSumOp(
                self.coeff * self.primitive + other.coeff * SparsePauliOp(other.primitive)
            )

        return SummedOp([self, other])

    def mul(self, scalar: Union[complex, ParameterExpression]) -> OperatorBase:
        if isinstance(scalar, (int, float, complex)) and scalar != 0:
            return PauliSumOp(scalar * self.primitive, coeff=self.coeff)

        return PauliSumOp(self.primitive, coeff=self.coeff * scalar)

    def adjoint(self) -> "PauliSumOp":
        return PauliSumOp(self.primitive.adjoint(), coeff=self.coeff.conjugate())

    def equals(self, other: OperatorBase) -> bool:
        self_reduced, other_reduced = self.reduce(), other.reduce()

        if not isinstance(other_reduced, PauliSumOp):
            return False

        if isinstance(self_reduced.coeff, ParameterExpression) or isinstance(
            other_reduced.coeff, ParameterExpression
        ):
            return (
                self_reduced.coeff == other_reduced.coeff
                and self_reduced.primitive == other_reduced.primitive
            )
        return (
            len(self_reduced) == len(other_reduced)
            and self_reduced.primitive == other_reduced.primitive
        )

    def _expand_dim(self, num_qubits: int) -> "PauliSumOp":
        return PauliSumOp(
            self.primitive.tensor(SparsePauliOp(Pauli("I" * num_qubits))),
            coeff=self.coeff,
        )

    def tensor(self, other: OperatorBase) -> Union["PauliSumOp", TensoredOp]:
        if isinstance(other, PauliSumOp):
            return PauliSumOp(
                self.primitive.tensor(other.primitive),
                coeff=self.coeff * other.coeff,
            )
        if isinstance(other, PauliOp):
            return PauliSumOp(
                self.primitive.tensor(other.primitive),
                coeff=self.coeff * other.coeff,
            )

        return TensoredOp([self, other])

    def permute(self, permutation: List[int]) -> "PauliSumOp":
        """Permutes the sequence of ``PauliSumOp``.

        Args:
            permutation: A list defining where each Pauli should be permuted. The Pauli at index
                j of the primitive should be permuted to position permutation[j].

        Returns:
              A new PauliSumOp representing the permuted operator. For operator (X ^ Y ^ Z) and
              indices=[1,2,4], it returns (X ^ I ^ Y ^ Z ^ I).

        Raises:
            OpflowError: if indices do not define a new index for each qubit.
        """
        if len(permutation) != self.num_qubits:
            raise OpflowError(
                "List of indices to permute must have the " "same size as Pauli Operator"
            )
        length = max(permutation) + 1
        spop = self.primitive.tensor(SparsePauliOp(Pauli("I" * (length - self.num_qubits))))
        permutation = [i for i in range(length) if i not in permutation] + permutation
        permu_arr = np.arange(length)[np.argsort(permutation)]
        permu_arr = np.hstack([permu_arr, permu_arr + length])
        spop.table.array = spop.table.array[:, permu_arr]
        return PauliSumOp(spop, self.coeff)

    def compose(
        self,
        other: OperatorBase,
        permutation: Optional[List[int]] = None,
        front: bool = False,
    ) -> OperatorBase:

        new_self, other = self._expand_shorter_operator_and_permute(other, permutation)
        new_self = cast(PauliSumOp, new_self)

        if front:
            return other.compose(new_self)
        # If self is identity, just return other.
        if not np.any(new_self.primitive.table.array):
            return other * new_self.coeff * sum(new_self.primitive.coeffs)

        # Both PauliSumOps
        if isinstance(other, PauliSumOp):
            return PauliSumOp(
                new_self.primitive.dot(other.primitive),
                coeff=new_self.coeff * other.coeff,
            )
        if isinstance(other, PauliOp):
            other_primitive = SparsePauliOp(other.primitive)
            return PauliSumOp(
                new_self.primitive.dot(other_primitive),
                coeff=new_self.coeff * other.coeff,
            )

        # pylint: disable=cyclic-import
        from ..state_fns.circuit_state_fn import CircuitStateFn
        from .circuit_op import CircuitOp

        if isinstance(other, (CircuitOp, CircuitStateFn)):
            pauli_op = cast(Union[PauliOp, SummedOp], new_self.to_pauli_op())
            return pauli_op.to_circuit_op().compose(other)

        return super(PauliSumOp, new_self).compose(other)

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        OperatorBase._check_massive("to_matrix", True, self.num_qubits, massive)
        if isinstance(self.coeff, ParameterExpression):
            return (self.primitive.to_matrix(sparse=True)).toarray() * self.coeff
        return (self.primitive.to_matrix(sparse=True) * self.coeff).toarray()

    def __str__(self) -> str:
        def format_sign(x):
            return x.real if np.isreal(x) else x

        def format_number(x):
            x = format_sign(x)
            if isinstance(x, (int, float)) and x < 0:
                return f"- {-x}"
            return f"+ {x}"

        indent = "" if self.coeff == 1 else "  "
        prim_list = self.primitive.to_list()
        if prim_list:
            first = prim_list[0]
            if isinstance(first[1], (int, float)) and first[1] < 0:
                main_string = indent + f"- {-first[1].real} * {first[0]}"
            else:
                main_string = indent + f"{format_sign(first[1])} * {first[0]}"

        main_string += "".join([f"\n{indent}{format_number(c)} * {p}" for p, c in prim_list[1:]])
        return f"{main_string}" if self.coeff == 1 else f"{self.coeff} * (\n{main_string}\n)"

    def eval(
        self,
        front: Optional[
            Union[str, Dict[str, complex], np.ndarray, OperatorBase, Statevector]
        ] = None,
    ) -> Union[OperatorBase, complex]:
        if front is None:
            return self.to_matrix_op()

        # pylint: disable=cyclic-import
        from ..list_ops.list_op import ListOp
        from ..state_fns.circuit_state_fn import CircuitStateFn
        from ..state_fns.dict_state_fn import DictStateFn
        from ..state_fns.state_fn import StateFn
        from .circuit_op import CircuitOp

        # For now, always do this. If it's not performant, we can be more granular.
        if not isinstance(front, OperatorBase):
            front = StateFn(front, is_measurement=False)

        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn(
                [self.eval(front.coeff * front_elem) for front_elem in front.oplist]
            )

        else:

            if self.num_qubits != front.num_qubits:
                raise ValueError(
                    "eval does not support operands with differing numbers of qubits, "
                    "{} and {}, respectively.".format(self.num_qubits, front.num_qubits)
                )

            if isinstance(front, DictStateFn):
                new_dict: Dict[str, int] = defaultdict(int)
                corrected_x_bits = self.primitive.table.X[::, ::-1]
                corrected_z_bits = self.primitive.table.Z[::, ::-1]
                coeffs = self.primitive.coeffs
                for bstr, v in front.primitive.items():
                    bitstr = np.fromiter(bstr, dtype=int).astype(bool)
                    new_b_str = np.logical_xor(bitstr, corrected_x_bits)
                    new_str = ["".join([str(b) for b in bs]) for bs in new_b_str.astype(int)]
                    z_factor = np.product(1 - 2 * np.logical_and(bitstr, corrected_z_bits), axis=1)
                    y_factor = np.product(
                        np.sqrt(1 - 2 * np.logical_and(corrected_x_bits, corrected_z_bits) + 0j),
                        axis=1,
                    )
                    for i, n_str in enumerate(new_str):
                        new_dict[n_str] += v * z_factor[i] * y_factor[i] * coeffs[i]
                return DictStateFn(new_dict, coeff=self.coeff * front.coeff)

            elif isinstance(front, StateFn) and front.is_measurement:
                raise ValueError("Operator composed with a measurement is undefined.")

            # Composable types with PauliOp
            elif isinstance(front, (PauliSumOp, PauliOp, CircuitOp, CircuitStateFn)):
                return self.compose(front).eval()

        # Covers VectorStateFn and OperatorStateFn
        front = cast(StateFn, front)
        return self.to_matrix_op().eval(front.to_matrix_op())

    def exp_i(self) -> OperatorBase:
        """Return a ``CircuitOp`` equivalent to e^-iH for this operator H."""
        # TODO: optimize for some special cases
        from ..evolutions.evolved_op import EvolvedOp

        return EvolvedOp(self)

    def to_instruction(self) -> Instruction:
        return self.to_matrix_op().to_circuit().to_instruction()  # type: ignore

    def to_pauli_op(self, massive: bool = False) -> Union[PauliOp, SummedOp]:
        def to_native(x):
            return x.item() if isinstance(x, np.generic) else x

        if len(self.primitive) == 1:
            return PauliOp(
                Pauli((self.primitive.table.Z[0], self.primitive.table.X[0])),
                to_native(np.real_if_close(self.primitive.coeffs[0])) * self.coeff,
            )
        tables = self.primitive.table
        coeffs = np.real_if_close(self.primitive.coeffs)
        return SummedOp(
            [
                PauliOp(
                    Pauli((t.Z[0], t.X[0])),
                    to_native(c),
                )
                for t, c in zip(tables, coeffs)
            ],
            coeff=self.coeff,
        )

    def __getitem__(self, offset: Union[int, slice]) -> "PauliSumOp":
        """Allows array-indexing style access to the ``PauliSumOp``.

        Args:
            offset: The index of ``PauliSumOp``.

        Returns:
            The ``PauliSumOp`` at index ``offset``,
        """
        return PauliSumOp(self.primitive[offset], self.coeff)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        """Length of ``SparsePauliOp``.

        Returns:
            An int equal to the length of SparsePauliOp.
        """
        return len(self.primitive)

    # pylint: disable=arguments-differ
    def reduce(self, atol: Optional[float] = None, rtol: Optional[float] = None) -> "PauliSumOp":
        """Simplify the primitive ``SparsePauliOp``.

        Args:
            atol: Absolute tolerance for checking if coefficients are zero (Default: 1e-8).
            rtol: Relative tolerance for checking if coefficients are zero (Default: 1e-5).

        Returns:
            The simplified ``PauliSumOp``.
        """
        if isinstance(self.coeff, (int, float, complex)):
            primitive = self.coeff * self.primitive
            return PauliSumOp(primitive.simplify(atol=atol, rtol=rtol))
        return PauliSumOp(self.primitive.simplify(atol=atol, rtol=rtol), self.coeff)

    def to_spmatrix(self) -> spmatrix:
        """Returns SciPy sparse matrix representation of the ``PauliSumOp``.

        Returns:
            CSR sparse matrix representation of the ``PauliSumOp``.

        Raises:
            ValueError: invalid parameters.
        """
        return self.primitive.to_matrix(sparse=True) * self.coeff

    @classmethod
    def from_list(
        cls,
        pauli_list: List[Tuple[str, complex]],
        coeff: Union[complex, ParameterExpression] = 1.0,
    ) -> "PauliSumOp":
        """Construct from a pauli_list with the form [(pauli_str, coeffs)]

        Args:
            pauli_list: A list of Tuple of pauli_str and coefficient.
            coeff: A coefficient multiplying the primitive.

        Returns:
            The PauliSumOp constructed from the pauli_list.
        """
        return cls(SparsePauliOp.from_list(pauli_list), coeff=coeff)

    def is_zero(self) -> bool:
        """
        Return this operator is zero operator or not.
        """
        op = self.reduce()
        primitive: SparsePauliOp = op.primitive
        return op.coeff == 1 and len(op) == 1 and primitive.coeffs[0] == 0
