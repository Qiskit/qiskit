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

"""SparseVectorStateFn class."""


from typing import Dict, Optional, Set, Union

import numpy as np
import scipy

from qiskit.circuit import ParameterExpression
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.opflow.state_fns.vector_state_fn import VectorStateFn
from qiskit.quantum_info import Statevector
from qiskit.utils import algorithm_globals
from qiskit.utils.deprecation import deprecate_func


class SparseVectorStateFn(StateFn):
    """Deprecated: A class for sparse state functions and measurements in vector representation.

    This class uses ``scipy.sparse.spmatrix`` for the internal representation.
    """

    primitive: scipy.sparse.spmatrix

    # TODO allow normalization somehow?
    @deprecate_func(
        since="0.24.0",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(
        self,
        primitive: scipy.sparse.spmatrix,
        coeff: Union[complex, ParameterExpression] = 1.0,
        is_measurement: bool = False,
    ) -> None:
        """
        Args:
            primitive: The underlying sparse vector.
            coeff: A coefficient multiplying the state function.
            is_measurement: Whether the StateFn is a measurement operator

        Raises:
            ValueError: If the primitive is not a column vector.
            ValueError: If the number of elements in the primitive is not a power of 2.

        """
        if primitive.shape[0] != 1:
            raise ValueError("The primitive must be a row vector of shape (x, 1).")

        # check if the primitive is a statevector of 2^n elements
        self._num_qubits = int(np.log2(primitive.shape[1]))
        if np.log2(primitive.shape[1]) != self._num_qubits:
            raise ValueError("The number of vector elements must be a power of 2.")

        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)

    def primitive_strings(self) -> Set[str]:
        return {"SparseVector"}

    @property
    def num_qubits(self) -> int:
        return self._num_qubits

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                "Sum over statefns with different numbers of qubits, {} and {}, is not well "
                "defined".format(self.num_qubits, other.num_qubits)
            )

        # Right now doesn't make sense to add a StateFn to a Measurement
        if isinstance(other, SparseVectorStateFn) and self.is_measurement == other.is_measurement:
            # Covers Statevector and custom.
            added = self.coeff * self.primitive + other.coeff * other.primitive
            return SparseVectorStateFn(added, is_measurement=self._is_measurement)

        return SummedOp([self, other])

    def adjoint(self) -> "SparseVectorStateFn":
        return SparseVectorStateFn(
            self.primitive.conjugate(),
            coeff=self.coeff.conjugate(),
            is_measurement=(not self.is_measurement),
        )

    def equals(self, other: OperatorBase) -> bool:
        if not isinstance(other, SparseVectorStateFn) or not self.coeff == other.coeff:
            return False

        if self.primitive.shape != other.primitive.shape:
            return False

        if self.primitive.count_nonzero() != other.primitive.count_nonzero():
            return False

        # equal if no elements are different (using != for efficiency)
        return (self.primitive != other.primitive).nnz == 0

    def to_dict_fn(self) -> StateFn:
        """Convert this state function to a ``DictStateFn``.

        Returns:
            A new DictStateFn equivalent to ``self``.
        """
        from .dict_state_fn import DictStateFn

        num_qubits = self.num_qubits
        dok = self.primitive.todok()
        new_dict = {format(i[1], "b").zfill(num_qubits): v for i, v in dok.items()}
        return DictStateFn(new_dict, coeff=self.coeff, is_measurement=self.is_measurement)

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        OperatorBase._check_massive("to_matrix", False, self.num_qubits, massive)
        vec = self.primitive.toarray() * self.coeff
        return vec if not self.is_measurement else vec.reshape(1, -1)

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        return VectorStateFn(self.to_matrix())

    def to_spmatrix(self) -> OperatorBase:
        return self

    def to_circuit_op(self) -> OperatorBase:
        """Convert this state function to a ``CircuitStateFn``."""
        # pylint: disable=cyclic-import
        from .circuit_state_fn import CircuitStateFn

        csfn = CircuitStateFn.from_vector(self.primitive) * self.coeff
        return csfn.adjoint() if self.is_measurement else csfn

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format(
                "SparseVectorStateFn" if not self.is_measurement else "MeasurementSparseVector",
                prim_str,
            )
        else:
            return "{}({}) * {}".format(
                "SparseVectorStateFn" if not self.is_measurement else "SparseMeasurementVector",
                prim_str,
                self.coeff,
            )

    # pylint: disable=too-many-return-statements
    def eval(
        self,
        front: Optional[
            Union[str, Dict[str, complex], np.ndarray, Statevector, OperatorBase]
        ] = None,
    ) -> Union[OperatorBase, complex]:
        if front is None:
            return self

        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError(
                "Cannot compute overlap with StateFn or Operator if not Measurement. "
                "Try taking sf.adjoint() first to convert to measurement."
            )

        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn(
                [self.eval(front.coeff * front_elem) for front_elem in front.oplist]
            )

        if not isinstance(front, OperatorBase):
            front = StateFn(front)

        # pylint: disable=cyclic-import
        from ..operator_globals import EVAL_SIG_DIGITS
        from .operator_state_fn import OperatorStateFn
        from .circuit_state_fn import CircuitStateFn
        from .dict_state_fn import DictStateFn

        if isinstance(front, DictStateFn):
            return np.round(
                sum(
                    v * self.primitive.data[int(b, 2)] * front.coeff
                    for (b, v) in front.primitive.items()
                )
                * self.coeff,
                decimals=EVAL_SIG_DIGITS,
            )

        if isinstance(front, VectorStateFn):
            # Need to extract the element or np.array([1]) is returned.
            return np.round(
                np.dot(self.to_matrix(), front.to_matrix())[0], decimals=EVAL_SIG_DIGITS
            )

        if isinstance(front, CircuitStateFn):
            # Don't reimplement logic from CircuitStateFn
            return np.conj(front.adjoint().eval(self.adjoint().primitive)) * self.coeff

        if isinstance(front, OperatorStateFn):
            return front.adjoint().eval(self.primitive) * self.coeff

        return front.adjoint().eval(self.adjoint().primitive).adjoint() * self.coeff  # type: ignore

    def sample(
        self, shots: int = 1024, massive: bool = False, reverse_endianness: bool = False
    ) -> dict:
        as_dict = self.to_dict_fn().primitive
        all_states = sum(as_dict.keys())
        deterministic_counts = {key: value / all_states for key, value in as_dict.items()}
        # Don't need to square because probabilities_dict already does.
        probs = np.array(list(deterministic_counts.values()))
        unique, counts = np.unique(
            algorithm_globals.random.choice(
                list(deterministic_counts.keys()), size=shots, p=(probs / sum(probs))
            ),
            return_counts=True,
        )
        counts = dict(zip(unique, counts))
        if reverse_endianness:
            scaled_dict = {bstr[::-1]: (prob / shots) for (bstr, prob) in counts.items()}
        else:
            scaled_dict = {bstr: (prob / shots) for (bstr, prob) in counts.items()}
        return dict(sorted(scaled_dict.items(), key=lambda x: x[1], reverse=True))
