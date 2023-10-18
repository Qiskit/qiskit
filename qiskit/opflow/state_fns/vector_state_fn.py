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

"""VectorStateFn Class"""

import warnings
from typing import Dict, List, Optional, Set, Union, cast

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterExpression
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.list_ops.tensored_op import TensoredOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.quantum_info import Statevector
from qiskit.utils import algorithm_globals, arithmetic
from qiskit.utils.deprecation import deprecate_func


class VectorStateFn(StateFn):
    """Deprecated: A class for state functions and measurements which are defined in vector
    representation, and stored using Terra's ``Statevector`` class.
    """

    primitive: Statevector

    # TODO allow normalization somehow?
    @deprecate_func(
        since="0.24.0",
        package_name="qiskit-terra",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(
        self,
        primitive: Union[list, np.ndarray, Statevector] = None,
        coeff: Union[complex, ParameterExpression] = 1.0,
        is_measurement: bool = False,
    ) -> None:
        """
        Args:
            primitive: The ``Statevector``, NumPy array, or list, which defines the behavior of
                the underlying function.
            coeff: A coefficient multiplying the state function.
            is_measurement: Whether the StateFn is a measurement operator
        """
        # Lists and Numpy arrays representing statevectors are stored
        # in Statevector objects for easier handling.
        if isinstance(primitive, (np.ndarray, list)):
            primitive = Statevector(primitive)

        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)

    def primitive_strings(self) -> Set[str]:
        return {"Vector"}

    @property
    def num_qubits(self) -> int:
        return len(self.primitive.dims())

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                "Sum over statefns with different numbers of qubits, {} and {}, is not well "
                "defined".format(self.num_qubits, other.num_qubits)
            )

        # Right now doesn't make sense to add a StateFn to a Measurement
        if isinstance(other, VectorStateFn) and self.is_measurement == other.is_measurement:
            # Covers Statevector and custom.
            return VectorStateFn(
                (self.coeff * self.primitive) + (other.primitive * other.coeff),
                is_measurement=self._is_measurement,
            )
        return SummedOp([self, other])

    def adjoint(self) -> "VectorStateFn":
        return VectorStateFn(
            self.primitive.conjugate(),
            coeff=self.coeff.conjugate(),
            is_measurement=(not self.is_measurement),
        )

    def permute(self, permutation: List[int]) -> "VectorStateFn":
        new_self = self
        new_num_qubits = max(permutation) + 1

        if self.num_qubits != len(permutation):
            # raise OpflowError("New index must be defined for each qubit of the operator.")
            pass
        if self.num_qubits < new_num_qubits:
            # pad the operator with identities
            new_self = self._expand_dim(new_num_qubits - self.num_qubits)
        qc = QuantumCircuit(new_num_qubits)

        # extend the permutation indices to match the size of the new matrix
        permutation = (
            list(filter(lambda x: x not in permutation, range(new_num_qubits))) + permutation
        )

        # decompose permutation into sequence of transpositions
        transpositions = arithmetic.transpositions(permutation)
        for trans in transpositions:
            qc.swap(trans[0], trans[1])

        from ..primitive_ops.circuit_op import CircuitOp

        matrix = CircuitOp(qc).to_matrix()
        vector = new_self.primitive.data
        new_vector = cast(np.ndarray, matrix.dot(vector))
        return VectorStateFn(
            primitive=new_vector, coeff=self.coeff, is_measurement=self.is_measurement
        )

    def to_dict_fn(self) -> StateFn:
        """Creates the equivalent state function of type DictStateFn.

        Returns:
            A new DictStateFn equivalent to ``self``.
        """
        from .dict_state_fn import DictStateFn

        num_qubits = self.num_qubits
        new_dict = {format(i, "b").zfill(num_qubits): v for i, v in enumerate(self.primitive.data)}
        return DictStateFn(new_dict, coeff=self.coeff, is_measurement=self.is_measurement)

    def _expand_dim(self, num_qubits: int) -> "VectorStateFn":
        primitive = np.zeros(2**num_qubits, dtype=complex)
        return VectorStateFn(
            self.primitive.tensor(primitive), coeff=self.coeff, is_measurement=self.is_measurement
        )

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if isinstance(other, VectorStateFn):
            return StateFn(
                self.primitive.tensor(other.primitive),
                coeff=self.coeff * other.coeff,
                is_measurement=self.is_measurement,
            )
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        OperatorBase._check_massive("to_density_matrix", True, self.num_qubits, massive)
        return self.primitive.to_operator().data * self.coeff

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        OperatorBase._check_massive("to_matrix", False, self.num_qubits, massive)
        vec = self.primitive.data * self.coeff
        return vec if not self.is_measurement else vec.reshape(1, -1)

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        return self

    def to_circuit_op(self) -> OperatorBase:
        """Return ``StateFnCircuit`` corresponding to this StateFn."""
        # pylint: disable=cyclic-import
        from .circuit_state_fn import CircuitStateFn

        csfn = CircuitStateFn.from_vector(self.primitive.data) * self.coeff
        return csfn.adjoint() if self.is_measurement else csfn

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format(
                "VectorStateFn" if not self.is_measurement else "MeasurementVector", prim_str
            )
        else:
            return "{}({}) * {}".format(
                "VectorStateFn" if not self.is_measurement else "MeasurementVector",
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
        if front is None:  # this object is already a VectorStateFn
            return self

        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError(
                "Cannot compute overlap with StateFn or Operator if not Measurement. Try taking "
                "sf.adjoint() first to convert to measurement."
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
        deterministic_counts = self.primitive.probabilities_dict()
        # Don't need to square because probabilities_dict already does.
        probs = np.array(list(deterministic_counts.values()))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
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
