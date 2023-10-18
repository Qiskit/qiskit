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

"""DictStateFn Class"""

import itertools
import warnings
from typing import Dict, List, Optional, Set, Union, cast

import numpy as np
from scipy import sparse

from qiskit.circuit import ParameterExpression
from qiskit.opflow.exceptions import OpflowError
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.opflow.state_fns.vector_state_fn import VectorStateFn
from qiskit.quantum_info import Statevector
from qiskit.result import Result
from qiskit.utils import algorithm_globals
from qiskit.utils.deprecation import deprecate_func


class DictStateFn(StateFn):
    """Deprecated: A class for state functions and measurements which are defined by a lookup table,
    stored in a dict.
    """

    primitive: Dict[str, complex]

    # TODO allow normalization somehow?
    @deprecate_func(
        since="0.24.0",
        package_name="qiskit-terra",
        additional_msg="For code migration guidelines, visit https://qisk.it/opflow_migration.",
    )
    def __init__(
        self,
        primitive: Union[str, dict, Result] = None,
        coeff: Union[complex, ParameterExpression] = 1.0,
        is_measurement: bool = False,
        from_operator: bool = False,
    ) -> None:
        """
        Args:
            primitive: The dict, single bitstring (if defining a basis sate), or Qiskit
                Result, which defines the behavior of the underlying function.
            coeff: A coefficient by which to multiply the state function.
            is_measurement: Whether the StateFn is a measurement operator.
            from_operator: if True the StateFn is derived from OperatorStateFn. (Default: False)

        Raises:
            TypeError: invalid parameters.
        """
        # If the initial density is a string, treat this as a density dict
        # with only a single basis state.
        if isinstance(primitive, str):
            primitive = {primitive: 1}

        # NOTE:
        # 1) This is not the same as passing in the counts dict directly, as this will
        # convert the shot numbers to
        # probabilities, whereas passing in the counts dict will not.
        # 2) This will extract counts for both shot and statevector simulations.
        # To use the statevector,
        # simply pass in the statevector.
        # 3) This will only extract the first result.
        if isinstance(primitive, Result):
            counts = primitive.get_counts()
            # NOTE: Need to square root to take correct Pauli measurements!
            primitive = {
                bstr: (shots / sum(counts.values())) ** 0.5 for (bstr, shots) in counts.items()
            }

        if not isinstance(primitive, dict):
            raise TypeError(
                "DictStateFn can only be instantiated with dict, "
                "string, or Qiskit Result, not {}".format(type(primitive))
            )

        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)
        self.from_operator = from_operator

    def primitive_strings(self) -> Set[str]:
        return {"Dict"}

    @property
    def num_qubits(self) -> int:
        return len(next(iter(self.primitive)))

    @property
    def settings(self) -> Dict:
        """Return settings."""
        data = super().settings
        data["from_operator"] = self.from_operator
        return data

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                "Sum over statefns with different numbers of qubits, {} and {}, is not well "
                "defined".format(self.num_qubits, other.num_qubits)
            )

        # Right now doesn't make sense to add a StateFn to a Measurement
        if isinstance(other, DictStateFn) and self.is_measurement == other.is_measurement:
            # TODO add compatibility with vector and Operator?
            if self.primitive == other.primitive:
                return DictStateFn(
                    self.primitive,
                    coeff=self.coeff + other.coeff,
                    is_measurement=self.is_measurement,
                )
            else:
                new_dict = {
                    b: (v * self.coeff) + (other.primitive.get(b, 0) * other.coeff)
                    for (b, v) in self.primitive.items()
                }
                new_dict.update(
                    {
                        b: v * other.coeff
                        for (b, v) in other.primitive.items()
                        if b not in self.primitive
                    }
                )
                return DictStateFn(new_dict, is_measurement=self._is_measurement)
        # pylint: disable=cyclic-import
        from ..list_ops.summed_op import SummedOp

        return SummedOp([self, other])

    def adjoint(self) -> "DictStateFn":
        return DictStateFn(
            {b: np.conj(v) for (b, v) in self.primitive.items()},
            coeff=self.coeff.conjugate(),
            is_measurement=(not self.is_measurement),
        )

    def permute(self, permutation: List[int]) -> "DictStateFn":
        new_num_qubits = max(permutation) + 1
        if self.num_qubits != len(permutation):
            raise OpflowError("New index must be defined for each qubit of the operator.")

        # helper function to permute the key
        def perm(key):
            list_key = ["0"] * new_num_qubits
            for i, k in enumerate(permutation):
                list_key[k] = key[i]
            return "".join(list_key)

        new_dict = {perm(key): value for key, value in self.primitive.items()}
        return DictStateFn(new_dict, coeff=self.coeff, is_measurement=self.is_measurement)

    def _expand_dim(self, num_qubits: int) -> "DictStateFn":
        pad = "0" * num_qubits
        new_dict = {key + pad: value for key, value in self.primitive.items()}
        return DictStateFn(new_dict, coeff=self.coeff, is_measurement=self.is_measurement)

    def tensor(self, other: OperatorBase) -> OperatorBase:
        # Both dicts
        if isinstance(other, DictStateFn):
            new_dict = {
                k1 + k2: v1 * v2
                for (
                    (
                        k1,
                        v1,
                    ),
                    (k2, v2),
                ) in itertools.product(self.primitive.items(), other.primitive.items())
            }
            return StateFn(
                new_dict, coeff=self.coeff * other.coeff, is_measurement=self.is_measurement
            )
        # pylint: disable=cyclic-import
        from ..list_ops.tensored_op import TensoredOp

        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        OperatorBase._check_massive("to_density_matrix", True, self.num_qubits, massive)
        states = int(2**self.num_qubits)
        return self.to_matrix(massive=massive) * np.eye(states) * self.coeff

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        OperatorBase._check_massive("to_matrix", False, self.num_qubits, massive)
        states = int(2**self.num_qubits)
        probs = np.zeros(states) + 0.0j
        for k, v in self.primitive.items():
            probs[int(k, 2)] = v
        vec = probs * self.coeff

        # Reshape for measurements so np.dot still works for composition.
        return vec if not self.is_measurement else vec.reshape(1, -1)

    def to_spmatrix(self) -> sparse.spmatrix:
        """Same as to_matrix, but returns csr sparse matrix.

        Returns:
            CSR sparse matrix representation of the State function.

        Raises:
            ValueError: invalid parameters.
        """

        indices = [int(v, 2) for v in self.primitive.keys()]
        vals = np.array(list(self.primitive.values())) * self.coeff
        spvec = sparse.csr_matrix(
            (vals, (np.zeros(len(indices), dtype=int), indices)), shape=(1, 2**self.num_qubits)
        )
        return spvec if not self.is_measurement else spvec.transpose()

    def to_spmatrix_op(self) -> OperatorBase:
        """Convert this state function to a ``SparseVectorStateFn``."""
        from .sparse_vector_state_fn import SparseVectorStateFn

        return SparseVectorStateFn(self.to_spmatrix(), self.coeff, self.is_measurement)

    def to_circuit_op(self) -> OperatorBase:
        """Convert this state function to a ``CircuitStateFn``."""
        from .circuit_state_fn import CircuitStateFn

        csfn = CircuitStateFn.from_dict(self.primitive) * self.coeff
        return csfn.adjoint() if self.is_measurement else csfn

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format(
                "DictStateFn" if not self.is_measurement else "DictMeasurement", prim_str
            )
        else:
            return "{}({}) * {}".format(
                "DictStateFn" if not self.is_measurement else "DictMeasurement",
                prim_str,
                self.coeff,
            )

    # pylint: disable=too-many-return-statements
    def eval(
        self,
        front: Optional[
            Union[str, Dict[str, complex], np.ndarray, OperatorBase, Statevector]
        ] = None,
    ) -> Union[OperatorBase, complex]:
        if front is None:
            sparse_vector_state_fn = self.to_spmatrix_op().eval()
            return sparse_vector_state_fn

        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError(
                "Cannot compute overlap with StateFn or Operator if not Measurement. Try taking "
                "sf.adjoint() first to convert to measurement."
            )

        if isinstance(front, ListOp) and front.distributive:
            return front.combo_fn(
                [self.eval(front.coeff * front_elem) for front_elem in front.oplist]
            )

        # For now, always do this. If it's not performant, we can be more granular.
        if not isinstance(front, OperatorBase):
            front = StateFn(front)

        # pylint: disable=cyclic-import
        from ..operator_globals import EVAL_SIG_DIGITS

        # If the primitive is a lookup of bitstrings,
        # we define all missing strings to have a function value of
        # zero.
        if isinstance(front, DictStateFn):
            # If self is come from operator, it should be expanded as
            # <self|front> = <front| self | front>.
            front_coeff = (
                front.coeff * front.coeff.conjugate() if self.from_operator else front.coeff
            )
            return np.round(
                cast(
                    float,
                    sum(v * front.primitive.get(b, 0) for (b, v) in self.primitive.items())
                    * self.coeff
                    * front_coeff,
                ),
                decimals=EVAL_SIG_DIGITS,
            )

        # All remaining possibilities only apply when self.is_measurement is True

        if isinstance(front, VectorStateFn):
            # TODO does it need to be this way for measurement?
            # return sum([v * front.primitive.data[int(b, 2)] *
            # np.conj(front.primitive.data[int(b, 2)])
            return np.round(
                cast(
                    float,
                    sum(v * front.primitive.data[int(b, 2)] for (b, v) in self.primitive.items())
                    * self.coeff,
                ),
                decimals=EVAL_SIG_DIGITS,
            )

        from .circuit_state_fn import CircuitStateFn

        if isinstance(front, CircuitStateFn):
            # Don't reimplement logic from CircuitStateFn
            self_adjoint = cast(DictStateFn, self.adjoint())
            return np.conj(front.adjoint().eval(self_adjoint.primitive)) * self.coeff

        from .operator_state_fn import OperatorStateFn

        if isinstance(front, OperatorStateFn):
            return cast(Union[OperatorBase, complex], front.adjoint().eval(self.adjoint()))

        # All other OperatorBases go here
        self_adjoint = cast(DictStateFn, self.adjoint())
        adjointed_eval = cast(OperatorBase, front.adjoint().eval(self_adjoint.primitive))
        return adjointed_eval.adjoint() * self.coeff

    def sample(
        self, shots: int = 1024, massive: bool = False, reverse_endianness: bool = False
    ) -> Dict[str, float]:
        probs = np.square(np.abs(np.array(list(self.primitive.values()))))
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            unique, counts = np.unique(
                algorithm_globals.random.choice(
                    list(self.primitive.keys()), size=shots, p=(probs / sum(probs))
                ),
                return_counts=True,
            )
        counts = dict(zip(unique, counts))
        if reverse_endianness:
            scaled_dict = {bstr[::-1]: (prob / shots) for (bstr, prob) in counts.items()}
        else:
            scaled_dict = {bstr: (prob / shots) for (bstr, prob) in counts.items()}
        return dict(sorted(scaled_dict.items(), key=lambda x: x[1], reverse=True))
