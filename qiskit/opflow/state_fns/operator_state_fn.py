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

""" OperatorStateFn Class """

from typing import List, Optional, Set, Union, cast

import numpy as np

from qiskit.circuit import ParameterExpression
from qiskit.opflow.list_ops.list_op import ListOp
from qiskit.opflow.list_ops.summed_op import SummedOp
from qiskit.opflow.list_ops.tensored_op import TensoredOp
from qiskit.opflow.operator_base import OperatorBase
from qiskit.opflow.primitive_ops.matrix_op import MatrixOp
from qiskit.opflow.primitive_ops.pauli_sum_op import PauliSumOp
from qiskit.opflow.state_fns.state_fn import StateFn
from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.quantum_info import Statevector


class OperatorStateFn(StateFn):
    r"""
    A class for state functions and measurements which are defined by a density Operator,
    stored using an ``OperatorBase``.
    """
    primitive: OperatorBase

    # TODO allow normalization somehow?
    def __init__(
        self,
        primitive: OperatorBase,
        coeff: Union[complex, ParameterExpression] = 1.0,
        is_measurement: bool = False,
    ) -> None:
        """
        Args:
            primitive: The ``OperatorBase`` which defines the behavior of the underlying State
                function.
            coeff: A coefficient by which to multiply the state function
            is_measurement: Whether the StateFn is a measurement operator
        """

        super().__init__(primitive, coeff=coeff, is_measurement=is_measurement)

    def primitive_strings(self) -> Set[str]:
        return self.primitive.primitive_strings()

    @property
    def num_qubits(self) -> int:
        return self.primitive.num_qubits

    def add(self, other: OperatorBase) -> Union["OperatorStateFn", SummedOp]:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                "Sum over statefns with different numbers of qubits, {} and {}, is not well "
                "defined".format(self.num_qubits, other.num_qubits)
            )

        # Right now doesn't make sense to add a StateFn to a Measurement
        if isinstance(other, OperatorStateFn) and self.is_measurement == other.is_measurement:
            if isinstance(other.primitive, OperatorBase) and self.primitive == other.primitive:
                return OperatorStateFn(
                    self.primitive,
                    coeff=self.coeff + other.coeff,
                    is_measurement=self.is_measurement,
                )
            # Covers Statevector and custom.
            elif isinstance(other, OperatorStateFn):
                # Also assumes scalar multiplication is available
                return OperatorStateFn(
                    (self.coeff * self.primitive).add(other.primitive * other.coeff),
                    is_measurement=self._is_measurement,
                )

        return SummedOp([self, other])

    def adjoint(self) -> "OperatorStateFn":
        return OperatorStateFn(
            self.primitive.adjoint(),
            coeff=self.coeff.conjugate(),
            is_measurement=(not self.is_measurement),
        )

    def _expand_dim(self, num_qubits: int) -> "OperatorStateFn":
        return OperatorStateFn(
            self.primitive._expand_dim(num_qubits),
            coeff=self.coeff,
            is_measurement=self.is_measurement,
        )

    def permute(self, permutation: List[int]) -> "OperatorStateFn":
        return OperatorStateFn(
            self.primitive.permute(permutation),
            coeff=self.coeff,
            is_measurement=self.is_measurement,
        )

    def tensor(self, other: OperatorBase) -> Union["OperatorStateFn", TensoredOp]:
        if isinstance(other, OperatorStateFn):
            return OperatorStateFn(
                self.primitive.tensor(other.primitive),
                coeff=self.coeff * other.coeff,
                is_measurement=self.is_measurement,
            )

        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        """Return numpy matrix of density operator, warn if more than 16 qubits
        to force the user to set
        massive=True if they want such a large matrix. Generally big methods like
        this should require the use of a
        converter, but in this case a convenience method for quick hacking and
        access to classical tools is
        appropriate."""
        OperatorBase._check_massive("to_density_matrix", True, self.num_qubits, massive)
        return self.primitive.to_matrix() * self.coeff

    def to_matrix_op(self, massive: bool = False) -> "OperatorStateFn":
        """Return a MatrixOp for this operator."""
        return OperatorStateFn(
            self.primitive.to_matrix_op(massive=massive) * self.coeff,
            is_measurement=self.is_measurement,
        )

    def to_matrix(self, massive: bool = False) -> np.ndarray:
        r"""
        Note: this does not return a density matrix, it returns a classical matrix
        containing the quantum or classical vector representing the evaluation of the state
        function on each binary basis state. Do not assume this is is a normalized quantum or
        classical probability vector. If we allowed this to return a density matrix,
        then we would need to change the definition of composition to be ~Op @ StateFn @ Op for
        those cases, whereas by this methodology we can ensure that composition always means Op
        @ StateFn.

        Return numpy vector of state vector, warn if more than 16 qubits to force the user to set
        massive=True if they want such a large vector.

        Args:
            massive: Whether to allow large conversions, e.g. creating a matrix representing
                over 16 qubits.

        Returns:
            np.ndarray: Vector of state vector

        Raises:
            ValueError: Invalid parameters.
        """
        OperatorBase._check_massive("to_matrix", False, self.num_qubits, massive)
        # Operator - return diagonal (real values, not complex),
        # not rank 1 decomposition (statevector)!
        mat = self.primitive.to_matrix(massive=massive)
        # TODO change to weighted sum of eigenvectors' StateFns?

        # ListOp primitives can return lists of matrices (or trees for nested ListOps),
        # so we need to recurse over the
        # possible tree.
        def diag_over_tree(op):
            if isinstance(op, list):
                return [diag_over_tree(o) for o in op]
            else:
                vec = np.diag(op) * self.coeff
                # Reshape for measurements so np.dot still works for composition.
                return vec if not self.is_measurement else vec.reshape(1, -1)

        return diag_over_tree(mat)

    def to_circuit_op(self):
        r"""Return ``StateFnCircuit`` corresponding to this StateFn. Ignore for now because this is
        undefined. TODO maybe call to_pauli_op and diagonalize here, but that could be very
        inefficient, e.g. splitting one Stabilizer measurement into hundreds of 1 qubit Paulis."""
        raise NotImplementedError

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format(
                "OperatorStateFn" if not self.is_measurement else "OperatorMeasurement", prim_str
            )
        else:
            return "{}({}) * {}".format(
                "OperatorStateFn" if not self.is_measurement else "OperatorMeasurement",
                prim_str,
                self.coeff,
            )

    def eval(
        self, front: Optional[Union[str, dict, np.ndarray, OperatorBase, Statevector]] = None
    ) -> Union[OperatorBase, complex]:
        if front is None:
            matrix = cast(MatrixOp, self.primitive.to_matrix_op()).primitive.data
            # pylint: disable=cyclic-import
            from .vector_state_fn import VectorStateFn

            return VectorStateFn(matrix[0, :])

        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError(
                "Cannot compute overlap with StateFn or Operator if not Measurement. Try taking "
                "sf.adjoint() first to convert to measurement."
            )

        if not isinstance(front, OperatorBase):
            front = StateFn(front)

        if isinstance(self.primitive, ListOp) and self.primitive.distributive:
            evals = [
                OperatorStateFn(op, is_measurement=self.is_measurement).eval(front)
                for op in self.primitive.oplist
            ]
            result = self.primitive.combo_fn(evals)
            if isinstance(result, list):
                multiplied = self.primitive.coeff * self.coeff * np.array(result)
                return multiplied.tolist()
            return result * self.coeff * self.primitive.coeff

        # pylint: disable=cyclic-import
        from .vector_state_fn import VectorStateFn

        if isinstance(self.primitive, PauliSumOp) and isinstance(front, VectorStateFn):
            return (
                front.primitive.expectation_value(self.primitive.primitive)
                * self.coeff
                * front.coeff
            )

        # Need an ListOp-specific carve-out here to make sure measurement over a ListOp doesn't
        # produce two-dimensional ListOp from composing from both sides of primitive.
        # Can't use isinstance because this would include subclasses.
        # pylint: disable=unidiomatic-typecheck
        if isinstance(front, ListOp) and type(front) == ListOp:
            return front.combo_fn(
                [self.eval(front.coeff * front_elem) for front_elem in front.oplist]
            )

        # If we evaluate against a circuit, evaluate it to a vector so we
        # make sure to only do the expensive circuit simulation once
        if isinstance(front, CircuitStateFn):
            front = front.eval()

        return front.adjoint().eval(cast(OperatorBase, self.primitive.eval(front))) * self.coeff

    def sample(self, shots: int = 1024, massive: bool = False, reverse_endianness: bool = False):
        raise NotImplementedError
