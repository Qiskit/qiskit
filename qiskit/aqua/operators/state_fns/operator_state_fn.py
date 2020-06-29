# -*- coding: utf-8 -*-

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

from typing import Union, Set
import numpy as np

from qiskit.circuit import ParameterExpression

from ..operator_base import OperatorBase
from .state_fn import StateFn
from ..list_ops.list_op import ListOp
from ..list_ops.summed_op import SummedOp


# pylint: disable=invalid-name

class OperatorStateFn(StateFn):
    r"""
    A class for state functions and measurements which are defined by a density Operator,
    stored using an ``OperatorBase``.
    """

    # TODO allow normalization somehow?
    def __init__(self,
                 primitive: Union[OperatorBase] = None,
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 is_measurement: bool = False) -> None:
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

    def add(self, other: OperatorBase) -> OperatorBase:
        if not self.num_qubits == other.num_qubits:
            raise ValueError(
                'Sum over statefns with different numbers of qubits, {} and {}, is not well '
                'defined'.format(self.num_qubits, other.num_qubits))

        # Right now doesn't make sense to add a StateFn to a Measurement
        if isinstance(other, OperatorStateFn) and self.is_measurement == other.is_measurement:
            if isinstance(self.primitive.primitive, type(other.primitive.primitive)) and \
                    self.primitive == other.primitive:
                return StateFn(self.primitive,
                               coeff=self.coeff + other.coeff,
                               is_measurement=self.is_measurement)
            # Covers MatrixOperator, Statevector and custom.
            elif isinstance(other, OperatorStateFn):
                # Also assumes scalar multiplication is available
                return OperatorStateFn(
                    (self.coeff * self.primitive).add(other.primitive * other.coeff),
                    is_measurement=self._is_measurement)

        return SummedOp([self, other])

    def adjoint(self) -> OperatorBase:
        return OperatorStateFn(self.primitive.adjoint(),
                               coeff=np.conj(self.coeff),
                               is_measurement=(not self.is_measurement))

    def tensor(self, other: OperatorBase) -> OperatorBase:
        if isinstance(other, OperatorStateFn):
            return StateFn(self.primitive.tensor(other.primitive),
                           coeff=self.coeff * other.coeff,
                           is_measurement=self.is_measurement)
        # pylint: disable=cyclic-import,import-outside-toplevel
        from .. import TensoredOp
        return TensoredOp([self, other])

    def to_density_matrix(self, massive: bool = False) -> np.ndarray:
        """ Return numpy matrix of density operator, warn if more than 16 qubits
        to force the user to set
        massive=True if they want such a large matrix. Generally big methods like
        this should require the use of a
        converter, but in this case a convenience method for quick hacking and
        access to classical tools is
        appropriate. """

        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_matrix will return an exponentially large matrix,'
                ' in this case {0}x{0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        return self.primitive.to_matrix() * self.coeff

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        """ Return a MatrixOp for this operator. """
        return OperatorStateFn(self.primitive.to_matrix_op(massive=massive) * self.coeff,
                               is_measurement=self.is_measurement)

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

        if self.num_qubits > 16 and not massive:
            raise ValueError(
                'to_vector will return an exponentially large vector, in this case {0} elements.'
                ' Set massive=True if you want to proceed.'.format(2 ** self.num_qubits))

        # Operator - return diagonal (real values, not complex),
        # not rank 1 decomposition (statevector)!
        mat = self.primitive.to_matrix()
        # TODO change to weighted sum of eigenvectors' StateFns?

        # ListOp primitives can return lists of matrices (or trees for nested ListOps),
        # so we need to recurse over the
        # possible tree.
        def diag_over_tree(t):
            if isinstance(t, list):
                return [diag_over_tree(o) for o in t]
            else:
                vec = np.diag(t) * self.coeff
                # Reshape for measurements so np.dot still works for composition.
                return vec if not self.is_measurement else vec.reshape(1, -1)

        return diag_over_tree(mat)

    def to_circuit_op(self) -> OperatorBase:
        r""" Return ``StateFnCircuit`` corresponding to this StateFn. Ignore for now because this is
        undefined. TODO maybe call to_pauli_op and diagonalize here, but that could be very
        inefficient, e.g. splitting one Stabilizer measurement into hundreds of 1 qubit Paulis."""
        raise NotImplementedError

    def __str__(self) -> str:
        prim_str = str(self.primitive)
        if self.coeff == 1.0:
            return "{}({})".format('OperatorStateFn' if not self.is_measurement
                                   else 'OperatorMeasurement', prim_str)
        else:
            return "{}({}) * {}".format(
                'OperatorStateFn' if not self.is_measurement else 'OperatorMeasurement',
                prim_str,
                self.coeff)

    # pylint: disable=too-many-return-statements
    def eval(self,
             front: Union[str, dict, np.ndarray,
                          OperatorBase] = None) -> Union[OperatorBase, float, complex]:
        if not self.is_measurement and isinstance(front, OperatorBase):
            raise ValueError(
                'Cannot compute overlap with StateFn or Operator if not Measurement. Try taking '
                'sf.adjoint() first to convert to measurement.')

        if not isinstance(front, OperatorBase):
            front = StateFn(front)

        if isinstance(self.primitive, ListOp) and self.primitive.distributive:
            coeff = self.coeff * self.primitive.coeff
            evals = [OperatorStateFn(op, coeff=coeff, is_measurement=self.is_measurement).eval(
                front) for op in self.primitive.oplist]
            return self.primitive.combo_fn(evals)

        # Need an ListOp-specific carve-out here to make sure measurement over a ListOp doesn't
        # produce two-dimensional ListOp from composing from both sides of primitive.
        # Can't use isinstance because this would include subclasses.
        # pylint: disable=unidiomatic-typecheck
        if type(front) == ListOp:
            return front.combo_fn([self.eval(front.coeff * front_elem)  # type: ignore
                                   for front_elem in front.oplist])  # type: ignore

        return front.adjoint().eval(self.primitive.eval(front)) * self.coeff  # type: ignore

    def sample(self,
               shots: int = 1024,
               massive: bool = False,
               reverse_endianness: bool = False) -> dict:
        raise NotImplementedError
