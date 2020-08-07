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

""" SummedOp Class """

from typing import List, Union, cast
import warnings

import numpy as np

from qiskit.circuit import ParameterExpression
from .list_op import ListOp
from ..legacy.base_operator import LegacyBaseOperator
from ..legacy.weighted_pauli_operator import WeightedPauliOperator
from ..operator_base import OperatorBase
from ..primitive_ops.primitive_op import PrimitiveOp


class SummedOp(ListOp):
    """ A class for lazily representing sums of Operators. Often Operators cannot be
    efficiently added to one another, but may be manipulated further so that they can be
    later. This class holds logic to indicate that the Operators in ``oplist`` are meant to
    be added together, and therefore if they reach a point in which they can be, such as after
    evaluation or conversion to matrices, they can be reduced by addition. """

    def __init__(self,
                 oplist: List[OperatorBase],
                 coeff: Union[int, float, complex, ParameterExpression] = 1.0,
                 abelian: bool = False) -> None:
        """
        Args:
            oplist: The Operators being summed.
            coeff: A coefficient multiplying the operator
            abelian: Indicates whether the Operators in ``oplist`` are known to mutually commute.
        """
        super().__init__(oplist,
                         combo_fn=lambda x: np.sum(x, axis=0),
                         coeff=coeff,
                         abelian=abelian)

    @property
    def num_qubits(self) -> int:
        return self.oplist[0].num_qubits

    @property
    def distributive(self) -> bool:
        return True

    def add(self, other: OperatorBase) -> OperatorBase:
        """Return Operator addition of ``self`` and ``other``, overloaded by ``+``.

        Note:
            This appends ``other`` to ``self.oplist`` without checking ``other`` is already
            included or not. If you want to simplify them, please use :meth:`simplify`.

        Args:
            other: An ``OperatorBase`` with the same number of qubits as self, and in the same
                'Operator', 'State function', or 'Measurement' category as self (i.e. the same type
                of underlying function).

        Returns:
            A ``SummedOp`` equivalent to the sum of self and other.
        """
        self_new_ops = self.oplist if self.coeff == 1 \
            else [op.mul(self.coeff) for op in self.oplist]
        if isinstance(other, SummedOp):
            other_new_ops = other.oplist if other.coeff == 1 \
                else [op.mul(other.coeff) for op in other.oplist]
        else:
            other_new_ops = [other]
        return SummedOp(self_new_ops + other_new_ops)

    def collapse_summands(self) -> 'SummedOp':
        """Return Operator by simplifying duplicate operators.

        E.g., ``SummedOp([2 * X ^ Y, X ^ Y]).collapse_summands() -> SummedOp([3 * X ^ Y])``.

        Returns:
            A simplified ``SummedOp`` equivalent to self.
        """
        oplist = []  # type: List[OperatorBase]
        coeffs = []  # type: List[Union[int, float, complex, ParameterExpression]]
        for op in self.oplist:
            if isinstance(op, PrimitiveOp):
                new_op = PrimitiveOp(op.primitive)
                new_coeff = op.coeff * self.coeff
                if new_op in oplist:
                    index = oplist.index(new_op)
                    coeffs[index] += new_coeff
                else:
                    oplist.append(new_op)
                    coeffs.append(new_coeff)
            else:
                if op in oplist:
                    index = oplist.index(op)
                    coeffs[index] += self.coeff
                else:
                    oplist.append(op)
                    coeffs.append(self.coeff)
        return SummedOp([op * coeff for op, coeff in zip(oplist, coeffs)])  # type: ignore

    # TODO be smarter about the fact that any two ops in oplist could be evaluated for sum.
    def reduce(self) -> OperatorBase:
        """Try collapsing list or trees of sums.

        Tries to sum up duplicate operators and reduces the operators
        in the sum.

        Returns:
            A collapsed version of self, if possible.
        """
        # reduce constituents
        reduced_ops = sum(op.reduce() for op in self.oplist) * self.coeff

        # group duplicate operators
        if isinstance(reduced_ops, SummedOp):
            reduced_ops = reduced_ops.collapse_summands()

        if isinstance(reduced_ops, SummedOp) and len(reduced_ops.oplist) == 1:
            return reduced_ops.oplist[0]
        else:
            return cast(OperatorBase, reduced_ops)

    def to_matrix_op(self, massive: bool = False) -> OperatorBase:
        """ Returns an equivalent Operator composed of only NumPy-based primitives, such as
        ``MatrixOp`` and ``VectorStateFn``. """
        accum = self.oplist[0].to_matrix_op(massive=massive)  # type: ignore
        for i in range(1, len(self.oplist)):
            accum += self.oplist[i].to_matrix_op(massive=massive)  # type: ignore

        return accum * self.coeff

    def to_legacy_op(self, massive: bool = False) -> LegacyBaseOperator:
        # We do this recursively in case there are SummedOps of PauliOps in oplist.
        legacy_ops = [op.to_legacy_op(massive=massive) for op in self.oplist]

        if not all(isinstance(op, WeightedPauliOperator) for op in legacy_ops):
            # If any Operators in oplist cannot be represented by Legacy Operators, the error
            # will be raised in the offending matrix-converted result (e.g. StateFn or ListOp)
            return self.to_matrix_op(massive=massive).to_legacy_op(massive=massive)

        if isinstance(self.coeff, ParameterExpression):
            try:
                coeff = float(self.coeff)
            except TypeError:
                raise TypeError('Cannot convert Operator with unbound parameter {} to Legacy '
                                'Operator'.format(self.coeff))
        else:
            coeff = cast(float, self.coeff)

        return self.combo_fn(legacy_ops) * coeff

    def print_details(self):
        """
        Print out the operator in details.
        Returns:
            str: a formatted string describes the operator.
        """
        warnings.warn("print_details() is deprecated and will be removed in "
                      "a future release. Instead you can use .to_legacy_op() "
                      "and call print_details() on it's output",
                      DeprecationWarning)
        ret = self.to_legacy_op().print_details()
        return ret

    def equals(self, other: OperatorBase) -> bool:
        """Check if other is equal to self.

        Note:
            This is not a mathematical check for equality.
            If ``self`` and ``other`` implement the same operation but differ
            in the representation (e.g. different type of summands)
            ``equals`` will evaluate to ``False``.

        Args:
            other: The other operator to check for equality.

        Returns:
            True, if other and self are equal, otherwise False.

        Examples:
            >>> from qiskit.aqua.operators import X, Z
            >>> 2 * X == X + X
            True
            >>> X + Z == Z + X
            True
        """
        self_reduced, other_reduced = self.reduce(), other.reduce()
        if not isinstance(other_reduced, type(self_reduced)):
            return False

        # check if reduced op is still a SummedOp
        if not isinstance(self_reduced, SummedOp):
            return self_reduced == other_reduced

        self_reduced = cast(SummedOp, self_reduced)
        other_reduced = cast(SummedOp, other_reduced)
        if len(self_reduced.oplist) != len(other_reduced.oplist):
            return False

        # absorb coeffs into the operators
        if self_reduced.coeff != 1:
            self_reduced = SummedOp(
                [op * self_reduced.coeff for op in self_reduced.oplist])  # type: ignore
        if other_reduced.coeff != 1:
            other_reduced = SummedOp(
                [op * other_reduced.coeff for op in other_reduced.oplist])  # type: ignore

        # compare independent of order
        return set(self_reduced) == set(other_reduced)
