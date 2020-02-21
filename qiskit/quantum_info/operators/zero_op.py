# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
ZeroOp class
"""

import numpy as np
from numbers import Number

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator


class ZeroOp(BaseOperator):
    """Zero operator class.

    This is a symbolic representation of an empty (zero matrix) operator on
    multiple subsystems. It may be used to initialize an empty operator and
    then be implicitly converted to other classes of operators by using the
    :meth:`add`, :meth:`subtract` methods.
    """

    def __init__(self, input_dims, output_dims=None):
        """Initialize an operator object."""
        if isinstance(input_dims, BaseOperator):
            output_dims = input_dims.output_dims()
            input_dims = input_dims.input_dims()
        else:
            if output_dims is None:
                output_dims = input_dims
                input_dims = self._automatic_dims(
                                input_dims, np.product(input_dims))
                output_dims = self._automatic_dims(
                                output_dims, np.product(output_dims))
        super().__init__(input_dims, input_dims)

    def __repr__(self):
        return 'ZeroOp(input_dims={}, output_dims={})'.format(
            self._input_dims, self._output_dims)

    def conjugate(self):
        """Return the conjugate of the operator."""
        return self

    def transpose(self):
        """Return the transpose of the operator."""
        return self

    def to_matrix(self):
        """Convert to a Numpy zero matrix."""
        din, dout = self.dim
        return np.zeros((dout, din), dtype=complex)

    def to_operator(self):
        """Convert to an Operator object."""
        return Operator(self.to_matrix(),
                        input_dims=self._input_dims,
                        output_dims=self._output_dims)

    def tensor(self, other):
        """Return the tensor product operator self ⊗ other.

        Tensoring a ZeroOp with another operator will return a
        ZeroOp with combined subsystem dimensions.

        Args:
            other (BaseOperator): an operator object.

        Returns:
            ZeroOp: a zero operator of the combined dimension.
        """
        if not isinstance(other, BaseOperator):
            other = Operator(other)
        input_dims = other._input_dims + self._input_dims
        output_dims = other._output_dim + self._input_dims
        return ZeroOp(input_dims, output_dims)

    def expand(self, other):
        """Return the tensor product operator other ⊗ self.

        Expanding a ZeroOp with another operator will return a
        ZeroOp with combined subsystem dimensions.

        Args:
            other (BaseOperator): an operator object.

        Returns:
            ZeroOp: a zero operator with the combined dimension.
        """
        if not isinstance(other, BaseOperator):
            other = Operator(other)
        input_dims = self._input_dims + other._input_dims
        output_dims = self._input_dims + other._output_dim
        return ZeroOp(input_dims, output_dims)

    def compose(self, other, qargs=None, front=False):
        """Return the composed operator.

        Composing another operator with a ZeroOp will always return a
        ZeroOp, though possibly with reshaped dimensions depending on
        the subsystem dimensions of the composed operator.

        Args:
            other (BaseOperator): an operator object.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].
            front (bool): If True compose using right operator multiplication,
                          instead of left multiplication [default: False].

        Returns:
            ZeroOp: The empty output operator.

        Raises:
            QiskitError: if other has incompatible dimensions for specified
                         subsystems.

        Additional Information:
            Composition (``@``) is defined as `left` matrix multiplication for
            matrix operators. That is that ``A @ B`` is equal to ``B * A``.
            Setting ``front=True`` returns `right` matrix multiplication
            ``A * B`` and is equivalent to the :meth:`dot` method.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if not isinstance(other, BaseOperator):
            other = Operator(other)
        input_dims, output_dims = self._get_compose_dims(other, qargs, front)
        return ZeroOp(input_dims, output_dims)

    def power(self, n):
        """Return the compose of a operator with itself n times.

        Args:
            n (int): the number of times to compose with self (n>0).

        Returns:
            ZeroOp: the composed operator.

        Raises:
            QiskitError: if the input and output dimensions of the operator
            are not equal, or the power is not a positive integer.
        """
        if self.input_dims() != self.output_dims():
            raise QiskitError("Cannot take power of non-square Operator.")
        return self

    def _add(self, other):
        """Return the operator self + other.

        Adding another operator to the ZeroOp will return the other
        operator with reshaped subsystem dimensions that match the
        ZeroOp subsystem dimensions.

        Args:
            other (BaseOperator): an operator object.

        Returns:
            BaseOperator: the other operator.

        Raises:
            QiskitError: if other has incompatible dimensions.
        """
        if not isinstance(other, BaseOperator):
            other = Operator(other)
        self._validate_add_dims(other)
        # Return a shallow copy with reshaped dimensions
        return other.reshape(self._input_dims, self._output_dims)

    def _multiply(self, other):
        """Return the ZeroOp.

        Args:
            other (complex): a complex number.

        Returns:
            ZeroOp: a zero operator.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        return self
