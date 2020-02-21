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
IdentityOp class
"""

import copy
import numpy as np
from numbers import Number

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator


class IdentityOp(BaseOperator):
    """Identity operator class.

    This is a symbolic representation of an identity operator on multiple
    subsystems. It may be used to initialize a symbolic identity and then be
    implicitly converted to other kinds of operator subclasses by using the
    :meth:`compose`, :meth:`dot`, :meth:`tensor`, :meth:`expand` methods.
    """

    def __init__(self, dims, coeff=None):
        """Initialize an operator object.

        Args:
            dims (int or tuple): subsystem dimensions.
            coeffs (complex or None): an optional scalar coefficient for
                the identity operator (Default: None).

        Raises:
            QiskitError: If the optional coefficient is invalid.
        """
        input_dims = self._automatic_dims(dims, np.product(dims))
        if coeff is not None and not isinstance(coeff, Number):
            raise QiskitError("Coeff {} must be None or a number.".format(coeff))
        self._coeff = coeff
        super().__init__(input_dims, input_dims)

    def __repr__(self):
        if self._coeff is None:
            return 'IdentityOp({})'.format(self._input_dims)
        return 'IdentityOp({}, coeff={})'.format(
            self._input_dims, self._coeff)

    @property
    def coeff(self):
        """Return the coefficient"""
        return self._coeff

    def conjugate(self):
        """Return the conjugate of the operator."""
        if self._coeff is None:
            return self
        ret = copy.copy(self)
        ret._coeff = np.conjugate(self._coeff)
        return ret

    def transpose(self):
        """Return the transpose of the operator."""
        return self

    def is_unitary(self, atol=None, rtol=None):
        """Return True if operator is a unitary matrix."""
        if self._coeff is None:
            return True
        if atol is None:
            atol = self._atol
        if rtol is None:
            rtol = self._rtol
        return np.isclose(np.abs(self._coeff), 1, atol=atol, rtol=rtol)

    def to_matrix(self):
        """Convert to a Numpy identity matrix."""
        dim, _ = self.dim
        iden = np.eye(dim, dtype=complex)
        if self._coeff is None:
            return iden
        return self._coeff * iden

    def to_operator(self):
        """Convert to an Operator object."""
        return Operator(self.to_matrix(),
                        input_dims=self._input_dims,
                        output_dims=self._output_dims)

    def tensor(self, other):
        """Return the tensor product operator self ⊗ other.

        Args:
            other (BaseOperator): an operator object.

        Returns:
            IdentityOp: if other is an IdentityOp.
            BaseOperator: if other is not an IdentityOp.
        """
        if not isinstance(other, BaseOperator):
            other = Operator(other)
        if isinstance(other, IdentityOp):
            if self._coeff is None:
                coeff = other.data
            elif other.data is None:
                coeff = self._coeff
            else:
                coeff = self._coeff * other.data
            dims = other._input_dims + self._input_dims
            return IdentityOp(dims, coeff=coeff)
        return other.expand(self)

    def expand(self, other):
        """Return the tensor product operator other ⊗ self.

        Args:
            other (IdentityOp or Operator): an operator object.

        Returns:
            IdentityOp: if other is an IdentityOp.
            BaseOperator: if other is not an IdentityOp.
        """
        if not isinstance(other, BaseOperator):
            other = Operator(other)
        if isinstance(other, IdentityOp):
            if self._coeff is None:
                coeff = other.data
            elif other.data is None:
                coeff = self._coeff
            else:
                coeff = self._coeff * other.data
            dims = self._input_dims + other._input_dims
            return IdentityOp(dims, coeff=coeff)
        return other.tensor(self)

    def compose(self, other, qargs=None, front=False):
        """Return the composed operator.

        Args:
            other (BaseOperator): an operator object.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].
            front (bool): If True compose using right operator multiplication,
                          instead of left multiplication [default: False].

        Returns:
            BaseOperator: The operator self @ other.

        Raises:
            QiskitError: if other has incompatible dimensions for specified
                         subsystems.

        Additional Information:
            Composition (``@``) is defined as `left` matrix multiplication for
            matrix operators. That is that ``A @ B`` is equal to ``B * A``.
            Setting ``front=True`` returns `right` matrix multiplication
            ``A * B`` and is equivalent to the :meth:`dot` method.
        """
        if not isinstance(other, BaseOperator):
            other = Operator(other)
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        input_dims, output_dims = self._get_compose_dims(other, qargs, front)

        # If other is also an IdentityOp we only need to possibly
        # update the coefficient and dimensions
        if isinstance(other, IdentityOp):
            if self._coeff is None:
                coeff = other._coeff
            elif other._coeff is None:
                coeff = self._coeff
            else:
                coeff = self._coeff * other._coeff
            return IdentityOp(input_dims, coeff=coeff)

        # If we are composing on the full system we return the
        # other operator with reshaped dimensions
        if qargs is None:
            ret = other.reshape(input_dims, output_dims)
            if self._coeff is None or self._coeff == 1:
                return ret
            return self._coeff * ret
        # Otherwise compose using other operators method
        # Note that in this case that operator must know how to initalize
        # from an IdentityOp either using its to_operator method or having
        # a specific initialization case.
        return other.__class__(self).compose(
            other, qargs=qargs, front=not front)

    def power(self, n):
        """Return the compose of a operator with itself n times.

        Args:
            n (int): the number of times to compose with self (n>0).

        Returns:
            BaseOperator: the n-times composed operator.

        Raises:
            QiskitError: if the input and output dimensions of the operator
            are not equal, or the power is not a positive integer.
        """
        if self._coeff is None:
            return self
        coeff = self._coeff ** n
        return IdentityOp(self._input_dims, coeff=coeff)

    def _add(self, other):
        """Return the operator self + other.

        Args:
            other (BaseOperator): an operator object.

        Returns:
            IdentityOp: if other is an IdentityOp.
            BaseOperator: if other is not an IdentityOp.

        Raises:
            QiskitError: if other has incompatible dimensions.
        """
        if not isinstance(other, BaseOperator):
            other = Operator(other)
        self._validate_add_dims(other)
        if isinstance(other, IdentityOp):
            coeff1 = 1 if self._coeff is None else self._coeff
            coeff2 = 1 if other.data is None else other.data
            return IdentityOp(self._input_dims, coeff=coeff1+coeff2)
        return other._add(self).reshape(self._input_dims, self._output_dims)

    def _multiply(self, other):
        """Return the IdentityOp other * self.

        Args:
            other (complex): a complex number.

        Returns:
            IdentityOp: the scaled identity operator other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        if other == 1:
            return self
        coeff = other if self._coeff is None else other * self._coeff
        return IdentityOp(self._input_dim, coeff=coeff)
