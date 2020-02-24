# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Abstract BaseOperator class.
"""

import copy
import warnings
from abc import ABC, abstractmethod

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT


class BaseOperator(ABC):
    """Abstract linear operator base class"""

    ATOL = ATOL_DEFAULT
    RTOL = RTOL_DEFAULT
    MAX_TOL = 1e-4

    def __init__(self, data, input_dims, output_dims):
        """Initialize an operator object."""
        self._data = data
        # Shape lists the dimension of each subsystem starting from
        # least significant through to most significant.
        self._input_dims = tuple(input_dims)
        self._output_dims = tuple(output_dims)
        # The total input and output dimensions are given by the product
        # of all subsystem dimension in the input_dims/output_dims.
        self._input_dim = np.product(input_dims)
        self._output_dim = np.product(output_dims)

    def __eq__(self, other):
        if (isinstance(other, self.__class__)
                and self.input_dims() == other.input_dims()
                and self.output_dims() == other.output_dims()):
            return np.allclose(
                self.data, other.data, rtol=self._rtol, atol=self._atol)
        return False

    @property
    def dim(self):
        """Return tuple (input_shape, output_shape)."""
        return self._input_dim, self._output_dim

    @property
    def data(self):
        """Return data."""
        return self._data

    @property
    def _atol(self):
        """The absolute tolerance parameter for float comparisons."""
        return self.__class__.ATOL

    @_atol.setter
    def _atol(self, atol):
        """Set the absolute tolerance parameter for float comparisons."""
        # NOTE: that this overrides the class value so applies to all
        # instances of the class.
        max_tol = self.__class__.MAX_TOL
        if atol < 0:
            raise QiskitError("Invalid atol: must be non-negative.")
        if atol > max_tol:
            raise QiskitError(
                "Invalid atol: must be less than {}.".format(max_tol))
        self.__class__.ATOL = atol

    @property
    def _rtol(self):
        """The relative tolerance parameter for float comparisons."""
        return self.__class__.RTOL

    @_rtol.setter
    def _rtol(self, rtol):
        """Set the relative tolerance parameter for float comparisons."""
        # NOTE: that this overrides the class value so applies to all
        # instances of the class.
        max_tol = self.__class__.MAX_TOL
        if rtol < 0:
            raise QiskitError("Invalid rtol: must be non-negative.")
        if rtol > max_tol:
            raise QiskitError(
                "Invalid rtol: must be less than {}.".format(max_tol))
        self.__class__.RTOL = rtol

    def reshape(self, input_dims=None, output_dims=None):
        """Return a shallow copy with reshaped input and output subsystem dimensions.

        Arg:
            input_dims (None or tuple): new subsystem input dimensions.
                If None the original input dims will be preserved
                [Default: None].
            output_dims (None or tuple): new subsystem output dimensions.
                If None the original output dims will be preserved
                [Default: None].

        Returns:
            BaseOperator: returns self with reshaped input and output dimensions.

        Raises:
            QiskitError: if combined size of all subsystem input dimension or
            subsystem output dimensions is not constant.
        """
        clone = copy.copy(self)
        if output_dims is None and input_dims is None:
            return clone
        if input_dims is not None:
            if np.product(input_dims) != self._input_dim:
                raise QiskitError(
                    "Reshaped input_dims ({}) are incompatible with combined"
                    " input dimension ({}).".format(input_dims, self._input_dim))
            clone._input_dims = tuple(input_dims)
        if output_dims is not None:
            if np.product(output_dims) != self._output_dim:
                raise QiskitError(
                    "Reshaped output_dims ({}) are incompatible with combined"
                    " output dimension ({}).".format(output_dims, self._output_dim))
            clone._output_dims = tuple(output_dims)
        return clone

    def input_dims(self, qargs=None):
        """Return tuple of input dimension for specified subsystems."""
        if qargs is None:
            return self._input_dims
        return tuple(self._input_dims[i] for i in qargs)

    def output_dims(self, qargs=None):
        """Return tuple of output dimension for specified subsystems."""
        if qargs is None:
            return self._output_dims
        return tuple(self._output_dims[i] for i in qargs)

    def copy(self):
        """Make a deep copy of current operator."""
        return copy.deepcopy(self)

    def adjoint(self):
        """Return the adjoint of the operator."""
        return self.conjugate().transpose()

    @abstractmethod
    def conjugate(self):
        """Return the conjugate of the operator."""
        pass

    @abstractmethod
    def transpose(self):
        """Return the transpose of the operator."""
        pass

    @abstractmethod
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
            QiskitError: if other cannot be converted to an operator, or has
            incompatible dimensions for specified subsystems.

        Additional Information:
            Composition (``@``) is defined as `left` matrix multiplication for
            matrix operators. That is that ``A @ B`` is equal to ``B * A``.
            Setting ``front=True`` returns `right` matrix multiplication
            ``A * B`` and is equivalent to the :meth:`dot` method.
        """
        pass

    def dot(self, other, qargs=None):
        """Return the right multiplied operator self * other.

        Args:
            other (BaseOperator): an operator object.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].

        Returns:
            BaseOperator: The operator self * other.

        Raises:
            QiskitError: if other cannot be converted to an operator, or has
            incompatible dimensions for specified subsystems.
        """
        return self.compose(other, qargs=qargs, front=True)

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
        # NOTE: if a subclass can have negative or non-integer powers
        # this method should be overridden in that class.
        if not isinstance(n, (int, np.integer)) or n < 1:
            raise QiskitError("Can only power with positive integer powers.")
        if self._input_dim != self._output_dim:
            raise QiskitError("Can only power with input_dim = output_dim.")
        ret = self.copy()
        for _ in range(1, n):
            ret = ret.compose(self)
        return ret

    @abstractmethod
    def tensor(self, other):
        """Return the tensor product operator self ⊗ other.

        Args:
            other (BaseOperator): a operator subclass object.

        Returns:
            BaseOperator: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other is not an operator.
        """
        pass

    @abstractmethod
    def expand(self, other):
        """Return the tensor product operator other ⊗ self.

        Args:
            other (BaseOperator): an operator object.

        Returns:
            BaseOperator: the tensor product operator other ⊗ self.

        Raises:
            QiskitError: if other is not an operator.
        """
        pass

    def add(self, other):
        """Return the linear operator self + other.

        DEPRECATED: use `+` or `_add` instead.

        Args:
            other (BaseOperator): an operator object.

        Returns:
            BaseOperator: the operator self + other.
        """
        warnings.warn("`BaseOperator.add` method is deprecated, use the `+`"
                      " operator or `BaseOperator._add` instead",
                      DeprecationWarning)
        return self._add(other)

    def subtract(self, other):
        """Return the linear operator self - other.

        DEPRECATED: use `-` instead.

        Args:
            other (BaseOperator): an operator object.

        Returns:
            BaseOperator: the operator self - other.
        """
        warnings.warn("`BaseOperator.subtract` method is deprecated, use the `-`"
                      " operator or `BaseOperator._add(-other)` instead",
                      DeprecationWarning)
        return self._add(-other)

    def multiply(self, other):
        """Return the linear operator other * self.

        DEPRECATED: use `*` of `_multiply` instead.

        Args:
            other (complex): a complex number.

        Returns:
            BaseOperator: the linear operator other * self.

        Raises:
            NotImplementedError: if subclass does not support multiplication.
        """
        warnings.warn("`BaseOperator.multiply` method is deprecated, use the `*`"
                      " operator or `BaseOperator._multiply` instead",
                      DeprecationWarning)
        return self._multiply(other)

    def _add(self, other):
        """Return the linear operator self + other.

        Args:
            other (BaseOperator): an operator object.

        Returns:
            BaseOperator: the operator self + other.

        Raises:
            NotImplementedError: if subclass does not support addition.
        """
        raise NotImplementedError(
            "{} does not support addition".format(type(self)))

    def _multiply(self, other):
        """Return the linear operator other * self.

        Args:
            other (complex): a complex number.

        Returns:
            BaseOperator: the linear operator other * self.

        Raises:
            NotImplementedError: if subclass does not support multiplication.
        """
        raise NotImplementedError(
            "{} does not support scalar multiplication".format(type(self)))

    @classmethod
    def _automatic_dims(cls, dims, size):
        """Check if input dimension corresponds to qubit subsystems."""
        if dims is None:
            dims = size
        elif np.product(dims) != size:
            raise QiskitError("dimensions do not match size.")
        if isinstance(dims, (int, np.integer)):
            num_qubits = int(np.log2(dims))
            if 2 ** num_qubits == size:
                return num_qubits * (2,)
            return (dims,)
        return tuple(dims)

    # Overloads
    def __matmul__(self, other):
        return self.compose(other)

    def __mul__(self, other):
        return self.dot(other)

    def __rmul__(self, other):
        return self._multiply(other)

    def __pow__(self, n):
        return self.power(n)

    def __xor__(self, other):
        return self.tensor(other)

    def __truediv__(self, other):
        return self._multiply(1 / other)

    def __add__(self, other):
        return self._add(other)

    def __sub__(self, other):
        return self._add(-other)

    def __neg__(self):
        return self._multiply(-1)
