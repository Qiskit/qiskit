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
from abc import ABCMeta, abstractmethod

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT


class TolerancesMeta(type):
    """Metaclass to handle tolerances"""
    def __init__(cls, *args, **kwargs):
        cls._ATOL_DEFAULT = ATOL_DEFAULT
        cls._RTOL_DEFAULT = RTOL_DEFAULT
        cls._MAX_TOL = 1e-4
        super().__init__(cls, args, kwargs)

    @property
    def atol(cls):
        """The default absolute tolerance parameter for float comparisons."""
        return cls._ATOL_DEFAULT

    def _check_value(cls, value, value_name):
        """Check if value is within valid ranges"""
        if value < 0:
            raise QiskitError(
                "Invalid {} ({}) must be non-negative.".format(value_name, value))
        if value > cls._MAX_TOL:
            raise QiskitError(
                "Invalid {} ({}) must be less than {}.".format(value_name, value, cls._MAX_TOL))

    @atol.setter
    def atol(cls, value):
        """Set the class default absolute tolerance parameter for float comparisons."""
        cls._check_value(value, "atol")  # pylint: disable=no-value-for-parameter
        cls._ATOL_DEFAULT = value

    @property
    def rtol(cls):
        """The relative tolerance parameter for float comparisons."""
        return cls._RTOL_DEFAULT

    @rtol.setter
    def rtol(cls, value):
        """Set the class default relative tolerance parameter for float comparisons."""
        cls._check_value(value, "rtol")  # pylint: disable=no-value-for-parameter
        cls._RTOL_DEFAULT = value


class AbstractTolerancesMeta(TolerancesMeta, ABCMeta):
    """Abstract Metaclass to handle tolerances"""
    pass


class BaseOperator(metaclass=AbstractTolerancesMeta):
    """Abstract linear operator base class."""

    def __init__(self, input_dims, output_dims):
        """Initialize an operator object."""
        # Dimension attributes
        # Note that the tuples of input and output dims are ordered
        # from least-significant to most-significant subsystems
        self._qargs = None        # qargs for composition, set with __call__
        self._input_dims = None   # tuple of input dimensions of each subsystem
        self._output_dims = None  # tuple of output dimensions of each subsystem
        self._input_dim = None    # combined input dimension of all subsystems
        self._output_dim = None   # combined output dimension of all subsystems
        self._num_qubits = None   # number of qubit subsystems if N-qubit operator
        self._set_dims(input_dims, output_dims)

    def __call__(self, qargs):
        """Return a clone with qargs set"""
        if isinstance(qargs, int):
            qargs = [qargs]
        n_qargs = len(qargs)
        # We don't know if qargs will be applied to input our output
        # dimensions so we just check it matches one of them.
        if n_qargs not in (len(self._input_dims), len(self._output_dims)):
            raise QiskitError(
                "Length of qargs ({}) does not match number of input ({})"
                " or output ({}) subsystems.".format(
                    n_qargs, len(self._input_dims), len(self._output_dims)))
        # Make a shallow copy
        ret = copy.copy(self)
        ret._qargs = qargs
        return ret

    def __eq__(self, other):
        """Check types and subsystem dimensions are equal"""
        return (isinstance(other, self.__class__) and
                self._input_dims == other._input_dims and
                self._output_dims == other._output_dims)

    @property
    def qargs(self):
        """Return the qargs for the operator."""
        return self._qargs

    @property
    def dim(self):
        """Return tuple (input_shape, output_shape)."""
        return self._input_dim, self._output_dim

    @property
    def num_qubits(self):
        """Return the number of qubits if a N-qubit operator or None otherwise."""
        return self._num_qubits

    @property
    def atol(self):
        """The default absolute tolerance parameter for float comparisons."""
        return self.__class__.atol

    @property
    def rtol(self):
        """The relative tolerance parameter for float comparisons."""
        return self.__class__.rtol

    @classmethod
    def set_atol(cls, value):
        """Set the class default absolute tolerance parameter for float comparisons.

        DEPRECATED: use operator.atol = value instead
        """
        warnings.warn("`{}.set_atol` method is deprecated, use `{}.atol = "
                      "value` instead.".format(cls.__name__, cls.__name__),
                      DeprecationWarning)
        cls.atol = value

    @classmethod
    def set_rtol(cls, value):
        """Set the class default relative tolerance parameter for float comparisons.

        DEPRECATED: use operator.rtol = value instead
        """
        warnings.warn("`{}.set_rtol` method is deprecated, use `{}.rtol = "
                      "value` instead.".format(cls.__name__, cls.__name__),
                      DeprecationWarning)
        cls.rtol = value

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

    def add(self, other):
        """Return the linear operator self + other.

        DEPRECATED: use ``operator + other`` instead.

        Args:
            other (BaseOperator): an operator object.

        Returns:
            BaseOperator: the operator self + other.
        """
        warnings.warn("`BaseOperator.add` method is deprecated, use"
                      "`op + other` instead.", DeprecationWarning)
        return self._add(other)

    def subtract(self, other):
        """Return the linear operator self - other.

        DEPRECATED: use ``operator - other`` instead.

        Args:
            other (BaseOperator): an operator object.

        Returns:
            BaseOperator: the operator self - other.
        """
        warnings.warn("`BaseOperator.subtract` method is deprecated, use"
                      "`op - other` instead", DeprecationWarning)
        return self._add(-other)

    def multiply(self, other):
        """Return the linear operator other * self.

        DEPRECATED: use ``other * operator`` instead.

        Args:
            other (complex): a complex number.

        Returns:
            BaseOperator: the linear operator other * self.

        Raises:
            NotImplementedError: if subclass does not support multiplication.
        """
        warnings.warn("`BaseOperator.multiply` method is deprecated, use"
                      "the `other * op` instead", DeprecationWarning)
        return self._multiply(other)

    def _add(self, other, qargs=None):
        """Return the linear operator self + other.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (BaseOperator): an operator object.
            qargs (None or list): optional subsystems to add on
                                  (Default: None)

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

    def _set_dims(self, input_dims, output_dims):
        """Set dimension attributes"""
        # Shape lists the dimension of each subsystem starting from
        # least significant through to most significant.
        self._input_dims = tuple(input_dims)
        self._output_dims = tuple(output_dims)
        # The total input and output dimensions are given by the product
        # of all subsystem dimension in the input_dims/output_dims.
        self._input_dim = np.product(input_dims)
        self._output_dim = np.product(output_dims)
        # Check if an N-qubit operator
        if (self._input_dims == self._output_dims and
                set(self._input_dims) == set([2])):
            # If so set the number of qubits
            self._num_qubits = len(self._input_dims)
        else:
            # Otherwise set the number of qubits to None
            self._num_qubits = None

    def _get_compose_dims(self, other, qargs, front):
        """Check dimensions are compatible for composition.

        Args:
            other (BaseOperator): another operator object.
            qargs (None or list): compose qargs kwarg value.
            front (bool): compose front kwarg value.

        Returns:
            tuple: the tuple (input_dims, output_dims) for the composed
                   operator.
        Raises:
            QiskitError: if operator dimensions are invalid for compose.
        """
        if front:
            output_dims = self._output_dims
            if qargs is None:
                if other._output_dim != self._input_dim:
                    raise QiskitError(
                        "Other operator combined output dimension ({}) does not"
                        " match current combined input dimension ({}).".format(
                            other._output_dim, self._input_dim))
                input_dims = other._input_dims
            else:
                if other._output_dims != self.input_dims(qargs):
                    raise QiskitError(
                        "Other operator output dimensions ({}) does not"
                        " match current subsystem input dimensions ({}).".format(
                            other._output_dims, self.input_dims(qargs)))
                input_dims = list(self._input_dims)
                for i, qubit in enumerate(qargs):
                    input_dims[qubit] = other._input_dims[i]
        else:
            input_dims = self._input_dims
            if qargs is None:
                if self._output_dim != other._input_dim:
                    raise QiskitError(
                        "Other operator combined input dimension ({}) does not"
                        " match current combined output dimension ({}).".format(
                            other._input_dim, self._output_dim))
                output_dims = other._output_dims
            else:
                if self.output_dims(qargs) != other._input_dims:
                    raise QiskitError(
                        "Other operator input dimensions ({}) does not"
                        " match current subsystem output dimension ({}).".format(
                            other._input_dims, self.output_dims(qargs)))
                output_dims = list(self._output_dims)
                for i, qubit in enumerate(qargs):
                    output_dims[qubit] = other._output_dims[i]
        return input_dims, output_dims

    def _validate_add_dims(self, other, qargs=None):
        """Check dimensions are compatible for addition.

        Args:
            other (BaseOperator): another operator object.
            qargs (None or list): compose qargs kwarg value.

        Raises:
            QiskitError: if operators have incompatibile dimensions for addition.
        """
        if qargs is None:
            # For adding without qargs we only require that operators have
            # the same total dimensions rather than each subsystem dimension
            # matching.
            if self.dim != other.dim:
                raise QiskitError(
                    "Cannot add operators with different shapes"
                    " ({} != {}).".format(self.dim, other.dim))
        else:
            # If adding on subsystems the operators must have equal
            # shape on subsystems
            if (self._input_dims != self._output_dims or
                    other._input_dims != other._output_dims):
                raise QiskitError(
                    "Cannot add operators on subsystems for non-square"
                    " operator.")
            if self.input_dims(qargs) != other._input_dims:
                raise QiskitError(
                    "Cannot add operators on subsystems with different"
                    " dimensions ({} != {}).".format(
                        self.input_dims(qargs), other._input_dims))

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
