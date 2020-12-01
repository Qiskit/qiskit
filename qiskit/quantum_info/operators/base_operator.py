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
from abc import abstractmethod

import numpy as np

from qiskit.exceptions import QiskitError


class BaseOperator:
    """Abstract linear operator base class."""

    def __init__(self, input_dims=None, output_dims=None, num_qubits=None):
        """Initialize an operator object."""
        # qargs for composition, set with __call__
        self._qargs = None

        # Number of qubit subsystems if N-qubit operator.
        # None if operator is not an N-qubit operator
        self._num_qubits = None

        # General dimension operators
        # Note that the tuples of input and output dims are ordered
        # from least-significant to most-significant subsystems
        # These may be None for N-qubit operators
        self._input_dims = None   # tuple of input dimensions of each subsystem
        self._output_dims = None  # tuple of output dimensions of each subsystem

        # The number of input and output subsystems
        # This is either the number of qubits, or length of the input/output dims
        self._num_input = 0
        self._num_output = 0

        # Set dimension attributes
        if num_qubits:
            self._set_qubit_dims(num_qubits)
        elif input_dims is not None and output_dims is not None:
            self._set_dims(input_dims, output_dims)

    # Set higher priority than Numpy array and matrix classes
    __array_priority__ = 20

    def __call__(self, qargs):
        """Return a clone with qargs set"""
        if isinstance(qargs, int):
            qargs = [qargs]
        n_qargs = len(qargs)
        # We don't know if qargs will be applied to input our output
        # dimensions so we just check it matches one of them.
        if n_qargs != self._num_qubits and n_qargs not in (
                self._num_input, self._num_output):
            raise QiskitError(
                "Length of qargs ({}) does not match number of input ({})"
                " or output ({}) subsystems.".format(
                    n_qargs, self._num_input, self._num_output))
        # Make a shallow copy
        ret = copy.copy(self)
        ret._qargs = qargs
        return ret

    def __eq__(self, other):
        """Check types and subsystem dimensions are equal"""
        return (isinstance(other, self.__class__) and
                self._num_qubits == other._num_qubits and
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
    def _input_dim(self):
        """Return the total input dimension."""
        if self.num_qubits:
            return 2 ** self.num_qubits
        return np.product(self._input_dims)

    @property
    def _output_dim(self):
        """Return the total input dimension."""
        if self.num_qubits:
            return 2 ** self.num_qubits
        return np.product(self._output_dims)

    def reshape(self, input_dims=None, output_dims=None, num_qubits=None):
        """Return a shallow copy with reshaped input and output subsystem dimensions.

        Args:
            input_dims (None or tuple): new subsystem input dimensions.
                If None the original input dims will be preserved [Default: None].
            output_dims (None or tuple): new subsystem output dimensions.
                If None the original output dims will be preserved [Default: None].
            num_qubits (None or int): reshape to an N-qubit operator [Default: None].

        Returns:
            BaseOperator: returns self with reshaped input and output dimensions.

        Raises:
            QiskitError: if combined size of all subsystem input dimension or
                         subsystem output dimensions is not constant.
        """
        # Trivial case
        if num_qubits:
            if self.num_qubits == num_qubits:
                return self
            dim = 2 * (2 ** num_qubits, )
            if self.dim != dim:
                raise QiskitError(
                    "Reshaped num_qubits ({}) are incompatible with combined"
                    " input and output dimensions dimension ({}).".format(
                        num_qubits, self.dim))
            clone = copy.copy(self)
            clone._set_qubit_dims(num_qubits)
            return clone

        if output_dims is None and input_dims is None:
            return clone

        input_dim, output_dim = self.dim
        if input_dims is not None:
            if np.product(input_dims) != input_dim:
                raise QiskitError(
                    "Reshaped input_dims ({}) are incompatible with combined"
                    " input dimension ({}).".format(input_dims, input_dim))
        else:
            input_dims = self.input_dims()
        if output_dims is not None:
            if np.product(output_dims) != output_dim:
                raise QiskitError(
                    "Reshaped output_dims ({}) are incompatible with combined"
                    " output dimension ({}).".format(output_dims, output_dim))
        else:
            output_dim = self.output_dims()
        # Set new dimensions
        clone = copy.copy(self)
        clone._set_dims(input_dims, output_dims)
        return clone

    def input_dims(self, qargs=None):
        """Return tuple of input dimension for specified subsystems."""
        if self._num_qubits:
            num = self._num_qubits if qargs is None else len(qargs)
            return num * (2, )
        if qargs is None:
            return self._input_dims
        return tuple(self._input_dims[i] for i in qargs)

    def output_dims(self, qargs=None):
        """Return tuple of output dimension for specified subsystems."""
        if self._num_qubits:
            num = self._num_qubits if qargs is None else len(qargs)
            return num * (2, )
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
        if self._input_dims != self._output_dims:
            raise QiskitError("Can only power with input_dims = output_dims.")
        ret = self.copy()
        for _ in range(1, n):
            ret = ret.compose(self)
        return ret

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
    def _automatic_dims(cls, input_dims, input_size, output_dims=None, output_size=None):
        """Check if dimension corresponds to qubit subsystems."""
        if input_dims is None:
            input_dims = input_size
        elif np.product(input_dims) != input_size:
            raise QiskitError("Input dimensions do not match size.")
        din_int = isinstance(input_dims, (int, np.integer))

        if output_size is None:
            output_size = input_size
            output_dims = input_dims
        elif output_dims is None:
            output_dims = output_size
        elif np.product(output_dims) != output_size:
            raise QiskitError("Output dimensions do not match size.")
        dout_int = isinstance(output_dims, (int, np.integer))

        # Check if N-qubit
        if (input_size == output_size and din_int and dout_int):
            num_qubits = int(np.log2(input_dims))
            if 2 ** num_qubits == input_size:
                return None, None, num_qubits

        # General dims
        input_dims = (input_dims, ) if din_int else tuple(input_dims)
        output_dims = (output_dims, ) if dout_int else tuple(output_dims)
        return input_dims, output_dims, None

    def _set_qubit_dims(self, num_qubits):
        """Set dimension attributes"""
        self._num_qubits = int(num_qubits)
        self._num_input = self._num_qubits
        self._num_output = self._num_qubits
        self._input_dims = None
        self._output_dims = None

    def _set_dims(self, input_dims, output_dims):
        """Set dimension attributes"""
        # Shape lists the dimension of each subsystem starting from
        # least significant through to most significant.
        if (input_dims == output_dims and set(input_dims) == {2}):
            self._set_qubit_dims(len(input_dims))
        else:
            self._input_dims = tuple(input_dims)
            self._output_dims = tuple(output_dims)
            self._num_input = len(self._input_dims)
            self._num_output = len(self._output_dims)
            self._num_qubits = None

    def _get_compose_dims(self, other, qargs, front):
        """Check subsystems are compatible for composition."""
        # Check if both qubit operators
        if self.num_qubits and other.num_qubits:
            if qargs and other.num_qubits != len(qargs):
                raise QiskitError(
                    "Other operator number of qubits does not match the "
                    "number of qargs ({} != {})".format(
                        other.num_qubits, len(qargs)))
            if qargs is None and self.num_qubits != other.num_qubits:
                raise QiskitError(
                    "Other operator number of qubits does not match the "
                    "current operator ({} != {})".format(
                        other.num_qubits, self.num_qubits))
            dims = [2] * self.num_qubits
            return dims, dims

        # General case
        if front:
            if other.output_dims() != self.input_dims(qargs):
                raise QiskitError(
                    "Other operator output dimensions ({}) does not"
                    " match current input dimensions ({}).".format(
                        other.output_dims(qargs), self.input_dims()))
            output_dims = self.output_dims()
            if qargs is None:
                input_dims = other.input_dims()
            else:
                input_dims = list(self.input_dims())
                for qubit, dim in zip(qargs, other.input_dims()):
                    input_dims[qubit] = dim
        else:
            if other.input_dims() != self.output_dims(qargs):
                raise QiskitError(
                    "Other operator input dimensions ({}) does not"
                    " match current output dimensions ({}).".format(
                        other.output_dims(qargs), self.input_dims()))
            input_dims = self.input_dims()
            if qargs is None:
                output_dims = other.output_dims()
            else:
                output_dims = list(self.output_dims())
                for qubit, dim in zip(qargs, other.output_dims()):
                    output_dims[qubit] = dim
        return input_dims, output_dims

    def _validate_add_dims(self, other, qargs=None):
        """Check dimensions are compatible for addition.

        Args:
            other (BaseOperator): another operator object.
            qargs (None or list): compose qargs kwarg value.

        Raises:
            QiskitError: if operators have incompatibile dimensions for addition.
        """
        if self.num_qubits and other.num_qubits:
            self._validate_qubit_add_dims(other, qargs=qargs)
        else:
            self._validate_qudit_add_dims(other, qargs=qargs)

    def _validate_qubit_add_dims(self, other, qargs=None):
        """Check qubit operator dimensions are compatible for addition."""
        if qargs is None:
            # For adding without qargs we only require that operators have
            # the same total dimensions rather than each subsystem dimension
            # matching.
            if self.num_qubits != other.num_qubits:
                raise QiskitError(
                    "Cannot add operators with different numbers of qubits"
                    " ({} != {}).".format(self.num_qubits, other.num_qubits))
        else:
            if len(qargs) != other.num_qubits:
                raise QiskitError(
                    "Cannot add operators on subsystems with different"
                    " number of qubits ({} != {}).".format(
                        len(qargs), other.num_qubits))

    def _validate_qudit_add_dims(self, other, qargs=None):
        """Check general operatior dimensions are compatible for addition."""
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
            if (self.input_dims() != self.output_dims() or
                    other.input_dims() != other.output_dims()):
                raise QiskitError(
                    "Cannot add operators on subsystems for non-square"
                    " operator.")
            if self.input_dims(qargs) != other.input_dims():
                raise QiskitError(
                    "Cannot add operators on subsystems with different"
                    " dimensions ({} != {}).".format(
                        self.input_dims(qargs), other.input_dims()))

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
