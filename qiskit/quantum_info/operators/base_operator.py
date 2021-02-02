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
from qiskit.quantum_info.operators.op_shape import OpShape


class BaseOperator:
    """Abstract linear operator base class."""

    def __init__(self, input_dims=None, output_dims=None,
                 num_qubits=None, shape=None, op_shape=None):
        """Initialize a BaseOperator shape

        Args:
            input_dims (tuple or int or None): Optional, input dimensions.
            output_dims (tuple or int or None): Optional, output dimensions.
            num_qubits (int): Optional, the number of qubits of the operator.
            shape (tuple): Optional, matrix shape for automatically determining
                           qubit dimenions.
            op_shape (OpShape): Optional, an OpShape object for operator dimensions.

        .. note::

            If `op_shape`` is specified it will take precedence over other
            kwargs.
        """
        self._qargs = None
        if op_shape:
            self._op_shape = op_shape
        else:
            self._op_shape = OpShape.auto(shape=shape,
                                          dims_l=output_dims,
                                          dims_r=input_dims,
                                          num_qubits=num_qubits)

    # Set higher priority than Numpy array and matrix classes
    __array_priority__ = 20

    def __call__(self, *qargs):
        """Return a shallow copy with qargs attribute set"""
        if len(qargs) == 1 and isinstance(qargs[0], (tuple, list)):
            qargs = qargs[0]
        n_qargs = len(qargs)
        if n_qargs not in self._op_shape.num_qargs:
            raise QiskitError(
                f"qargs does not match the number of operator qargs "
                "({n_qargs} not in {self._op_shape.num_qargs})")
        ret = copy.copy(self)
        ret._qargs = tuple(qargs)
        return ret

    def __eq__(self, other):
        return (isinstance(other, type(self))
                and self._op_shape == other._op_shape)

    @property
    def qargs(self):
        """Return the qargs for the operator."""
        return self._qargs

    @property
    def dim(self):
        """Return tuple (input_shape, output_shape)."""
        return self._op_shape._dim_r, self._op_shape._dim_l

    @property
    def num_qubits(self):
        """Return the number of qubits if a N-qubit operator or None otherwise."""
        return self._op_shape.num_qubits

    @property
    def _input_dim(self):
        """Return the total input dimension."""
        return self._op_shape._dim_r

    @property
    def _output_dim(self):
        """Return the total input dimension."""
        return self._op_shape._dim_l

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
        new_shape = OpShape.auto(
            dims_l=output_dims, dims_r=input_dims, num_qubits=num_qubits,
            shape=self._op_shape.shape)
        ret = copy.copy(self)
        ret._op_shape = new_shape
        return ret

    def input_dims(self, qargs=None):
        """Return tuple of input dimension for specified subsystems."""
        return self._op_shape.dims_r(qargs)

    def output_dims(self, qargs=None):
        """Return tuple of output dimension for specified subsystems."""
        return self._op_shape.dims_l(qargs)

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
        if not self._op_shape.is_square:
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
