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

from abc import ABC, abstractmethod

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT


class BaseOperator(ABC):
    """Abstract linear operator base class"""

    ATOL = ATOL_DEFAULT
    RTOL = RTOL_DEFAULT
    MAX_TOL = 1e-4

    def __init__(self, rep, data, input_dims, output_dims):
        """Initialize an operator object."""
        if not isinstance(rep, str):
            raise QiskitError("rep must be a string not a {}".format(
                rep.__class__))
        self._rep = rep
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

    def __repr__(self):
        return '{}({}, input_dims={}, output_dims={})'.format(
            self.rep, self.data, self._input_dims, self._output_dims)

    @property
    def rep(self):
        """Return operator representation string."""
        return self._rep

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

    def _reshape(self, input_dims=None, output_dims=None):
        """Reshape input and output dimensions of operator.

        Arg:
            input_dims (tuple): new subsystem input dimensions.
            output_dims (tuple): new subsystem output dimensions.

        Returns:
            Operator: returns self with reshaped input and output dimensions.

        Raises:
            QiskitError: if combined size of all subsystem input dimension or
            subsystem output dimensions is not constant.
        """
        if input_dims is not None:
            if np.product(input_dims) != self._input_dim:
                raise QiskitError(
                    "Reshaped input_dims are incompatible with combined input dimension."
                )
            self._input_dims = tuple(input_dims)
        if output_dims is not None:
            if np.product(output_dims) != self._output_dim:
                raise QiskitError(
                    "Reshaped input_dims are incompatible with combined input dimension."
                )
            self._output_dims = tuple(output_dims)
        return self

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
        """Make a copy of current operator."""
        # pylint: disable=no-value-for-parameter
        # The constructor of subclasses from raw data should be a copy
        return self.__class__(self.data, self.input_dims(), self.output_dims())

    def adjoint(self):
        """Return the adjoint of the operator."""
        return self.conjugate().transpose()

    @abstractmethod
    def is_unitary(self, atol=None, rtol=None):
        """Return True if operator is a unitary matrix."""
        pass

    @abstractmethod
    def to_operator(self):
        """Convert operator to matrix operator class"""
        pass

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
        """Return the composition channel self∘other.

        Args:
            other (BaseOperator): an operator object.
            qargs (list): a list of subsystem positions to compose other on.
            front (bool): If False compose in standard order other(self(input))
                          otherwise compose in reverse order self(other(input))
                          [default: False]

        Returns:
            BaseOperator: The composed operator.

        Raises:
            QiskitError: if other cannot be converted to an operator, or has
            incompatible dimensions for specified subsystems.
        """
        pass

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
        # this method should be overriden in that class.
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

    @abstractmethod
    def add(self, other):
        """Return the linear operator self + other.

        Args:
            other (BaseOperator): an operator object.

        Returns:
            LinearOperator: the linear operator self + other.

        Raises:
            QiskitError: if other is not an operator, or has incompatible
            dimensions.
        """
        pass

    @abstractmethod
    def subtract(self, other):
        """Return the linear operator self - other.

        Args:
            other (BaseOperator): an operator object.

        Returns:
            LinearOperator: the linear operator self - other.

        Raises:
            QiskitError: if other is not an operator, or has incompatible
            dimensions.
        """
        pass

    @abstractmethod
    def multiply(self, other):
        """Return the linear operator self + other.

        Args:
            other (complex): a complex number.

        Returns:
            Operator: the linear operator other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        pass

    @abstractmethod
    def _evolve(self, state, qargs=None):
        """Evolve a quantum state by the operator.

        Args:
            state (QuantumState): The input statevector or density matrix.
            qargs (list): a list of QuantumState subsystem positions to apply
                           the operator on.

        Returns:
            QuantumState: the output quantum state.

        Raises:
            QiskitError: if the operator dimension does not match the
            specified QuantumState subsystem dimensions.
        """
        pass

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

    @classmethod
    def _einsum_matmul(cls, tensor, mat, indices, shift=0, right_mul=False):
        """Perform a contraction using Numpy.einsum

        Args:
            tensor (np.array): a vector or matrix reshaped to a rank-N tensor.
            mat (np.array): a matrix reshaped to a rank-2M tensor.
            indices (list): tensor indices to contract with mat.
            shift (int): shift for indices of tensor to contract [Default: 0].
            right_mul (bool): if True right multiply tensor by mat
                              (else left multiply) [Default: False].

        Returns:
            Numpy.ndarray: the matrix multiplied rank-N tensor.

        Raises:
            QiskitError: if mat is not an even rank tensor.
        """
        rank = tensor.ndim
        rank_mat = mat.ndim
        if rank_mat % 2 != 0:
            raise QiskitError(
                "Contracted matrix must have an even number of indices.")
        # Get einsum indices for tensor
        indices_tensor = list(range(rank))
        for j, index in enumerate(indices):
            indices_tensor[index + shift] = rank + j
        # Get einsum indices for mat
        mat_contract = list(reversed(range(rank, rank + len(indices))))
        mat_free = [index + shift for index in reversed(indices)]
        if right_mul:
            indices_mat = mat_contract + mat_free
        else:
            indices_mat = mat_free + mat_contract
        return np.einsum(tensor, indices_tensor, mat, indices_mat)

    # Overloads
    def __matmul__(self, other):
        return self.compose(other)

    def __pow__(self, n):
        return self.power(n)

    def __xor__(self, other):
        return self.tensor(other)

    def __mul__(self, other):
        return self.multiply(other)

    def __truediv__(self, other):
        return self.multiply(1 / other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __neg__(self):
        return self.multiply(-1)
