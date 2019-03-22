# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
Abstract base class for Quantum Channels.
"""

from abc import ABC, abstractmethod

import numpy as np

from qiskit.qiskiterror import QiskitError
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT


class QuantumChannel(ABC):
    """Quantum channel representation base class."""

    ATOL = ATOL_DEFAULT
    RTOL = RTOL_DEFAULT
    MAX_TOL = 1e-4

    def __init__(self, rep, data, input_dim, output_dim):
        if not isinstance(rep, str):
            raise QiskitError("rep must be a string not a {}".format(
                rep.__class__))
        self._rep = rep
        self._data = data
        self._input_dim = input_dim
        self._output_dim = output_dim

    def __eq__(self, other):
        if isinstance(other, self.__class__) and self.dims == other.dims:
            return np.allclose(
                self.data, other.data, rtol=self.rtol, atol=self.atol)
        return False

    def __repr__(self):
        return '{}({}, input_dim={}, output_dim={})'.format(
            self.rep, self.data, self._input_dim, self._output_dim)

    @property
    def rep(self):
        """Return Channel representation"""
        return self._rep

    @property
    def dims(self):
        """Return tuple (input dimension, output dimension)"""
        return self._input_dim, self._output_dim

    @property
    def data(self):
        """Return data"""
        return self._data

    @property
    def atol(self):
        """The absolute tolerence parameter for float comparisons."""
        # NOTE: This should really be a class method so that it can
        # be overriden for all QuantumChannel subclasses
        return QuantumChannel.ATOL

    @atol.setter
    def atol(self, atol):
        """Set the absolute tolerence parameter for float comparisons."""
        max_tol = QuantumChannel.MAX_TOL
        if atol < 0:
            raise QiskitError("Invalid atol: must be non-negative.")
        if atol > max_tol:
            raise QiskitError(
                "Invalid atol: must be less than {}.".format(max_tol))
        QuantumChannel.ATOL = atol

    @property
    def rtol(self):
        """The relative tolerence parameter for float comparisons."""
        # NOTE: This should really be a class method so that it can
        # be overriden for all QuantumChannel subclasses
        return QuantumChannel.RTOL

    @rtol.setter
    def rtol(self, rtol):
        """Set the relative tolerence parameter for float comparisons."""
        max_tol = QuantumChannel.MAX_TOL
        if rtol < 0:
            raise QiskitError("Invalid rtol: must be non-negative.")
        if rtol > max_tol:
            raise QiskitError(
                "Invalid rtol: must be less than {}.".format(max_tol))
        QuantumChannel.RTOL = rtol

    def copy(self):
        """Make a copy of current channel."""
        # pylint: disable=no-value-for-parameter
        # The constructor of subclasses from raw data should be a copy
        return self.__class__(self._data, self._input_dim, self._output_dim)

    @abstractmethod
    def is_cptp(self):
        """Return True if completely-positive trace-preserving."""
        pass

    @abstractmethod
    def _evolve(self, state):
        """Evolve a quantum state by the QuantumChannel.

        Args:
            state (QuantumState): A statevector or density matrix.

        Returns:
            QuantumState: the output quantum state.
        """
        pass

    @abstractmethod
    def conjugate(self, inplace=False):
        """Return the conjugate of the  QuantumChannel.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            QuantumChannel: the conjugate of the quantum channel.
        """
        pass

    @abstractmethod
    def transpose(self, inplace=False):
        """Return the transpose of the QuantumChannel.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            QuantumChannel: the transpose of the quantum channel.
        """
        pass

    def adjoint(self, inplace=False):
        """Return the adjoint of the QuantumChannel.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            QuantumChannel: the adjoint of the quantum channel.
        """
        return self.conjugate(inplace=inplace).transpose(inplace=inplace)

    @abstractmethod
    def compose(self, other, inplace=False, front=False):
        """Return the composition channel self∘other.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                            [Default: False]
            front (bool): If False compose in standard order other(self(input))
                          otherwise compose in reverse order self(other(input))
                          [default: False]

        Returns:
            QuantumChannel: the composition channel.
        """
        pass

    @abstractmethod
    def tensor(self, other, inplace=False):
        """Return the tensor product channel self ⊗ other.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                            [Default: False]

        Returns:
            QuantumChannel: the tensor product channel self ⊗ other.
        """
        pass

    @abstractmethod
    def expand(self, other, inplace=False):
        """Return the tensor product channel other ⊗ self.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                            [Default: False]

        Returns:
            QuantumChannel: the tensor product channel other ⊗ self.
        """
        pass

    def power(self, n, inplace=False):
        """Return the compose of a QuantumChannel with itself n times.

        Args:
            n (int): the number of times to compose with self (n>0).
            inplace (bool): If True modify the current object inplace
                            [Default: False]

        Returns:
            QuantumChannel: the n-times composition channel.

        Raises:
            QiskitError: if the input and output dimensions of the
            QuantumChannel are not equal, or the power is not a positive
            integer.
        """
        if not isinstance(n, int) or n < 1:
            raise QiskitError("Can only power with positive integer powers.")
        if self._input_dim != self._output_dim:
            raise QiskitError("Can only power with input_dim = output_dim.")
        # Update inplace
        if inplace:
            if n == 1:
                return self
            # cache current state to apply n-times
            cache = self.copy()
            for _ in range(1, n):
                self.compose(cache, inplace=True)
            return self
        # Return new object
        ret = self.copy()
        for _ in range(1, n):
            ret = ret.compose(self)
        return ret

    @abstractmethod
    def add(self, other, inplace=False):
        """Return the QuantumChannel self + other.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            QuantumChannel: the linear addition self + other.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        pass

    @abstractmethod
    def subtract(self, other, inplace=False):
        """Return the QuantumChannel self - other.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            QuantumChannel: the linear subtraction self - other.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        pass

    @abstractmethod
    def multiply(self, other, inplace=False):
        """Return the QuantumChannel self + other.

        Args:
            other (complex): a complex number
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            QuantumChannel: the scalar multiplication other * self.

        Raises:
            QiskitError: if other is not a valid scalar.
        """
        pass

    def _check_state(self, state):
        """Check input state is valid for quantum channel"""
        state = np.array(state)
        shape = state.shape
        ndim = state.ndim
        if ndim > 2:
            raise QiskitError('Input state is not a vector or matrix.')
        if shape[0] != self._input_dim:
            raise QiskitError(
                'Input state is wrong size for channel input dimension.')
        # Flatten column-vector to vector
        if ndim == 2:
            if shape[1] != 1 and shape[1] != self._input_dim:
                raise QiskitError(
                    'Input state is wrong size for channel input dimension.')
            if shape[1] == 1:
                # flatten colum-vector to vector
                state = np.reshape(state, shape[0])
        return state

    def _format_density_matrix(self, state):
        """Check if input state is valid for quantum channel and convert to density matrix"""
        if state.ndim == 1:
            state = np.outer(state, np.transpose(np.conj(state)))
        return state

    # Overloads
    def __matmul__(self, other):
        return self.compose(other)

    def __imatmul__(self, other):
        return self.compose(other, inplace=True)

    def __pow__(self, n):
        return self.power(n)

    def __ipow__(self, n):
        return self.power(n, inplace=True)

    def __xor__(self, other):
        return self.tensor(other)

    def __ixor__(self, other):
        return self.tensor(other, inplace=True)

    def __mul__(self, other):
        return self.multiply(other)

    def __imul__(self, other):
        return self.multiply(other, inplace=False)

    def __truediv__(self, other):
        return self.multiply(1 / other)

    def __itruediv__(self, other):
        return self.multiply(1 / other, inplace=False)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return self.add(other)

    def __iadd__(self, other):
        return self.add(other, inplace=True)

    def __sub__(self, other):
        return self.subtract(other)

    def __isub__(self, other):
        return self.subtract(other, inplace=True)

    def __neg__(self):
        return self.multiply(-1)
