# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from abc import ABC, abstractmethod

import numpy as np

from qiskit.qiskiterror import QiskitError
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT


class QuantumChannel(ABC):
    """Quantum channel representation base class."""

    ATOL = ATOL_DEFAULT
    MAX_ATOL = 1e-4

    def __init__(self, rep, data, input_dim, output_dim):
        if not isinstance(rep, str):
            raise QiskitError("rep must be a string not a {}".format(rep.__class__))
        self._rep = rep
        self._data = data
        self._input_dim = input_dim
        self._output_dim = output_dim

    def __eq__(self, other):
        if isinstance(other, self.__class__) and self.dims == other.dims:
            return np.allclose(self.data, other.data, atol=self.atol)
        return False

    def __repr__(self):
        return '{}({}, input_dim={}, output_dim={})'.format(self.rep,
                                                            self.data,
                                                            self._input_dim,
                                                            self._output_dim)

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
        """Atol parameter for float comparisons."""
        # NOTE: This should really be a class method so that it can
        # be overriden for all QuantumChannel subclasses
        return QuantumChannel.ATOL

    @atol.setter
    def atol(cls, atol):
        """Set atol parameter for float comparisons."""
        MAX_ATOL = QuantumChannel.MAX_ATOL
        if atol < 0:
            raise QiskitError("Invalid atol: must be non-negative.")
        if atol > MAX_ATOL:
            raise QiskitError("Invalid atol: must be less than {}.".format(MAX_ATOL))
        QuantumChannel.ATOL = atol

    def copy(self):
        """Make a copy of current channel."""
        # The constructor of subclasses from raw data should be a copy
        return self.__class__(self._data, self._input_dim, self._output_dim)

    @abstractmethod
    def evolve(self, state):
        """Apply the channel to a quantum state.

        Args:
            state (quantum_state like): A statevector or density matrix.

        Returns:
            A density matrix.
        """
        pass

    @abstractmethod
    def is_cptp(self):
        """Check if channel is completely-positive trace-preserving."""
        pass

    @abstractmethod
    def conjugate(self, inplace=False):
        """Return the conjugate channel"""
        pass

    @abstractmethod
    def transpose(self, inplace=False):
        """Return the transpose channel"""
        pass

    def adjoint(self, inplace=False):
        """Return the adjoint channel"""
        return self.conjugate(inplace=inplace).transpose(inplace=inplace)

    @abstractmethod
    def compose(self, other, inplace=False, front=False):
        """Return the composition channel.

        Args:
            other (QuantumChannel): A quantum channel representation object
            inplace (bool): If True modify the current object inplace [default: False]
            front (bool): If True compose in reverse order A(B(input)) [default: False]

        Returns:
            QuantumChannel: the composition channel.
        """
        pass

    @abstractmethod
    def tensor(self, other, inplace=False, front=False):
        """Return the tensor product channel.

        Args:
            other (QuantumChannel): A quantum channel representation object
            inplace (bool): If True modify the current object inplace [default: False]
            front (bool): If False return (other ⊗ self),
                          if True return (self ⊗ other) [Default: False]
        Returns:
            QuantumChannel: the tensor product channel.
        """
        pass

    def power(self, n, inplace=False):
        """ Rreturn composition of Channel with itself n times."""
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
        """Add another QuantumChannel"""
        pass

    @abstractmethod
    def subtract(self, other, inplace=False):
        """Subtract another QuantumChannel"""
        pass

    @abstractmethod
    def multiply(self, other, inplace=False):
        """Multiple by a scalar"""
        pass

    def _check_state(self, state):
        """Check input state is valid for quantum channel"""
        state = np.array(state)
        shape = state.shape
        ndim = state.ndim
        if ndim > 2:
            raise QiskitError('Input state is not a vector or matrix.')
        if shape[0] != self._input_dim:
            raise QiskitError('Input state is wrong size for channel input dimension.')
        # Flatten column-vector to vector
        if ndim == 2:
            if shape[1] != 1 and shape[1] != self._input_dim:
                raise QiskitError('Input state is wrong size for channel input dimension.')
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
