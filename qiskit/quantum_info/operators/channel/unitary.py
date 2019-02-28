# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from numbers import Number
import numpy as np

from qiskit.qiskiterror import QiskitError
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from .basechannel import QuantumChannel
from .transformations import _to_unitary


class UnitaryChannel(QuantumChannel):
    """UnitaryChannel representation of a unitary quantum channel"""

    def __init__(self, data, input_dim=None, output_dim=None):
        # Check if input is a quantum channel object
        # If so we disregard the dimension kwargs
        if issubclass(data.__class__, QuantumChannel):
            input_dim, output_dim = data.dims
            unitary = _to_unitary(data.rep, data._data, input_dim, output_dim)
        else:
            # We initialize directly from superoperator matrix
            unitary = np.array(data, dtype=complex)
            # Determine input and output dimensions
            dout, din = unitary.shape
            if output_dim is None:
                output_dim = dout
            if input_dim is None:
                input_dim = din
            # Check dimensions
            if output_dim != input_dim:
                raise QiskitError('Invalid unitary matrix: must be square')
            if output_dim != dout or input_dim != din:
                raise QiskitError("Invalid input and output dimension for unitary matrix.")
        super().__init__('UnitaryChannel', unitary, input_dim, output_dim)

    def is_cptp(self, atol=None):
        """Test if channel is completely-positive trace-preserving (CPTP)"""
        # If the matrix is a unitary matrix the channel is CPTP
        return is_unitary_matrix(self._data, atol=atol)

    def evolve(self, state):
        """Apply the channel to a quantum state.

        Args:
            state (quantum_state like): A statevector or density matrix.

        Returns:
            A state of the same form as the input state (statevector or
            density matrix).
        """
        state = self._check_state(state)
        if state.ndim == 1 or state.ndim == 2 and state.shape[1] == 1:
            # Return evolved statevector
            return np.dot(self._data, state)
        # Return evolved density matrix
        return np.dot(np.dot(self._data, state), np.transpose(np.conj(self._data)))

    def conjugate(self, inplace=False):
        """Return the conjugate channel"""
        if inplace:
            np.conjugate(self._data, out=self._data)
            return self
        return UnitaryChannel(np.conj(self._data), self._input_dim, self._output_dim)

    def transpose(self, inplace=False):
        """Return the transpose channel"""
        # Swaps input and output dimensions
        output_dim = self._input_dim
        input_dim = self._output_dim
        if inplace:
            self._data = np.transpose(self._data)
            self._input_dim = input_dim
            self._output_dim = output_dim
            return self
        return UnitaryChannel(np.transpose(self._data), input_dim, output_dim)

    def compose(self, other, inplace=False, front=False):
        """Return UnitaryChannel for the composition channel B(A(input))

        Args:
            other (QuantumChannel): A unitary channel
            inplace (bool): If True modify the current object inplace [default: False]
            front (bool): If True compose in reverse order A(B(input)) [default: False]

        Returns:
            UnitaryChannel: The composition channel.

        Raises:
            QiskitError: if other is a non-unitary channel.
        """
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        # Check dimensions match up
        if front and self._input_dim != other._output_dim:
            raise QiskitError('input_dim of self must match output_dim of other')
        if not front and self._output_dim != other._input_dim:
            raise QiskitError('input_dim of other must match output_dim of self')
        # Convert to UnitaryChannel matrix
        if not isinstance(other, UnitaryChannel):
            other = UnitaryChannel(other)

        if front:
            # Composition A(B(input))
            if self._input_dim != other._output_dim:
                raise QiskitError('input_dim of self must match output_dim of other')
            input_dim = other._input_dim
            output_dim = self._output_dim
            if inplace:
                np.dot(self._data, other.data, out=self._data)
                self._input_dim = input_dim
                self._output_dim = output_dim
                return self
            return UnitaryChannel(np.dot(self._data, other.data), input_dim, output_dim)
        # Composition B(A(input))
        if self._output_dim != other._input_dim:
            raise QiskitError('input_dim of other must match output_dim of self')
        input_dim = self._input_dim
        output_dim = other._output_dim
        if inplace:
            # Numpy out raises error if we try and use out=self._data here
            self._data = np.dot(other.data, self._data)
            self._input_dim = input_dim
            self._output_dim = output_dim
            return self
        return UnitaryChannel(np.dot(other.data, self._data), input_dim, output_dim)

    def tensor(self, other, inplace=False, front=False):
        """Return the tensor product unitary channel.

        Args:
            other (QuantumChannel): A unitary channel
            inplace (bool): If True modify the current object inplace [default: False]
            front (bool): If False return (other ⊗ self),
                          if True return (self ⊗ other) [Default: False]
        Returns:
            UnitaryChannel: The tensor product channel.

        Raises:
            QiskitError: if other is a non-unitary channel.
        """
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        # Convert to Choi matrix
        if not isinstance(other, UnitaryChannel):
            other = UnitaryChannel(other)

        # Combined channel dimensions
        a_in, a_out = self.dims
        b_in, b_out = other.dims
        input_dim = a_in * b_in
        output_dim = a_out * b_out
        if front:
            data = np.kron(self._data, other._data)
        else:
            data = np.kron(other._data, self._data)
        if inplace:
            self._data = data
            self._input_dim = input_dim
            self._output_dim = output_dim
            return self
        # Not inplace so return new object
        return UnitaryChannel(data, input_dim, output_dim)

    def power(self, n, inplace=False):
        """ Rreturn composition of Channel with itself n times."""
        if not isinstance(n, int) or n < 1:
            raise QiskitError("Can only power with positive integer powers.")
        if self._input_dim != self._output_dim:
            raise QiskitError("Can only power with input_dim = output_dim.")
        # Override base class power so we can implement more efficiently
        # using Numpy.matrix_power
        if inplace:
            if n == 1:
                return self
            self._data = np.linalg.matrix_power(self._data, n)
            return self
        # Return new object
        return UnitaryChannel(np.linalg.matrix_power(self._data, n), *self.dims)

    def add(self, other, inplace=False):
        """Add another UnitaryChannel"""
        if not isinstance(other, UnitaryChannel):
            raise QiskitError('Other channel must be a UnitaryChannel')
        if self.dims != other.dims:
            raise QiskitError("Channel dimensions are not equal")
        if inplace:
            self._data += other._data
            return self
        input_dim, output_dim = self.dims
        return UnitaryChannel(self._data + other.data, input_dim, output_dim)

    def subtract(self, other, inplace=False):
        """Subtract another UnitaryChannel"""
        if not isinstance(other, UnitaryChannel):
            raise QiskitError('Other channel must be a UnitaryChannel')
        if self.dims != other.dims:
            raise QiskitError("Channel dimensions are not equal")
        if inplace:
            self._data -= other.data
            return self
        input_dim, output_dim = self.dims
        return UnitaryChannel(self._data - other.data, input_dim, output_dim)

    def multiply(self, other, inplace=False):
        """Multiple by a scalar"""
        if not isinstance(other, Number):
            raise QiskitError("Not a number")
        if inplace:
            self._data *= other
            return self
        input_dim, output_dim = self.dims
        return UnitaryChannel(other * self._data, input_dim, output_dim)
