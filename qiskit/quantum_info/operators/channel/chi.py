# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from numbers import Number

import numpy as np

from qiskit.qiskiterror import QiskitError
from .basechannel import QuantumChannel
from .choi import Choi
from .transformations import _to_chi, _bipartite_tensor


class Chi(QuantumChannel):
    """Pauli basis Chi-matrix representation of a quantum channel

    The Chi-matrix is the Pauli-basis representation of the Chi-Matrix.
    """

    def __init__(self, data, input_dim=None, output_dim=None):
        # Check if input is a quantum channel object
        # If so we disregard the dimension kwargs
        if issubclass(data.__class__, QuantumChannel):
            input_dim, output_dim = data.dims
            if input_dim != output_dim:
                raise QiskitError("Cannot convert to Chi-matrix: input_dim " +
                                  "({}) != output_dim ({})".format(input_dim, output_dim))
            chi_mat = _to_chi(data.rep, data._data, input_dim, output_dim)
        else:
            chi_mat = np.array(data, dtype=complex)
            # Determine input and output dimensions
            dl, dr = chi_mat.shape
            if dl != dr:
                raise QiskitError('Invalid Choi-matrix input.')
            if output_dim is None and input_dim is None:
                output_dim = int(np.sqrt(dl))
                input_dim = dl // output_dim
            elif input_dim is None:
                input_dim = dl // output_dim
            elif output_dim is None:
                output_dim = dl // input_dim
            # Check dimensions
            if input_dim * output_dim != dl:
                raise QiskitError("Invalid input and output dimension for Chi-matrix input.")
            nqubits = int(np.log2(input_dim))
            if 2 ** nqubits != input_dim:
                raise QiskitError("Input is not an n-qubit Chi matrix.")
        super().__init__('Chi', chi_mat, input_dim, output_dim)

    @property
    def _bipartite_shape(self):
        """Return the shape for bipartite matrix"""
        return (self._input_dim, self._output_dim,
                self._input_dim, self._output_dim)

    def evolve(self, state):
        """Apply the channel to a quantum state.

        Args:
            state (quantum_state like): A statevector or density matrix.

        Returns:
            A density matrix.
        """
        return Choi(self).evolve(state)

    def is_cptp(self):
        """Test if channel completely-positive and trace preserving (CPTP)"""
        # We convert to the Choi representation to check if CPTP
        tmp = Choi(self)
        return tmp.is_cptp()

    def conjugate(self, inplace=False):
        """Return the conjugate channel"""
        # Since conjugation is basis dependent we transform
        # to the Choi representation to compute the
        # conjugate channel
        tmp = Chi(Choi(self).conjugate(inplace=True))
        if inplace:
            self._data = tmp._data
            self._input_dim = tmp._input_dim
            self._output_dim = tmp._output_dim
        return tmp

    def transpose(self, inplace=False):
        """Return the transpose channel"""
        # Since conjugation is basis dependent we transform
        # to the Choi representation to compute the
        # conjugate channel
        tmp = Chi(Choi(self).transpose(inplace=True))
        if inplace:
            self._data = tmp._data
            self._input_dim = tmp._input_dim
            self._output_dim = tmp._output_dim
        return tmp

    def compose(self, other, inplace=False, front=False):
        """Return Choi for the composition channel B(A(input))

        Args:
            other (QuantumChannel): a quantum channel
            inplace (bool): If True modify the current object inplace [default: False]
            front (bool): If True compose in reverse order A(B(input)) [default: False]

        Returns:
            Chi: The Chi representation for the composition channel.

        Raises:
            QiskitError: if other is not a QuantumChannel object or if the
            dimensions don't match.
        """
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        # Check dimensions match up
        if front and self._input_dim != other._output_dim:
            raise QiskitError('input_dim of self must match output_dim of other')
        if not front and self._output_dim != other._input_dim:
            raise QiskitError('input_dim of other must match output_dim of self')
        # Since we cannot directly add two channels in the Chi
        # representation we convert to the Choi representation
        tmp = Chi(Choi(self).compose(other, inplace=True, front=front))
        if inplace:
            self._data = tmp._data
            self._input_dim = tmp._input_dim
            self._output_dim = tmp._output_dim
            return self
        return tmp

    def tensor(self, other, inplace=False, front=False):
        """Return Chi for the tensor product channel.

        Args:
            other (Chi): A Choi channel
            inplace (bool): If True modify the current object inplace [default: False]
            front (bool): If False return (other ⊗ self),
                          if True return (self ⊗ other) [Default: False]
        Returns:
            Chi: The Chi for the composition channel.

        Raises:
            QiskitError: if b is not a Chi object
        """
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        if not isinstance(other, Chi):
            other = Chi(other)
        # Combined channel dimensions
        a_in, a_out = self.dims
        b_in, b_out = other.dims
        input_dim = a_in * b_in
        output_dim = a_out * b_out
        if front:
            data = np.kron(self._data, other.data)
        else:
            data = np.kron(other.data, self._data)
        if inplace:
            self._data = data
            self._input_dim = input_dim
            self._output_dim = output_dim
            return self
        # return new object
        return Chi(data, input_dim, output_dim)

    def add(self, other, inplace=False):
        """Add another channel"""
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        if self.dims != other.dims:
            raise QiskitError("Channel dimensions are not equal")
        if not isinstance(other, Chi):
            other = Chi(other)
        if inplace:
            self._data += other._data
            return self
        input_dim, output_dim = self.dims
        return Chi(self._data + other.data, input_dim, output_dim)

    def subtract(self, other, inplace=False):
        """Subtract another channel"""
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        if self.dims != other.dims:
            raise QiskitError("Channel dimensions are not equal")
        if not isinstance(other, Chi):
            other = Chi(other)
        if inplace:
            self._data -= other.data
            return self
        input_dim, output_dim = self.dims
        return Chi(self._data - other.data, input_dim, output_dim)

    def multiply(self, other, inplace=False):
        """Multiple by a scalar"""
        if not isinstance(other, Number):
            raise QiskitError("Not a number")
        if inplace:
            self._data *= other
            return self
        input_dim, output_dim = self.dims
        return Chi(other * self._data, input_dim, output_dim)
