# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
Pauli Transfer Matrix (PTM) representation of a Quantum Channel.

The PTM is the n-qubit superoperator defined with respect to vectorization in
the Pauli basis. For a quantum channel E, the PTM is defined by

    PTM_{i,j} = Tr[P_i.E(P_j)]

where [P_i, i=0,...4^{n-1}] is the n-qubit Pauli basis in lexicographic order.

Evolution is given by

    |E(ρ)⟩⟩_p = PTM|ρ⟩⟩_p

where |A⟩⟩_p denotes vectorization in the Pauli basis: ⟨i|A⟩⟩_p = Tr[P_i.A]

See [1] for further details.

References:
    [1] C.J. Wood, J.D. Biamonte, D.G. Cory, Quant. Inf. Comp. 15, 0579-0811 (2015)
        Open access: arXiv:1111.6950 [quant-ph]
"""

from numbers import Number

import numpy as np

from qiskit.qiskiterror import QiskitError
from .basechannel import QuantumChannel
from .choi import Choi
from .superop import SuperOp
from .transformations import _to_ptm


class PTM(QuantumChannel):
    """Pauli transfer matrix (PTM) representation of a quantum channel.

    The PTM is the Pauli-basis representation of the PTM.
    """

    def __init__(self, data, input_dim=None, output_dim=None):
        # Check if input is a quantum channel object
        # If so we disregard the dimension kwargs
        if issubclass(data.__class__, QuantumChannel):
            input_dim, output_dim = data.dims
            ptm = _to_ptm(data.rep, data._data, input_dim, output_dim)
        else:
            # Should we force this to be real?
            ptm = np.array(data, dtype=complex)
            # Determine input and output dimensions
            dout, din = ptm.shape
            if output_dim is None:
                output_dim = int(np.sqrt(dout))
            if input_dim is None:
                input_dim = int(np.sqrt(din))
            # Check dimensions
            if output_dim**2 != dout or input_dim**2 != din or input_dim != output_dim:
                raise QiskitError(
                    "Invalid input and output dimension for PTM input.")
            nqubits = int(np.log2(input_dim))
            if 2**nqubits != input_dim:
                raise QiskitError(
                    "Input is not an n-qubit Pauli transfer matrix.")
        super().__init__('PTM', ptm, input_dim, output_dim)

    @property
    def _bipartite_shape(self):
        """Return the shape for bipartite matrix"""
        return (self._output_dim, self._output_dim, self._input_dim,
                self._input_dim)

    def evolve(self, state):
        """Apply the channel to a quantum state.

        Args:
            state (quantum_state like): A statevector or density matrix.

        Returns:
            DensityMatrix: the output quantum state as a density matrix.
        """
        return SuperOp(self).evolve(state)

    def is_cptp(self):
        """Test if channel completely-positive and trace preserving (CPTP)"""
        # We convert to the Choi representation to check if CPTP
        tmp = Choi(self)
        return tmp.is_cptp()

    def conjugate(self, inplace=False):
        """Return the conjugate channel"""
        # Since conjugation is basis dependent we transform
        # to the SuperOp representation to compute the
        # conjugate channel
        tmp = PTM(SuperOp(self).conjugate(inplace=True))
        if inplace:
            self._data = tmp._data
            self._input_dim = tmp._input_dim
            self._output_dim = tmp._output_dim
        return tmp

    def transpose(self, inplace=False):
        """Return the transpose channel"""
        # Since conjugation is basis dependent we transform
        # to the SuperOp representation to compute the
        # conjugate channel
        tmp = PTM(SuperOp(self).transpose(inplace=True))
        if inplace:
            self._data = tmp._data
            self._input_dim = tmp._input_dim
            self._output_dim = tmp._output_dim
        return tmp

    def compose(self, other, inplace=False, front=False):
        """Return PTM for the composition channel B(A(input))

        Args:
            other (QuantumChannel): A quantum channel representation object
            inplace (bool): If True modify the current object inplace [default: False]
            front (bool): If True compose in reverse order A(B(input)) [default: False]

        Returns:
            PTM: The PTM for the composition channel.

        Raises:
            QiskitError: if other is not a PTM object
            QiskitError: if dimensions don't match.
        """
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        # Check dimensions match up
        if front and self._input_dim != other._output_dim:
            raise QiskitError(
                'input_dim of self must match output_dim of other')
        if not front and self._output_dim != other._input_dim:
            raise QiskitError(
                'input_dim of other must match output_dim of self')
        # Convert other to PTM
        if not isinstance(other, PTM):
            other = PTM(other)

        if front:
            # Composition A(B(input))
            input_dim = other._input_dim
            output_dim = self._output_dim
            if inplace:
                np.dot(self._data, other.data, out=self._data)
                self._input_dim = input_dim
                self._output_dim = output_dim
                return self
            return PTM(np.dot(self._data, other.data), input_dim, output_dim)
        # Composition B(A(input))
        input_dim = self._input_dim
        output_dim = other._output_dim
        if inplace:
            np.dot(other.data, self._data, out=self._data)
            self._input_dim = input_dim
            self._output_dim = output_dim
            return self
        return PTM(np.dot(other.data, self._data), input_dim, output_dim)

    def tensor(self, other, inplace=False, front=False):
        """Return PTM for the tensor product channel.

        Args:
            other (QuantumChannel): A quantum channel representation object
            inplace (bool): If True modify the current object inplace [default: False]
            front (bool): If False return (other ⊗ self),
                          if True return (self ⊗ other) [Default: False]
        Returns:
            PTM: The PTM for the composition channel.

        Raises:
            QiskitError: if b is not a PTM object
        """
        # Convert other to PTM
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        if not isinstance(other, PTM):
            other = PTM(other)
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
        return PTM(data, input_dim, output_dim)

    def add(self, other, inplace=False):
        """Add another channel"""
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        if self.dims != other.dims:
            raise QiskitError("Channel dimensions are not equal")
        if not isinstance(other, PTM):
            other = PTM(other)
        if inplace:
            self._data += other._data
            return self
        input_dim, output_dim = self.dims
        return PTM(self._data + other.data, input_dim, output_dim)

    def subtract(self, other, inplace=False):
        """Subtract another PTM"""
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        if self.dims != other.dims:
            raise QiskitError("Channel dimensions are not equal")
        if not isinstance(other, PTM):
            other = PTM(other)
        if inplace:
            self._data -= other.data
            return self
        input_dim, output_dim = self.dims
        return PTM(self._data - other.data, input_dim, output_dim)

    def multiply(self, other, inplace=False):
        """Multiple by a scalar"""
        if not isinstance(other, Number):
            raise QiskitError("Not a number")
        if inplace:
            self._data *= other
            return self
        input_dim, output_dim = self.dims
        return PTM(other * self._data, input_dim, output_dim)
