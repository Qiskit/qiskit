# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
Superoperator representation of a Quantum Channel.


For a quantum channel E, the superoperator is defined as the matrix S such that

    |E(ρ)⟩⟩ = S|ρ⟩⟩

where |A⟩⟩ denotes the column stacking vectorization of a matrix A.

See [1] for further details.

References:
    [1] C.J. Wood, J.D. Biamonte, D.G. Cory, Quant. Inf. Comp. 15, 0579-0811 (2015)
        Open access: arXiv:1111.6950 [quant-ph]
"""

from numbers import Number
import numpy as np

from qiskit.qiskiterror import QiskitError
from .basechannel import QuantumChannel
from .transformations import _to_superop, _bipartite_tensor
from .choi import Choi


class SuperOp(QuantumChannel):
    """Superoperator representation of a quantum channel"""

    def __init__(self, data, input_dim=None, output_dim=None):
        # Check if input is a quantum channel object
        # If so we disregard the dimension kwargs
        if issubclass(data.__class__, QuantumChannel):
            input_dim, output_dim = data.dims
            super_mat = _to_superop(data.rep, data._data, input_dim,
                                    output_dim)
        else:
            # We initialize directly from superoperator matrix
            super_mat = np.array(data, dtype=complex)
            # Determine input and output dimensions
            dout, din = super_mat.shape
            if output_dim is None:
                output_dim = int(np.sqrt(dout))
            if input_dim is None:
                input_dim = int(np.sqrt(din))
            # Check dimensions
            if output_dim**2 != dout or input_dim**2 != din:
                raise QiskitError(
                    "Invalid input and output dimension for superoperator input."
                )
        super().__init__('SuperOp', super_mat, input_dim, output_dim)

    @property
    def _bipartite_shape(self):
        """Return the shape for bipartite matrix"""
        return (self._output_dim, self._output_dim, self._input_dim,
                self._input_dim)

    def is_cptp(self):
        """Return True if completely-positive trace-preserving."""
        # We convert to the Choi representation to check if CPTP
        tmp = Choi(self)
        return tmp.is_cptp()

    def _evolve(self, state):
        """Evolve a quantum state by the QuantumChannel.

        Args:
            state (QuantumState): The input statevector or density matrix.

        Returns:
            DensityMatrix: the output quantum state as a density matrix.
        """
        state = self._format_density_matrix(self._check_state(state))
        shape_in = self._input_dim * self._input_dim
        shape_out = (self._output_dim, self._output_dim)
        return np.reshape(
            np.dot(self._data, np.reshape(state, shape_in, order='F')),
            shape_out,
            order='F')

    def conjugate(self, inplace=False):
        """Return the conjugate of the  QuantumChannel.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            SuperOp: the conjugate of the quantum channel as a SuperOp object.
        """
        if inplace:
            np.conjugate(self._data, out=self._data)
            return self
        return SuperOp(np.conj(self._data), self._input_dim, self._output_dim)

    def transpose(self, inplace=False):
        """Return the transpose of the QuantumChannel.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            SuperOp: the transpose of the quantum channel as a SuperOp object.
        """
        # Swaps input and output dimensions
        output_dim = self._input_dim
        input_dim = self._output_dim
        if inplace:
            self._data = np.transpose(self._data)
            self._input_dim = input_dim
            self._output_dim = output_dim
            return self
        return SuperOp(np.transpose(self._data), input_dim, output_dim)

    def adjoint(self, inplace=False):
        """Return the adjoint of the QuantumChannel.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            SuperOp: the adjoint of the quantum channel as a SuperOp object.
        """
        return super().adjoint(inplace=inplace)

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
            SuperOp: The composition channel as a SuperOp object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
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
        # Convert other to SuperOp
        if not isinstance(other, SuperOp):
            other = SuperOp(other)

        if front:
            # Composition A(B(input))
            input_dim = other._input_dim
            output_dim = self._output_dim
            if inplace:
                if self.dims == other.dims:
                    np.dot(self._data, other.data, out=self._data)
                else:
                    self._data = np.dot(self._data, other.data)
                self._input_dim = input_dim
                self._output_dim = output_dim
                return self
            return SuperOp(
                np.dot(self._data, other.data), input_dim, output_dim)
        # Composition B(A(input))
        input_dim = self._input_dim
        output_dim = other._output_dim
        if inplace:
            if self.dims == other.dims:
                np.dot(other.data, self._data, out=self._data)
            else:
                self._data = np.dot(other.data, self._data)
            self._input_dim = input_dim
            self._output_dim = output_dim
            return self
        return SuperOp(np.dot(other.data, self._data), input_dim, output_dim)

    def power(self, n, inplace=False):
        """Return the compose of a QuantumChannel with itself n times.

        Args:
            n (int): the number of times to compose with self (n>0).
            inplace (bool): If True modify the current object inplace
                            [Default: False]

        Returns:
            SuperOp: the n-times composition channel as a SuperOp object.

        Raises:
            QiskitError: if the input and output dimensions of the
            QuantumChannel are not equal, or the power is not a positive
            integer.
        """
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
        return SuperOp(np.linalg.matrix_power(self._data, n), *self.dims)

    def tensor(self, other, inplace=False):
        """Return the tensor product channel self ⊗ other.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            SuperOp: the tensor product channel self ⊗ other as a SuperOp
            object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass.
        """
        return self._tensor_product(other, inplace=inplace, reverse=False)

    def expand(self, other, inplace=False):
        """Return the tensor product channel other ⊗ self.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            SuperOp: the tensor product channel other ⊗ self as a SuperOp
            object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass.
        """
        return self._tensor_product(other, inplace=inplace, reverse=True)

    def add(self, other, inplace=False):
        """Return the QuantumChannel self + other.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            SuperOp: the linear addition self + other as a SuperOp object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('other is not a QuantumChannel subclass')
        if self.dims != other.dims:
            raise QiskitError("other QuantumChannel dimensions are not equal")
        if not isinstance(other, SuperOp):
            other = SuperOp(other)

        if inplace:
            self._data += other._data
            return self
        input_dim, output_dim = self.dims
        return SuperOp(self._data + other.data, input_dim, output_dim)

    def subtract(self, other, inplace=False):
        """Return the QuantumChannel self - other.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            SuperOp: the linear subtraction self - other as SuperOp object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('other is not a QuantumChannel subclass')
        if self.dims != other.dims:
            raise QiskitError("other QuantumChannel dimensions are not equal")
        if not isinstance(other, SuperOp):
            other = SuperOp(other)
        if inplace:
            self._data -= other.data
            return self
        input_dim, output_dim = self.dims
        return SuperOp(self._data - other.data, input_dim, output_dim)

    def multiply(self, other, inplace=False):
        """Return the QuantumChannel self + other.

        Args:
            other (complex): a complex number
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            SuperOp: the scalar multiplication other * self as a SuperOp object.

        Raises:
            QiskitError: if other is not a valid scalar.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        if inplace:
            self._data *= other
            return self
        input_dim, output_dim = self.dims
        return SuperOp(other * self._data, input_dim, output_dim)

    def _tensor_product(self, other, inplace=False, reverse=False):
        """Return the tensor product channel.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                            [default: False]
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False
        Returns:
            SuperOp: the tensor product channel as a SuperOp object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass.
        """
        # Convert other to SuperOp
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('other is not a QuantumChannel subclass')
        if not isinstance(other, SuperOp):
            other = SuperOp(other)

        # Reshuffle indicies
        a_in, a_out = self.dims
        b_in, b_out = other.dims

        # Combined channel dimensions
        input_dim = a_in * b_in
        output_dim = a_out * b_out

        if reverse:
            data = _bipartite_tensor(other.data, self._data,
                                     shape1=other._bipartite_shape,
                                     shape2=self._bipartite_shape)
        else:
            data = _bipartite_tensor(self._data, other.data,
                                     shape1=self._bipartite_shape,
                                     shape2=other._bipartite_shape)
        if inplace:
            self._data = data
            self._input_dim = input_dim
            self._output_dim = output_dim
            return self
        # return new object
        return SuperOp(data, input_dim, output_dim)
