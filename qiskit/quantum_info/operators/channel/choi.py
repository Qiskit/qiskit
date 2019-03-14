# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
Choi-matrix representation of a Quantum Channel.


For a quantum channel E, the Choi matrix Λ is defined by:
Λ = sum_{i,j} |i⟩⟨j|⊗E(|i⟩⟨j|)

Evolution of a density matrix with respect to the Choi-matrix is given by:

    E(ρ) = Tr_{1}[Λ.(ρ^T⊗I)]

See [1] for further details.

References:
    [1] C.J. Wood, J.D. Biamonte, D.G. Cory, Quant. Inf. Comp. 15, 0579-0811 (2015)
        Open access: arXiv:1111.6950 [quant-ph]
"""

from numbers import Number

import numpy as np

from qiskit.qiskiterror import QiskitError
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from .basechannel import QuantumChannel
from .transformations import _to_choi, _bipartite_tensor


class Choi(QuantumChannel):
    """Choi-matrix representation of a quantum channel"""

    def __init__(self, data, input_dim=None, output_dim=None):
        # Check if input is a quantum channel object
        # If so we disregard the dimension kwargs
        if issubclass(data.__class__, QuantumChannel):
            input_dim, output_dim = data.dims
            choi_mat = _to_choi(data.rep, data._data, input_dim, output_dim)
        else:
            choi_mat = np.array(data, dtype=complex)
            # Determine input and output dimensions
            dim_l, dim_r = choi_mat.shape
            if dim_l != dim_r:
                raise QiskitError('Invalid Choi-matrix input.')
            if output_dim is None and input_dim is None:
                output_dim = int(np.sqrt(dim_l))
                input_dim = dim_l // output_dim
            elif input_dim is None:
                input_dim = dim_l // output_dim
            elif output_dim is None:
                output_dim = dim_l // input_dim
            # Check dimensions
            if input_dim * output_dim != dim_l:
                raise QiskitError(
                    "Invalid input and output dimension for Choi-matrix input."
                )
        super().__init__('Choi', choi_mat, input_dim, output_dim)

    @property
    def _bipartite_shape(self):
        """Return the shape for bipartite matrix"""
        return (self._input_dim, self._output_dim, self._input_dim,
                self._output_dim)

    def evolve(self, state):
        """Apply the channel to a quantum state.

        Args:
            state (quantum_state like): A statevector or density matrix.

        Returns:
            DensityMatrix: the output quantum state as a density matrix.
        """
        state = self._format_density_matrix(self._check_state(state))
        return np.einsum('AB,AiBj->ij', state,
                         np.reshape(self._data, self._bipartite_shape))

    def conjugate(self, inplace=False):
        """Return the conjugate channel"""
        if inplace:
            np.conjugate(self._data, out=self._data)
            return self
        return Choi(np.conj(self._data), self._input_dim, self._output_dim)

    def transpose(self, inplace=False):
        """Return the transpose channel"""
        # Make bipartite matrix
        d_in, d_out = self.dims
        data = np.reshape(self._data, (d_in, d_out, d_in, d_out))
        # Swap input and output indicies on bipartite matrix
        data = np.transpose(data, (1, 0, 3, 2))
        # Transpose channel has input and output dimensions swapped
        data = np.reshape(data, (d_in * d_out, d_in * d_out))
        if inplace:
            self._data = data
            self._input_dim = d_out
            self._output_dim = d_in
            return self
        return Choi(data, d_out, d_in)

    def is_cptp(self):
        """Test if Choi-matrix is completely-positive and trace preserving (CPTP)"""
        return self._is_cp() and self._is_tp()

    def _is_cp(self):
        """Test if Choi-matrix is completely-positive (CP)"""
        # Test if Choi-matrix is Hermitian
        # This is required for eigenvalues to be real
        if not is_hermitian_matrix(self._data, atol=self.atol):
            return False
        # Check eigenvalues are all positive
        vals = np.linalg.eigvalsh(self._data)
        for v in vals:
            if v < -self.atol:
                return False
        return True

    def _is_tp(self):
        """Test if Choi-matrix is trace-preserving (TP)"""
        # Check if the partial trace is the identity matrix
        d_in, d_out = self.dims
        mat = np.trace(
            np.reshape(self._data, (d_in, d_out, d_in, d_out)),
            axis1=1,
            axis2=3)
        return is_identity_matrix(mat, atol=self.atol)

    def compose(self, other, inplace=False, front=False):
        """Return the composition channel B(A(input))

        Args:
            other (QuantumChannel): a quantum channel
            inplace (bool): If True modify the current object inplace [default: False]
            front (bool): If True compose in reverse order A(B(input)) [default: False]

        Returns:
            Choi: The Choi for the composition channel.

        Raises:
            QiskitError: if other is not a Choi object or if the
            dimensions don't match.
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
        # Convert to Choi matrix
        if not isinstance(other, Choi):
            other = Choi(other)

        if front:
            first = np.reshape(other._data, other._bipartite_shape)
            second = np.reshape(self._data, self._bipartite_shape)
            input_dim = other._input_dim
            output_dim = self._output_dim
        else:
            first = np.reshape(self._data, self._bipartite_shape)
            second = np.reshape(other._data, other._bipartite_shape)
            input_dim = self._input_dim
            output_dim = other._output_dim

        # Contract Choi matrices for composition
        data = np.reshape(
            np.einsum('iAjB,AkBl->ikjl', first, second),
            (input_dim * output_dim, input_dim * output_dim))
        if inplace:
            self._data = data
            self._input_dim = input_dim
            self._output_dim = output_dim
            return self
        return Choi(data, input_dim, output_dim)

    def tensor(self, other, inplace=False, front=False):
        """Return Choi for the tensor product channel.

        Args:
            other (Choi): A Choi channel
            inplace (bool): If True modify the current object inplace [default: False]
            front (bool): If False return (other ⊗ self),
                          if True return (self ⊗ other) [Default: False]
        Returns:
            Choi: The Choi for the composition channel.

        Raises:
            QiskitError: if b is not a Choi object
        """
        # Convert other to Choi
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        if not isinstance(other, Choi):
            other = Choi(other)

        # Reshuffle indicies
        a_in, a_out = self.dims
        b_in, b_out = other.dims

        # Combined channel dimensions
        input_dim = a_in * b_in
        output_dim = a_out * b_out

        data = _bipartite_tensor(
            self._data,
            other.data,
            front=front,
            shape1=self._bipartite_shape,
            shape2=other._bipartite_shape)
        if inplace:
            self._data = data
            self._input_dim = input_dim
            self._output_dim = output_dim
            return self
        # return new object
        return Choi(data, input_dim, output_dim)

    def add(self, other, inplace=False):
        """Add another Choi-matrix"""
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        if self.dims != other.dims:
            raise QiskitError("Channel dimensions are not equal")
        if not isinstance(other, Choi):
            other = Choi(other)
        if inplace:
            self._data += other._data
            return self
        input_dim, output_dim = self.dims
        return Choi(self._data + other.data, input_dim, output_dim)

    def subtract(self, other, inplace=False):
        """Subtract another Choi-matrix"""
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('Other is not a channel rep')
        if self.dims != other.dims:
            raise QiskitError("Channel dimensions are not equal")
        if not isinstance(other, Choi):
            other = Choi(other)
        if inplace:
            self._data -= other.data
            return self
        input_dim, output_dim = self.dims
        return Choi(self._data - other.data, input_dim, output_dim)

    def multiply(self, other, inplace=False):
        """Multiple by a scalar"""
        if not isinstance(other, Number):
            raise QiskitError("Not a number")
        if inplace:
            self._data *= other
            return self
        input_dim, output_dim = self.dims
        return Choi(other * self._data, input_dim, output_dim)
