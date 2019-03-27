# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
Chi-matrix representation of a Quantum Channel.


This is the matrix χ such that:

    E(ρ) = sum_{i, j} χ_{i,j} P_i.ρ.P_j^dagger

where [P_i, i=0,...4^{n-1}] is the n-qubit Pauli basis in lexicographic order.

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
from .transformations import _to_chi


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
                raise QiskitError(
                    "Cannot convert to Chi-matrix: input_dim " +
                    "({}) != output_dim ({})".format(input_dim, output_dim))
            chi_mat = _to_chi(data.rep, data._data, input_dim, output_dim)
        else:
            chi_mat = np.array(data, dtype=complex)
            # Determine input and output dimensions
            dim_l, dim_r = chi_mat.shape
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
                    "Invalid input and output dimension for Chi-matrix input.")
            nqubits = int(np.log2(input_dim))
            if 2**nqubits != input_dim:
                raise QiskitError("Input is not an n-qubit Chi matrix.")
        super().__init__('Chi', chi_mat, input_dim, output_dim)

    @property
    def _bipartite_shape(self):
        """Return the shape for bipartite matrix"""
        return (self._input_dim, self._output_dim, self._input_dim,
                self._output_dim)

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
        return Choi(self)._evolve(state)

    def conjugate(self, inplace=False):
        """Return the conjugate of the  QuantumChannel.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Chi: the conjugate of the quantum channel as a Chi object.
        """
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
        """Return the transpose of the QuantumChannel.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Chi: the transpose of the quantum channel as a Chi object.
        """
        # Since conjugation is basis dependent we transform
        # to the Choi representation to compute the
        # conjugate channel
        tmp = Chi(Choi(self).transpose(inplace=True))
        if inplace:
            self._data = tmp._data
            self._input_dim = tmp._input_dim
            self._output_dim = tmp._output_dim
        return tmp

    def adjoint(self, inplace=False):
        """Return the adjoint of the QuantumChannel.

        Args:
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Chi: the adjoint of the quantum channel as a Chi object.
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
            Chi: The composition channel as a Chi object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('other is not a QuantumChannel subclass')
        # Check dimensions match up
        if front and self._input_dim != other._output_dim:
            raise QiskitError(
                'input_dim of self must match output_dim of other')
        if not front and self._output_dim != other._input_dim:
            raise QiskitError(
                'input_dim of other must match output_dim of self')
        # Since we cannot directly add two channels in the Chi
        # representation we convert to the Choi representation
        tmp = Chi(Choi(self).compose(other, inplace=True, front=front))
        if inplace:
            self._data = tmp._data
            self._input_dim = tmp._input_dim
            self._output_dim = tmp._output_dim
            return self
        return tmp

    def power(self, n, inplace=False):
        """Return the compose of a QuantumChannel with itself n times.

        Args:
            n (int): the number of times to compose with self (n>0).
            inplace (bool): If True modify the current object inplace
                            [Default: False]

        Returns:
            Chi: the n-times composition channel as a Chi object.

        Raises:
            QiskitError: if the input and output dimensions of the
            QuantumChannel are not equal, or the power is not a positive
            integer.
        """
        return super().power(n, inplace=inplace)

    def tensor(self, other, inplace=False):
        """Return the tensor product channel self ⊗ other.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Chi: the tensor product channel self ⊗ other as a Chi object.

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
            Chi: the tensor product channel other ⊗ self as a Chi object.

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
            Chi: the linear addition self + other as a Chi object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('other is not a QuantumChannel subclass')
        if self.dims != other.dims:
            raise QiskitError("other QuantumChannel dimensions are not equal")
        if not isinstance(other, Chi):
            other = Chi(other)
        if inplace:
            self._data += other._data
            return self
        input_dim, output_dim = self.dims
        return Chi(self._data + other.data, input_dim, output_dim)

    def subtract(self, other, inplace=False):
        """Return the QuantumChannel self - other.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Chi: the linear subtraction self - other as Chi object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('other is not a QuantumChannel subclass')
        if self.dims != other.dims:
            raise QiskitError("other QuantumChannel dimensions are not equal")
        if not isinstance(other, Chi):
            other = Chi(other)
        if inplace:
            self._data -= other.data
            return self
        input_dim, output_dim = self.dims
        return Chi(self._data - other.data, input_dim, output_dim)

    def multiply(self, other, inplace=False):
        """Return the QuantumChannel self + other.

        Args:
            other (complex): a complex number
            inplace (bool): If True modify the current object inplace
                           [Default: False]

        Returns:
            Chi: the scalar multiplication other * self as a Chi object.

        Raises:
            QiskitError: if other is not a valid scalar.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        if inplace:
            self._data *= other
            return self
        input_dim, output_dim = self.dims
        return Chi(other * self._data, input_dim, output_dim)

    def _tensor_product(self, other, inplace=False, reverse=False):
        """Return the tensor product channel.

        Args:
            other (QuantumChannel): a quantum channel subclass
            inplace (bool): If True modify the current object inplace
                            [default: False]
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False
        Returns:
            Chi: the tensor product channel as a Chi object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass.
        """
        if not issubclass(other.__class__, QuantumChannel):
            raise QiskitError('other is not a QuantumChannel subclass')
        if not isinstance(other, Chi):
            other = Chi(other)
        # Combined channel dimensions
        a_in, a_out = self.dims
        b_in, b_out = other.dims
        input_dim = a_in * b_in
        output_dim = a_out * b_out
        if reverse:
            data = np.kron(other.data, self._data)
        else:
            data = np.kron(self._data, other.data)
        if inplace:
            self._data = data
            self._input_dim = input_dim
            self._output_dim = output_dim
            return self
        # return new object
        return Chi(data, input_dim, output_dim)
