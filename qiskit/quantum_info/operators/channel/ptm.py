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
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.transformations import _to_ptm


class PTM(QuantumChannel):
    """Pauli transfer matrix (PTM) representation of a quantum channel.

    The PTM is the Pauli-basis representation of the PTM.
    """

    def __init__(self, data, input_dims=None, output_dims=None):
        """Initialize a PTM quantum channel operator."""
        if issubclass(data.__class__, BaseOperator):
            # If not a channel we use `to_operator` method to get
            # the unitary-representation matrix for input
            if not issubclass(data.__class__, QuantumChannel):
                data = data.to_operator()
            input_dim, output_dim = data.dim
            ptm = _to_ptm(data.rep, data._data, input_dim, output_dim)
            if input_dims is None:
                input_dims = data.input_dims()
            if output_dims is None:
                output_dims = data.output_dims()
        elif isinstance(data, (list, np.ndarray)):
            # Should we force this to be real?
            ptm = np.array(data, dtype=complex)
            # Determine input and output dimensions
            dout, din = ptm.shape
            if input_dims:
                input_dim = np.product(input_dims)
            else:
                input_dim = int(np.sqrt(din))
            if output_dims:
                output_dim = np.product(input_dims)
            else:
                output_dim = int(np.sqrt(dout))
            if output_dim**2 != dout or input_dim**2 != din or input_dim != output_dim:
                raise QiskitError("Invalid shape for PTM matrix.")
        else:
            raise QiskitError("Invalid input data format for PTM")

        nqubits = int(np.log2(input_dim))
        if 2**nqubits != input_dim:
            raise QiskitError("Input is not an n-qubit Pauli transfer matrix.")
        # Check and format input and output dimensions
        input_dims = self._automatic_dims(input_dims, input_dim)
        output_dims = self._automatic_dims(output_dims, output_dim)
        super().__init__('PTM', ptm, input_dims, output_dims)

    @property
    def _bipartite_shape(self):
        """Return the shape for bipartite matrix"""
        return (self._output_dim, self._output_dim, self._input_dim,
                self._input_dim)

    def conjugate(self):
        """Return the conjugate of the QuantumChannel."""
        # Since conjugation is basis dependent we transform
        # to the SuperOp representation to compute the
        # conjugate channel
        return PTM(SuperOp(self).conjugate())

    def transpose(self):
        """Return the transpose of the QuantumChannel."""
        # Since conjugation is basis dependent we transform
        # to the SuperOp representation to compute the
        # conjugate channel
        return PTM(SuperOp(self).transpose())

    def compose(self, other, qubits=None, front=False):
        """Return the composition channel self∘other.

        Args:
            other (QuantumChannel): a quantum channel.
            qubits (list): a list of subsystem positions to compose other on.
            front (bool): If False compose in standard order other(self(input))
                          otherwise compose in reverse order self(other(input))
                          [default: False]

        Returns:
            PTM: The composition channel as a PTM object.

        Raises:
            QiskitError: if other cannot be converted to a channel or
            has incompatible dimensions.
        """
        if qubits is not None:
            # TODO
            raise QiskitError("NOT IMPLEMENTED: subsystem composition.")

        # Convert other to PTM
        if not isinstance(other, PTM):
            other = PTM(other)
        # Check dimensions match up
        if front and self._input_dim != other._output_dim:
            raise QiskitError(
                'input_dim of self must match output_dim of other')
        if not front and self._output_dim != other._input_dim:
            raise QiskitError(
                'input_dim of other must match output_dim of self')
        if front:
            # Composition A(B(input))
            input_dim = other._input_dim
            output_dim = self._output_dim
            return PTM(np.dot(self._data, other.data), input_dim, output_dim)
        # Composition B(A(input))
        input_dim = self._input_dim
        output_dim = other._output_dim
        return PTM(np.dot(other.data, self._data), input_dim, output_dim)

    def power(self, n):
        """The matrix power of the channel.

        Args:
            n (int): compute the matrix power of the superoperator matrix.

        Returns:
            PTM: the matrix power of the SuperOp converted to a PTM channel.

        Raises:
            QiskitError: if the input and output dimensions of the
            QuantumChannel are not equal, or the power is not an integer.
        """
        if n > 0:
            return super().power(n)
        return PTM(SuperOp(self).power(n))

    def tensor(self, other):
        """Return the tensor product channel self ⊗ other.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            PTM: the tensor product channel self ⊗ other as a PTM object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        return self._tensor_product(other, reverse=False)

    def expand(self, other):
        """Return the tensor product channel other ⊗ self.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            PTM: the tensor product channel other ⊗ self as a PTM object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        return self._tensor_product(other, reverse=True)

    def add(self, other):
        """Return the QuantumChannel self + other.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            PTM: the linear addition self + other as a PTM object.

        Raises:
            QiskitError: if other cannot be converted to a channel or
            has incompatible dimensions.
        """
        if not isinstance(other, PTM):
            other = PTM(other)
        if self.dim != other.dim:
            raise QiskitError("other QuantumChannel dimensions are not equal")
        return PTM(self._data + other.data, self._input_dims,
                   self._output_dims)

    def subtract(self, other):
        """Return the QuantumChannel self - other.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            PTM: the linear subtraction self - other as PTM object.

        Raises:
            QiskitError: if other cannot be converted to a channel or
            has incompatible dimensions.
        """
        if not isinstance(other, PTM):
            other = PTM(other)
        if self.dim != other.dim:
            raise QiskitError("other QuantumChannel dimensions are not equal")
        return PTM(self._data - other.data, self._input_dims,
                   self._output_dims)

    def multiply(self, other):
        """Return the QuantumChannel self + other.

        Args:
            other (complex): a complex number.

        Returns:
            PTM: the scalar multiplication other * self as a PTM object.

        Raises:
            QiskitError: if other is not a valid scalar.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        return PTM(other * self._data, self._input_dims, self._output_dims)

    def _evolve(self, state, qubits=None):
        """Evolve a quantum state by the QuantumChannel.

        Args:
            state (QuantumState): The input statevector or density matrix.
            qubits (list): a list of QuantumState subsystem positions to apply
                           the operator on.

        Returns:
            DensityMatrix: the output quantum state as a density matrix.
        """
        return SuperOp(self)._evolve(state, qubits)

    def _tensor_product(self, other, reverse=False):
        """Return the tensor product channel.

        Args:
            other (QuantumChannel): a quantum channel.
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False
        Returns:
            PTM: the tensor product channel as a PTM object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        if not isinstance(other, PTM):
            other = PTM(other)
        if reverse:
            input_dims = self.input_dims() + other.input_dims()
            output_dims = self.output_dims() + other.output_dims()
            data = np.kron(other.data, self._data)
        else:
            input_dims = other.input_dims() + self.input_dims()
            output_dims = other.output_dims() + self.output_dims()
            data = np.kron(self._data, other.data)
        return PTM(data, input_dims, output_dims)
