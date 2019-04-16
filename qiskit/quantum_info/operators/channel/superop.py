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
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.transformations import _to_superop
from qiskit.quantum_info.operators.channel.transformations import _bipartite_tensor


class SuperOp(QuantumChannel):
    """Superoperator representation of a quantum channel"""

    def __init__(self, data, input_dims=None, output_dims=None):
        """Initialize a SuperOp quantum channel operator."""
        if issubclass(data.__class__, BaseOperator):
            # If not a channel we use `to_operator` method to get
            # the unitary-representation matrix for input
            if not issubclass(data.__class__, QuantumChannel):
                data = data.to_operator()
            input_dim, output_dim = data.dim
            super_mat = _to_superop(data.rep, data._data, input_dim,
                                    output_dim)
            if input_dims is None:
                input_dims = data.input_dims()
            if output_dims is None:
                output_dims = data.output_dims()
        elif isinstance(data, (list, np.ndarray)):
            # We initialize directly from superoperator matrix
            super_mat = np.array(data, dtype=complex)
            # Determine total input and output dimensions
            dout, din = super_mat.shape
            input_dim = int(np.sqrt(din))
            output_dim = int(np.sqrt(dout))
            if output_dim**2 != dout or input_dim**2 != din:
                raise QiskitError("Invalid shape for SuperOp matrix.")
        else:
            raise QiskitError("Invalid input data format for SuperOp")
        # Check and format input and output dimensions
        input_dims = self._automatic_dims(input_dims, input_dim)
        output_dims = self._automatic_dims(output_dims, output_dim)
        super().__init__('SuperOp', super_mat, input_dims, output_dims)

    @property
    def _shape(self):
        """Return the tensor shape of the superopertor matrix"""
        return 2 * tuple(reversed(self.output_dims())) + 2 * tuple(
            reversed(self.input_dims()))

    @property
    def _bipartite_shape(self):
        """Return the shape for bipartite matrix"""
        return (self._output_dim, self._output_dim, self._input_dim,
                self._input_dim)

    def conjugate(self):
        """Return the conjugate of the QuantumChannel."""
        return SuperOp(
            np.conj(self._data), self.input_dims(), self.output_dims())

    def transpose(self):
        """Return the transpose of the QuantumChannel."""
        return SuperOp(
            np.transpose(self._data),
            input_dims=self.output_dims(),
            output_dims=self.input_dims())

    def compose(self, other, qubits=None, front=False):
        """Return the composition channel self∘other.

        Args:
            other (QuantumChannel): a quantum channel.
            qubits (list): a list of subsystem positions to compose other on.
            front (bool): If False compose in standard order other(self(input))
                          otherwise compose in reverse order self(other(input))
                          [default: False]

        Returns:
            SuperOp: The composition channel as a SuperOp object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        # Convert other to SuperOp
        if not isinstance(other, SuperOp):
            other = SuperOp(other)
        # Check dimensions are compatible
        if front and self.input_dims(qubits=qubits) != other.output_dims():
            raise QiskitError(
                'output_dims of other must match subsystem input_dims')
        if not front and self.output_dims(qubits=qubits) != other.input_dims():
            raise QiskitError(
                'input_dims of other must match subsystem output_dims')

        # Full composition of superoperators
        if qubits is None:
            if front:
                # Composition A(B(input))
                return SuperOp(
                    np.dot(self._data, other.data),
                    input_dims=other.input_dims(),
                    output_dims=self.output_dims())
            # Composition B(A(input))
            return SuperOp(
                np.dot(other.data, self._data),
                input_dims=self.input_dims(),
                output_dims=other.output_dims())
        # Composition on subsystem
        return self._compose_subsystem(other, qubits, front)

    def power(self, n):
        """Return the compose of a QuantumChannel with itself n times.

        Args:
            n (int): compute the matrix power of the superoperator matrix.

        Returns:
            SuperOp: the n-times composition channel as a SuperOp object.

        Raises:
            QiskitError: if the input and output dimensions of the
            QuantumChannel are not equal, or the power is not an integer.
        """
        if not isinstance(n, (int, np.integer)):
            raise QiskitError("Can only power with integer powers.")
        if self._input_dim != self._output_dim:
            raise QiskitError("Can only power with input_dim = output_dim.")
        # Override base class power so we can implement more efficiently
        # using Numpy.matrix_power
        return SuperOp(
            np.linalg.matrix_power(self._data, n), self.input_dims(),
            self.output_dims())

    def tensor(self, other):
        """Return the tensor product channel self ⊗ other.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            SuperOp: the tensor product channel self ⊗ other as a SuperOp
            object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        return self._tensor_product(other, reverse=False)

    def expand(self, other):
        """Return the tensor product channel other ⊗ self.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            SuperOp: the tensor product channel other ⊗ self as a SuperOp
            object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        return self._tensor_product(other, reverse=True)

    def add(self, other):
        """Return the QuantumChannel self + other.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            SuperOp: the linear addition self + other as a SuperOp object.

        Raises:
            QiskitError: if other cannot be converted to a channel or
            has incompatible dimensions.
        """
        # Convert other to SuperOp
        if not isinstance(other, SuperOp):
            other = SuperOp(other)
        if self.dim != other.dim:
            raise QiskitError("other QuantumChannel dimensions are not equal")
        return SuperOp(self._data + other.data, self.input_dims(),
                       self.output_dims())

    def subtract(self, other):
        """Return the QuantumChannel self - other.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            SuperOp: the linear subtraction self - other as SuperOp object.

        Raises:
            QiskitError: if other cannot be converted to a channel or
            has incompatible dimensions.
        """
        # Convert other to SuperOp
        if not isinstance(other, SuperOp):
            other = SuperOp(other)
        if self.dim != other.dim:
            raise QiskitError("other QuantumChannel dimensions are not equal")
        return SuperOp(self._data - other.data, self.input_dims(),
                       self.output_dims())

    def multiply(self, other):
        """Return the QuantumChannel self + other.

        Args:
            other (complex): a complex number.

        Returns:
            SuperOp: the scalar multiplication other * self as a SuperOp object.

        Raises:
            QiskitError: if other is not a valid scalar.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        return SuperOp(other * self._data, self.input_dims(),
                       self.output_dims())

    def _evolve(self, state, qubits=None):
        """Evolve a quantum state by the QuantumChannel.

        Args:
            state (QuantumState): The input statevector or density matrix.
            qubits (list): a list of QuantumState subsystem positions to apply
                           the operator on.

        Returns:
            DensityMatrix: the output quantum state as a density matrix.

        Raises:
            QiskitError: if the operator dimension does not match the
            specified QuantumState subsystem dimensions.
        """
        state = self._format_state(state, density_matrix=True)
        if qubits is None:
            if state.shape[0] != self._input_dim:
                raise QiskitError(
                    "QuantumChannel input dimension is not equal to state dimension."
                )
            shape_in = self._input_dim * self._input_dim
            shape_out = (self._output_dim, self._output_dim)
            # Return evolved density matrix
            return np.reshape(
                np.dot(self._data, np.reshape(state, shape_in, order='F')),
                shape_out,
                order='F')
        # Subsystem evolution
        return self._evolve_subsystem(state, qubits)

    def _compose_subsystem(self, other, qubits, front=False):
        """Return the composition channel."""
        # Compute tensor contraction indices from qubits
        input_dims = list(self.input_dims())
        output_dims = list(self.output_dims())
        if front:
            num_indices = len(self.input_dims())
            shift = 2 * len(self.output_dims())
            right_mul = True
            for pos, qubit in enumerate(qubits):
                input_dims[qubit] = other._input_dims[pos]
        else:
            num_indices = len(self.output_dims())
            shift = 0
            right_mul = False
            for pos, qubit in enumerate(qubits):
                output_dims[qubit] = other._output_dims[pos]
        # Reshape current matrix
        # Note that we must reverse the subsystem dimension order as
        # qubit 0 corresponds to the right-most position in the tensor
        # product, which is the last tensor wire index.
        tensor = np.reshape(self.data, self._shape)
        mat = np.reshape(other.data, other._shape)
        # Add first set of indicies
        indices = [2 * num_indices - 1 - qubit for qubit in qubits
                   ] + [num_indices - 1 - qubit for qubit in qubits]
        final_shape = [np.product(output_dims)**2, np.product(input_dims)**2]
        data = np.reshape(
            self._einsum_matmul(tensor, mat, indices, shift, right_mul),
            final_shape)
        return SuperOp(data, input_dims, output_dims)

    def _evolve_subsystem(self, state, qubits):
        """Evolve a quantum state by the operator.

        Args:
            state (QuantumState): The input statevector or density matrix.
            qubits (list): a list of QuantumState subsystem positions to apply
                           the operator on.

        Returns:
            QuantumState: the output quantum state.

        Raises:
            QiskitError: if the operator dimension does not match the
            specified QuantumState subsystem dimensions.
        """
        mat = np.reshape(self.data, self._shape)
        # Hack to assume state is a N-qubit state until a proper class for states
        # is in place
        state_size = len(state)
        state_dims = self._automatic_dims(None, state_size)
        if self.input_dims() != len(qubits) * (2, ):
            raise QiskitError(
                "Channel input dimensions are not compatible with state subsystem dimensions."
            )
        # Return evolved density matrix
        tensor = np.reshape(state, 2 * state_dims)
        num_inidices = len(state_dims)
        indices = [num_inidices - 1 - qubit for qubit in qubits
                   ] + [2 * num_inidices - 1 - qubit for qubit in qubits]
        tensor = self._einsum_matmul(tensor, mat, indices)
        return np.reshape(tensor, [state_size, state_size])

    def _tensor_product(self, other, reverse=False):
        """Return the tensor product channel.

        Args:
            other (QuantumChannel): a quantum channel.
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False
        Returns:
            SuperOp: the tensor product channel as a SuperOp object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        # Convert other to SuperOp
        if not isinstance(other, SuperOp):
            other = SuperOp(other)

        if reverse:
            input_dims = self.input_dims() + other.input_dims()
            output_dims = self.output_dims() + other.output_dims()
            data = _bipartite_tensor(
                other.data,
                self._data,
                shape1=other._bipartite_shape,
                shape2=self._bipartite_shape)
        else:
            input_dims = other.input_dims() + self.input_dims()
            output_dims = other.output_dims() + self.output_dims()
            data = _bipartite_tensor(
                self._data,
                other.data,
                shape1=self._bipartite_shape,
                shape2=other._bipartite_shape)
        return SuperOp(data, input_dims, output_dims)
