# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.
"""
Matrix Operator class.
"""

from numbers import Number

import numpy as np

from qiskit.qiskiterror import QiskitError
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.quantum_info.operators.pauli import Pauli
from qiskit.quantum_info.operators.base_operator import BaseOperator


class Operator(BaseOperator):
    """Matrix operator class

    This represents a matrix operator `M` that acts on a statevector as: `M|v⟩`
    or on a density matrix as M.ρ.M^dagger.
    """

    def __init__(self, data, input_dims=None, output_dims=None):
        """Initialize an operator object.

        Args:
            data (BaseOperator or Numpy.array): data to initialize operator.
            input_dims (tuple): the input subsystem dimensions.
                                [Default: None]
            output_dims (tuple): the output subsystem dimensions.
                                 [Default: None]

        Raises:
            QiskitError: if input data cannot be initialized as an operator.

        Additional Information
        ----------------------
        If the input or output dimensions are None, they will be
        automatically determined from the input data. If the input data is
        a Numpy array of shape (2**N, 2**N) qubit systems will be used. If
        the input operator is not an N-qubit operator, it will assign a
        single subsystem with dimension specifed by the shape of the input.
        """
        if hasattr(data, 'to_operator'):
            data = data.to_operator()
            mat = data.data
            if input_dims is None:
                input_dims = data.input_dims()
            if output_dims is None:
                output_dims = data.output_dims()
        elif isinstance(data, Pauli):
            # TODO:
            # This is a place holder until Pauli is modified to inherit
            # from BaseOperator and have its own `to_operator` method
            mat = data.to_matrix()
            if input_dims is None:
                input_dims = len(data) * [2]
            if output_dims is None:
                output_dims = len(data) * [2]
        elif isinstance(data, (list, np.ndarray)):
            # We initialize directly from operator matrix
            mat = np.array(data, dtype=complex)
        else:
            raise QiskitError("Invalid input data format for Operator")
        # Determine input and output dimensions
        dout, din = mat.shape
        output_dims = self._automatic_dims(output_dims, dout)
        input_dims = self._automatic_dims(input_dims, din)
        super().__init__('Operator', mat, input_dims, output_dims)

    def is_unitary(self):
        """Return True if operator is a unitary matrix."""
        return is_unitary_matrix(self._data, rtol=self._rtol, atol=self._atol)

    def to_operator(self):
        """Convert operator to matrix operator class"""
        return self

    def conjugate(self):
        """Return the conjugate of the operator."""
        return Operator(
            np.conj(self.data), self.input_dims(), self.output_dims())

    def transpose(self):
        """Return the transpose of the operator."""
        return Operator(
            np.transpose(self.data), self.input_dims(), self.output_dims())

    def compose(self, other, qubits=None, front=False):
        """Return the composition channel self∘other.

        Args:
            other (Operator): an operator object.
            qubits (list): a list of subsystem positions to compose other on.
            front (bool): If False compose in standard order other(self(input))
                          otherwise compose in reverse order self(other(input))
                          [default: False]

        Returns:
            Operator: The composed operator.

        Raises:
            QiskitError: if other cannot be converted to an Operator or has
            incompatible dimensions.
        """
        # Convert to Operator
        if not isinstance(other, Operator):
            other = Operator(other)
        # Check dimensions are compatible
        if front and self.input_dims(qubits=qubits) != other.output_dims():
            raise QiskitError(
                'output_dims of other must match subsystem input_dims')
        if not front and self.output_dims(qubits=qubits) != other.input_dims():
            raise QiskitError(
                'input_dims of other must match subsystem output_dims')
        # Full composition of operators
        if qubits is None:
            if front:
                # Composition A(B(input))
                input_dims = other.input_dims()
                output_dims = self.output_dims()
                data = np.dot(self._data, other.data)
            else:
                # Composition B(A(input))
                input_dims = self.input_dims()
                output_dims = other.output_dims()
                data = np.dot(other.data, self._data)
            return Operator(data, input_dims, output_dims)
        # Compose with other on subsystem
        return self._compose_subsystem(other, qubits, front)

    def power(self, n):
        """Return the matrix power of the operator.

        Args:
            n (int): the power to raise the matrix to.

        Returns:
            BaseOperator: the n-times composed operator.

        Raises:
            QiskitError: if the input and output dimensions of the operator
            are not equal, or the power is not a positive integer.
        """
        if not isinstance(n, int):
            raise QiskitError("Can only take integer powers of Operator.")
        if self.input_dims() != self.output_dims():
            raise QiskitError("Can only power with input_dims = output_dims.")
        # Override base class power so we can implement more efficiently
        # using Numpy.matrix_power
        return Operator(
            np.linalg.matrix_power(self.data, n), self.input_dims(),
            self.output_dims())

    def tensor(self, other):
        """Return the tensor product operator self ⊗ other.

        Args:
            other (Operator): a operator subclass object.

        Returns:
            Operator: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other cannot be converted to an operator.
        """
        return self._tensor_product(other, reverse=False)

    def expand(self, other):
        """Return the tensor product operator other ⊗ self.

        Args:
            other (Operator): an operator object.

        Returns:
            Operator: the tensor product operator other ⊗ self.

        Raises:
            QiskitError: if other cannot be converted to an operator.
        """
        return self._tensor_product(other, reverse=True)

    def add(self, other):
        """Return the operator self + other.

        Args:
            other (Operator): an operator object.

        Returns:
            Operator: the operator self + other.

        Raises:
            QiskitError: if other is not an operator, or has incompatible
            dimensions.
        """
        if not isinstance(other, Operator):
            other = Operator(other)
        if self.dim != other.dim:
            raise QiskitError("other operator has different dimensions.")
        return Operator(self.data + other.data, self.input_dims(),
                        self.output_dims())

    def subtract(self, other):
        """Return the operator self - other.

        Args:
            other (Operator): an operator object.

        Returns:
            Operator: the operator self - other.

        Raises:
            QiskitError: if other is not an operator, or has incompatible
            dimensions.
        """
        if not isinstance(other, Operator):
            other = Operator(other)
        if self.dim != other.dim:
            raise QiskitError("other operator has different dimensions.")
        return Operator(self.data - other.data, self.input_dims(),
                        self.output_dims())

    def multiply(self, other):
        """Return the operator self + other.

        Args:
            other (complex): a complex number.

        Returns:
            Operator: the operator other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        return Operator(other * self.data, self.input_dims(),
                        self.output_dims())

    @property
    def _shape(self):
        """Return the tensor shape of the matrix operator"""
        return tuple(reversed(self.output_dims())) + tuple(
            reversed(self.input_dims()))

    def _evolve(self, state, qubits=None):
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
        state = self._format_state(state)
        if qubits is None:
            if state.shape[0] != self._input_dim:
                raise QiskitError(
                    "Operator input dimension is not equal to state dimension."
                )
            if state.ndim == 1:
                # Return evolved statevector
                return np.dot(self.data, state)
            # Return evolved density matrix
            return np.dot(
                np.dot(self.data, state), np.transpose(np.conj(self.data)))
        # Subsystem evolution
        return self._evolve_subsystem(state, qubits)

    def _tensor_product(self, other, reverse=False):
        """Return the tensor product operator.

        Args:
            other (Operator): another operator.
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False
        Returns:
            Operator: the tensor product operator.

        Raises:
            QiskitError: if other cannot be converted into an Operator.
        """
        # Convert other to Operator
        if not isinstance(other, Operator):
            other = Operator(other)
        if reverse:
            input_dims = self.input_dims() + other.input_dims()
            output_dims = self.output_dims() + other.output_dims()
            data = np.kron(other._data, self._data)
        else:
            input_dims = other.input_dims() + self.input_dims()
            output_dims = other.output_dims() + self.output_dims()
            data = np.kron(self._data, other._data)
        return Operator(data, input_dims, output_dims)

    def _compose_subsystem(self, other, qubits, front=False):
        """Return the composition channel."""
        # Compute tensor contraction indices from qubits
        input_dims = list(self.input_dims())
        output_dims = list(self.output_dims())
        if front:
            num_indices = len(self.input_dims())
            shift = len(self.output_dims())
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
        indices = [num_indices - 1 - qubit for qubit in qubits]
        final_shape = [np.product(output_dims), np.product(input_dims)]
        data = np.reshape(
            self._einsum_matmul(tensor, mat, indices, shift, right_mul),
            final_shape)
        return Operator(data, input_dims, output_dims)

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
        if self.input_dims() != len(qubits) * (2,):
            raise QiskitError(
                "Operator input dimensions are not compatible with state subsystem dimensions."
            )
        if state.ndim == 1:
            # Return evolved statevector
            tensor = np.reshape(state, state_dims)
            indices = [len(state_dims) - 1 - qubit for qubit in qubits]
            tensor = self._einsum_matmul(tensor, mat, indices)
            return np.reshape(tensor, state_size)
        # Return evolved density matrix
        tensor = np.reshape(state, 2 * state_dims)
        indices = [len(state_dims) - 1 - qubit for qubit in qubits]
        right_shift = len(state_dims)
        # Left multiply by operator
        tensor = self._einsum_matmul(tensor, mat, indices)
        # Right multiply by adjoint operator
        # We implement the transpose by doing left multiplication instead of right
        # in the _einsum_matmul function
        tensor = self._einsum_matmul(
            tensor, np.conj(mat), indices, shift=right_shift)
        return np.reshape(tensor, [state_size, state_size])

    def _format_state(self, state):
        """Format input state so it is statevector or density matrix"""
        state = np.array(state)
        shape = state.shape
        ndim = state.ndim
        if ndim > 2:
            raise QiskitError('Input state is not a vector or matrix.')
        # Flatten column-vector to vector
        if ndim == 2:
            if shape[1] != 1 and shape[1] != shape[0]:
                raise QiskitError('Input state is not a vector or matrix.')
            if shape[1] == 1:
                # flatten colum-vector to vector
                state = np.reshape(state, shape[0])
        return state
