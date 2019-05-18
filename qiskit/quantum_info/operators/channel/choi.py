# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

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

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.transformations import _to_choi
from qiskit.quantum_info.operators.channel.transformations import _bipartite_tensor


class Choi(QuantumChannel):
    """Choi-matrix representation of a quantum channel"""

    def __init__(self, data, input_dims=None, output_dims=None):
        """Initialize a quantum channel Choi matrix operator.

        Args:
            data (QuantumCircuit or
                  Instruction or
                  BaseOperator or
                  matrix): data to initialize superoperator.
            input_dims (tuple): the input subsystem dimensions.
                                [Default: None]
            output_dims (tuple): the output subsystem dimensions.
                                 [Default: None]

        Raises:
            QiskitError: if input data cannot be initialized as a
            Choi matrix.

        Additional Information
        ----------------------
        If the input or output dimensions are None, they will be
        automatically determined from the input data. If the input data is
        a Numpy array of shape (4**N, 4**N) qubit systems will be used. If
        the input operator is not an N-qubit operator, it will assign a
        single subsystem with dimension specified by the shape of the input.
        """
        # If the input is a raw list or matrix we assume that it is
        # already a Choi matrix.
        if isinstance(data, (list, np.ndarray)):
            # Initialize from raw numpy or list matrix.
            choi_mat = np.array(data, dtype=complex)
            # Determine input and output dimensions
            dim_l, dim_r = choi_mat.shape
            if dim_l != dim_r:
                raise QiskitError('Invalid Choi-matrix input.')
            if input_dims:
                input_dim = np.product(input_dims)
            if output_dims:
                output_dim = np.product(output_dims)
            if output_dims is None and input_dims is None:
                output_dim = int(np.sqrt(dim_l))
                input_dim = dim_l // output_dim
            elif input_dims is None:
                input_dim = dim_l // output_dim
            elif output_dims is None:
                output_dim = dim_l // input_dim
            # Check dimensions
            if input_dim * output_dim != dim_l:
                raise QiskitError("Invalid shape for input Choi-matrix.")
        else:
            # Otherwise we initialize by conversion from another Qiskit
            # object into the QuantumChannel.
            if isinstance(data, (QuantumCircuit, Instruction)):
                # If the input is a Terra QuantumCircuit or Instruction we
                # convert it to a SuperOp
                data = SuperOp._instruction_to_superop(data)
            else:
                # We use the QuantumChannel init transform to initialize
                # other objects into a QuantumChannel or Operator object.
                data = self._init_transformer(data)
            input_dim, output_dim = data.dim
            # Now that the input is an operator we convert it to a Choi object
            choi_mat = _to_choi(data.rep, data._data, input_dim, output_dim)
            if input_dims is None:
                input_dims = data.input_dims()
            if output_dims is None:
                output_dims = data.output_dims()
        # Check and format input and output dimensions
        input_dims = self._automatic_dims(input_dims, input_dim)
        output_dims = self._automatic_dims(output_dims, output_dim)
        super().__init__('Choi', choi_mat, input_dims, output_dims)

    @property
    def _bipartite_shape(self):
        """Return the shape for bipartite matrix"""
        return (self._input_dim, self._output_dim, self._input_dim,
                self._output_dim)

    def conjugate(self):
        """Return the conjugate of the QuantumChannel."""
        return Choi(np.conj(self._data), self.input_dims(), self.output_dims())

    def transpose(self):
        """Return the transpose of the QuantumChannel."""
        # Make bipartite matrix
        d_in, d_out = self.dim
        data = np.reshape(self._data, (d_in, d_out, d_in, d_out))
        # Swap input and output indices on bipartite matrix
        data = np.transpose(data, (1, 0, 3, 2))
        # Transpose channel has input and output dimensions swapped
        data = np.reshape(data, (d_in * d_out, d_in * d_out))
        return Choi(
            data, input_dims=self.output_dims(), output_dims=self.input_dims())

    def compose(self, other, qargs=None, front=False):
        """Return the composition channel self∘other.

        Args:
            other (QuantumChannel): a quantum channel.
            qargs (list): a list of subsystem positions to compose other on.
            front (bool): If False compose in standard order other(self(input))
                          otherwise compose in reverse order self(other(input))
                          [default: False]

        Returns:
            Choi: The composition channel as a Choi object.

        Raises:
            QiskitError: if other cannot be converted to a channel or
            has incompatible dimensions.
        """
        if qargs is not None:
            return Choi(
                SuperOp(self).compose(other, qargs=qargs, front=front))

        # Convert to Choi matrix
        if not isinstance(other, Choi):
            other = Choi(other)
        # Check dimensions match up
        if front and self._input_dim != other._output_dim:
            raise QiskitError(
                'input_dim of self must match output_dim of other')
        if not front and self._output_dim != other._input_dim:
            raise QiskitError(
                'input_dim of other must match output_dim of self')

        if front:
            first = np.reshape(other._data, other._bipartite_shape)
            second = np.reshape(self._data, self._bipartite_shape)
            input_dim = other._input_dim
            input_dims = other.input_dims()
            output_dim = self._output_dim
            output_dims = self.output_dims()
        else:
            first = np.reshape(self._data, self._bipartite_shape)
            second = np.reshape(other._data, other._bipartite_shape)
            input_dim = self._input_dim
            input_dims = self.input_dims()
            output_dim = other._output_dim
            output_dims = other.output_dims()

        # Contract Choi matrices for composition
        data = np.reshape(
            np.einsum('iAjB,AkBl->ikjl', first, second),
            (input_dim * output_dim, input_dim * output_dim))
        return Choi(data, input_dims, output_dims)

    def power(self, n):
        """The matrix power of the channel.

        Args:
            n (int): compute the matrix power of the superoperator matrix.

        Returns:
            Choi: the matrix power of the SuperOp converted to a Choi channel.

        Raises:
            QiskitError: if the input and output dimensions of the
            QuantumChannel are not equal, or the power is not an integer.
        """
        if n > 0:
            return super().power(n)
        return Choi(SuperOp(self).power(n))

    def tensor(self, other):
        """Return the tensor product channel self ⊗ other.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            Choi: the tensor product channel self ⊗ other as a Choi object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        return self._tensor_product(other, reverse=False)

    def expand(self, other):
        """Return the tensor product channel other ⊗ self.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            Choi: the tensor product channel other ⊗ self as a Choi object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        return self._tensor_product(other, reverse=True)

    def add(self, other):
        """Return the QuantumChannel self + other.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            Choi: the linear addition self + other as a Choi object.

        Raises:
            QiskitError: if other cannot be converted to a channel or
            has incompatible dimensions.
        """
        if not isinstance(other, Choi):
            other = Choi(other)
        if self.dim != other.dim:
            raise QiskitError("other QuantumChannel dimensions are not equal")
        return Choi(self._data + other.data, self._input_dims,
                    self._output_dims)

    def subtract(self, other):
        """Return the QuantumChannel self - other.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            Choi: the linear subtraction self - other as Choi object.

        Raises:
            QiskitError: if other cannot be converted to a channel or
            has incompatible dimensions.
        """
        if not isinstance(other, Choi):
            other = Choi(other)
        if self.dim != other.dim:
            raise QiskitError("other QuantumChannel dimensions are not equal")
        return Choi(self._data - other.data, self._input_dims,
                    self._output_dims)

    def multiply(self, other):
        """Return the QuantumChannel self + other.

        Args:
            other (complex): a complex number.

        Returns:
            Choi: the scalar multiplication other * self as a Choi object.

        Raises:
            QiskitError: if other is not a valid scalar.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        return Choi(other * self._data, self._input_dims, self._output_dims)

    def _evolve(self, state, qargs=None):
        """Evolve a quantum state by the QuantumChannel.

        Args:
            state (QuantumState): The input statevector or density matrix.
            qargs (list): a list of QuantumState subsystem positions to apply
                           the operator on.

        Returns:
            DensityMatrix: the output quantum state as a density matrix.

        Raises:
            QiskitError: if the operator dimension does not match the
            specified QuantumState subsystem dimensions.
        """
        # If subsystem evolution we use the SuperOp representation
        if qargs is not None:
            return SuperOp(self)._evolve(state, qargs)
        # Otherwise we compute full evolution directly
        state = self._format_state(state, density_matrix=True)
        if state.shape[0] != self._input_dim:
            raise QiskitError(
                "QuantumChannel input dimension is not equal to state dimension."
            )
        return np.einsum('AB,AiBj->ij', state,
                         np.reshape(self._data, self._bipartite_shape))

    def _tensor_product(self, other, reverse=False):
        """Return the tensor product channel.

        Args:
            other (QuantumChannel): a quantum channel.
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False
        Returns:
            Choi: the tensor product channel as a Choi object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass.
        """
        # Convert other to Choi
        if not isinstance(other, Choi):
            other = Choi(other)

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
        return Choi(data, input_dims, output_dims)
