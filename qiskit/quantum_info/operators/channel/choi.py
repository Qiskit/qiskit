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

# pylint: disable=unpacking-non-sequence

"""
Choi-matrix representation of a Quantum Channel.
"""

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.transformations import _to_choi
from qiskit.quantum_info.operators.channel.transformations import _bipartite_tensor


class Choi(QuantumChannel):
    r"""Choi-matrix representation of a Quantum Channel.

    The Choi-matrix representation of a quantum channel :math:`\mathcal{E}`
    is a matrix

    .. math::

        \Lambda = \sum_{i,j} |i\rangle\!\langle j|\otimes
                    \mathcal{E}\left(|i\rangle\!\langle j|\right)

    Evolution of a :class:`~qiskit.quantum_info.DensityMatrix`
    :math:`\rho` with respect to the Choi-matrix is given by

    .. math::

        \mathcal{E}(\rho) = \mbox{Tr}_{1}\left[\Lambda
                            (\rho^T \otimes \mathbb{I})\right]

    where :math:`\mbox{Tr}_1` is the :func:`partial_trace` over subsystem 1.

    See reference [1] for further details.

    References:
        1. C.J. Wood, J.D. Biamonte, D.G. Cory, *Tensor networks and graphical calculus
           for open quantum systems*, Quant. Inf. Comp. 15, 0579-0811 (2015).
           `arXiv:1111.6950 [quant-ph] <https://arxiv.org/abs/1111.6950>`_
    """

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

        Additional Information:
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
            choi_mat = np.asarray(data, dtype=complex)
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
                data = SuperOp._init_instruction(data)
            else:
                # We use the QuantumChannel init transform to initialize
                # other objects into a QuantumChannel or Operator object.
                data = self._init_transformer(data)
            input_dim, output_dim = data.dim
            # Now that the input is an operator we convert it to a Choi object
            rep = getattr(data, '_channel_rep', 'Operator')
            choi_mat = _to_choi(rep, data._data, input_dim, output_dim)
            if input_dims is None:
                input_dims = data.input_dims()
            if output_dims is None:
                output_dims = data.output_dims()
        # Check and format input and output dimensions
        input_dims = self._automatic_dims(input_dims, input_dim)
        output_dims = self._automatic_dims(output_dims, output_dim)
        super().__init__(choi_mat, input_dims, output_dims, 'Choi')

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
        return Choi(data,
                    input_dims=self.output_dims(),
                    output_dims=self.input_dims())

    def compose(self, other, qargs=None, front=False):
        """Return the composed quantum channel self @ other.

        Args:
            other (QuantumChannel): a quantum channel.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].
            front (bool): If True compose using right operator multiplication,
                          instead of left multiplication [default: False].

        Returns:
            Choi: The quantum channel self @ other.

        Raises:
            QiskitError: if other has incompatible dimensions.

        Additional Information:
            Composition (``@``) is defined as `left` matrix multiplication for
            :class:`SuperOp` matrices. That is that ``A @ B`` is equal to ``B * A``.
            Setting ``front=True`` returns `right` matrix multiplication
            ``A * B`` and is equivalent to the :meth:`dot` method.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if qargs is not None:
            return Choi(
                SuperOp(self).compose(other, qargs=qargs, front=front))

        if not isinstance(other, Choi):
            other = Choi(other)
        input_dims, output_dims = self._get_compose_dims(other, qargs, front)
        input_dim = np.product(input_dims)
        output_dim = np.product(output_dims)

        if front:
            first = np.reshape(other._data, other._bipartite_shape)
            second = np.reshape(self._data, self._bipartite_shape)
        else:
            first = np.reshape(self._data, self._bipartite_shape)
            second = np.reshape(other._data, other._bipartite_shape)

        # Contract Choi matrices for composition
        data = np.reshape(np.einsum('iAjB,AkBl->ikjl', first, second),
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
        # Convert other to Choi
        if not isinstance(other, Choi):
            other = Choi(other)

        input_dims = other.input_dims() + self.input_dims()
        output_dims = other.output_dims() + self.output_dims()
        data = _bipartite_tensor(self._data,
                                 other.data,
                                 shape1=self._bipartite_shape,
                                 shape2=other._bipartite_shape)
        return Choi(data, input_dims, output_dims)

    def expand(self, other):
        """Return the tensor product channel other ⊗ self.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            Choi: the tensor product channel other ⊗ self as a Choi object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        # Convert other to Choi
        if not isinstance(other, Choi):
            other = Choi(other)

        input_dims = self.input_dims() + other.input_dims()
        output_dims = self.output_dims() + other.output_dims()
        data = _bipartite_tensor(other.data,
                                 self._data,
                                 shape1=other._bipartite_shape,
                                 shape2=self._bipartite_shape)
        return Choi(data, input_dims, output_dims)

    def _evolve(self, state, qargs=None):
        """Evolve a quantum state by the quantum channel.

        Args:
            state (DensityMatrix or Statevector): The input state.
            qargs (list): a list of quantum state subsystem positions to apply
                           the quantum channel on.

        Returns:
            DensityMatrix: the output quantum state as a density matrix.

        Raises:
            QiskitError: if the quantum channel dimension does not match the
                         specified quantum state subsystem dimensions.
        """
        return SuperOp(self)._evolve(state, qargs)
