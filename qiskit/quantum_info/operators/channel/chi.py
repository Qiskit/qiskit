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
Chi-matrix representation of a Quantum Channel.
"""

from numbers import Number
import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.choi import Choi
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.transformations import _to_chi


class Chi(QuantumChannel):
    r"""Pauli basis Chi-matrix representation of a quantum channel.

    The Chi-matrix representation of an :math:`n`-qubit quantum channel
    :math:`\mathcal{E}` is a matrix :math:`\chi` such that the evolution of a
    :class:`~qiskit.quantum_info.DensityMatrix` :math:`\rho` is given by

    .. math::

        \mathcal{E}(ρ) = \sum_{i, j} \chi_{i,j} P_i ρ P_j

    where :math:`[P_0, P_1, ..., P_{4^{n}-1}]` is the :math:`n`-qubit Pauli basis in
    lexicographic order. It is related to the :class:`Choi` representation by a change
    of basis of the Choi-matrix into the Pauli basis.

    See reference [1] for further details.

    References:
        1. C.J. Wood, J.D. Biamonte, D.G. Cory, *Tensor networks and graphical calculus
           for open quantum systems*, Quant. Inf. Comp. 15, 0579-0811 (2015).
           `arXiv:1111.6950 [quant-ph] <https://arxiv.org/abs/1111.6950>`_
    """

    def __init__(self, data, input_dims=None, output_dims=None):
        """Initialize a quantum channel Chi-matrix operator.

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
            QiskitError: if input data is not an N-qubit channel or
            cannot be initialized as a Chi-matrix.

        Additional Information
        ----------------------
        If the input or output dimensions are None, they will be
        automatically determined from the input data. The Chi matrix
        representation is only valid for N-qubit channels.
        """
        # If the input is a raw list or matrix we assume that it is
        # already a Chi matrix.
        if isinstance(data, (list, np.ndarray)):
            # Initialize from raw numpy or list matrix.
            chi_mat = np.array(data, dtype=complex)
            # Determine input and output dimensions
            dim_l, dim_r = chi_mat.shape
            if dim_l != dim_r:
                raise QiskitError('Invalid Chi-matrix input.')
            if input_dims:
                input_dim = np.product(input_dims)
            if output_dims:
                output_dim = np.product(input_dims)
            if output_dims is None and input_dims is None:
                output_dim = int(np.sqrt(dim_l))
                input_dim = dim_l // output_dim
            elif input_dims is None:
                input_dim = dim_l // output_dim
            elif output_dims is None:
                output_dim = dim_l // input_dim
            # Check dimensions
            if input_dim * output_dim != dim_l:
                raise QiskitError("Invalid shape for Chi-matrix input.")
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
            # Now that the input is an operator we convert it to a Chi object
            chi_mat = _to_chi(data.rep, data._data, input_dim, output_dim)
            if input_dims is None:
                input_dims = data.input_dims()
            if output_dims is None:
                output_dims = data.output_dims()
        # Check input is N-qubit channel
        n_qubits = int(np.log2(input_dim))
        if 2**n_qubits != input_dim:
            raise QiskitError("Input is not an n-qubit Chi matrix.")
        # Check and format input and output dimensions
        input_dims = self._automatic_dims(input_dims, input_dim)
        output_dims = self._automatic_dims(output_dims, output_dim)
        super().__init__('Chi', chi_mat, input_dims, output_dims)

    @property
    def _bipartite_shape(self):
        """Return the shape for bipartite matrix"""
        return (self._input_dim, self._output_dim, self._input_dim,
                self._output_dim)

    def conjugate(self):
        """Return the conjugate of the QuantumChannel."""
        # Since conjugation is basis dependent we transform
        # to the Choi representation to compute the
        # conjugate channel
        return Chi(Choi(self).conjugate())

    def transpose(self):
        """Return the transpose of the QuantumChannel."""
        # Since conjugation is basis dependent we transform
        # to the Choi representation to compute the
        # conjugate channel
        return Chi(Choi(self).transpose())

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
            Chi: The quantum channel self @ other.

        Raises:
            QiskitError: if other cannot be converted to a Chi or has
            incompatible dimensions.

        Additional Information:
            Composition (``@``) is defined as `left` matrix multiplication for
            :class:`SuperOp` matrices. That is that ``A @ B`` is equal to ``B * A``.
            Setting ``front=True`` returns `right` matrix multiplication
            ``A * B`` and is equivalent to the :meth:`dot` method.
        """
        return super().compose(other, qargs=qargs, front=front)

    def dot(self, other, qargs=None):
        """Return the right multiplied quantum channel self * other.

        Args:
            other (QuantumChannel): a quantum channel.
            qargs (list): a list of subsystem positions to compose other on.

        Returns:
            Chi: The quantum channel self * other.

        Raises:
            QiskitError: if other cannot be converted to a Chi or has
            incompatible dimensions.
        """
        return super().dot(other, qargs=qargs)

    def power(self, n):
        """The matrix power of the channel.

        Args:
            n (int): compute the matrix power of the superoperator matrix.

        Returns:
            Chi: the matrix power of the SuperOp converted to a Chi channel.

        Raises:
            QiskitError: if the input and output dimensions of the
            QuantumChannel are not equal, or the power is not an integer.
        """
        if n > 0:
            return super().power(n)
        return Chi(SuperOp(self).power(n))

    def tensor(self, other):
        """Return the tensor product channel self ⊗ other.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            Chi: the tensor product channel self ⊗ other as a Chi object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass.
        """
        if not isinstance(other, Chi):
            other = Chi(other)
        input_dims = other.input_dims() + self.input_dims()
        output_dims = other.output_dims() + self.output_dims()
        data = np.kron(self._data, other.data)
        return Chi(data, input_dims, output_dims)

    def expand(self, other):
        """Return the tensor product channel other ⊗ self.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            Chi: the tensor product channel other ⊗ self as a Chi object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass.
        """
        if not isinstance(other, Chi):
            other = Chi(other)
        input_dims = self.input_dims() + other.input_dims()
        output_dims = self.output_dims() + other.output_dims()
        data = np.kron(other.data, self._data)
        return Chi(data, input_dims, output_dims)

    def add(self, other):
        """Return the QuantumChannel self + other.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            Chi: the linear addition self + other as a Chi object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        if not isinstance(other, Chi):
            other = Chi(other)
        if self.dim != other.dim:
            raise QiskitError("other QuantumChannel dimensions are not equal")
        return Chi(self._data + other.data, self._input_dims,
                   self._output_dims)

    def subtract(self, other):
        """Return the QuantumChannel self - other.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            Chi: the linear subtraction self - other as Chi object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        if not isinstance(other, Chi):
            other = Chi(other)
        if self.dim != other.dim:
            raise QiskitError("other QuantumChannel dimensions are not equal")
        return Chi(self._data - other.data, self._input_dims,
                   self._output_dims)

    def multiply(self, other):
        """Return the QuantumChannel self + other.

        Args:
            other (complex): a complex number.

        Returns:
            Chi: the scalar multiplication other * self as a Chi object.

        Raises:
            QiskitError: if other is not a valid scalar.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        return Chi(other * self._data, self._input_dims, self._output_dims)

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

    def _chanmul(self, other, qargs=None, left_multiply=False):
        """Multiply two quantum channels.

        Args:
            other (QuantumChannel): a quantum channel.
            qargs (list): a list of subsystem positions to compose other on.
            left_multiply (bool): If True return other * self
                                  If False return self * other [Default:False]

        Returns:
            Choi: The composition channel as a Chi object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        if qargs is not None:
            return Chi(
                SuperOp(self)._chanmul(other,
                                       qargs=qargs,
                                       left_multiply=left_multiply))

        # Convert other to Choi since we convert via Choi
        if not isinstance(other, Choi):
            other = Choi(other)
        # Check dimensions match up
        if not left_multiply and self._input_dim != other._output_dim:
            raise QiskitError(
                'input_dim of self must match output_dim of other')
        if left_multiply and self._output_dim != other._input_dim:
            raise QiskitError(
                'input_dim of other must match output_dim of self')
        # Since we cannot directly multiply two channels in the Chi
        # representation we convert to the Choi representation
        return Chi(Choi(self)._chanmul(other, left_multiply=left_multiply))
