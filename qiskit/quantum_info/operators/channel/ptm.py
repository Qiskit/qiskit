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
Pauli Transfer Matrix (PTM) representation of a Quantum Channel.
"""

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.transformations import _to_ptm


class PTM(QuantumChannel):
    r"""Pauli Transfer Matrix (PTM) representation of a Quantum Channel.

    The PTM representation of an :math:`n`-qubit quantum channel
    :math:`\mathcal{E}` is an :math:`n`-qubit :class:`SuperOp` :math:`R`
    defined with respect to vectorization in the Pauli basis instead of
    column-vectorization. The elements of the PTM :math:`R` are
    given by

    .. math::

        R_{i,j} = \mbox{Tr}\left[P_i \mathcal{E}(P_j) \right]

    where :math:`[P_0, P_1, ..., P_{4^{n}-1}]` is the :math:`n`-qubit Pauli basis in
    lexicographic order.

    Evolution of a :class:`~qiskit.quantum_info.DensityMatrix`
    :math:`\rho` with respect to the PTM is given by

    .. math::

        |\mathcal{E}(\rho)\rangle\!\rangle_P = S_P |\rho\rangle\!\rangle_P

    where :math:`|A\rangle\!\rangle_P` denotes vectorization in the Pauli basis
    :math:`\langle i | A\rangle\!\rangle_P = \mbox{Tr}[P_i A]`.

    See reference [1] for further details.

    References:
        1. C.J. Wood, J.D. Biamonte, D.G. Cory, *Tensor networks and graphical calculus
           for open quantum systems*, Quant. Inf. Comp. 15, 0579-0811 (2015).
           `arXiv:1111.6950 [quant-ph] <https://arxiv.org/abs/1111.6950>`_
    """

    def __init__(self, data, input_dims=None, output_dims=None):
        """Initialize a PTM quantum channel operator.

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
                         cannot be initialized as a PTM.

        Additional Information:
            If the input or output dimensions are None, they will be
            automatically determined from the input data. The PTM
            representation is only valid for N-qubit channels.
        """
        # If the input is a raw list or matrix we assume that it is
        # already a Chi matrix.
        if isinstance(data, (list, np.ndarray)):
            # Should we force this to be real?
            ptm = np.asarray(data, dtype=complex)
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
            # Now that the input is an operator we convert it to a PTM object
            rep = getattr(data, '_channel_rep', 'Operator')
            ptm = _to_ptm(rep, data._data, input_dim, output_dim)
            if input_dims is None:
                input_dims = data.input_dims()
            if output_dims is None:
                output_dims = data.output_dims()
        # Check input is N-qubit channel
        num_qubits = int(np.log2(input_dim))
        if 2**num_qubits != input_dim:
            raise QiskitError("Input is not an n-qubit Pauli transfer matrix.")
        # Check and format input and output dimensions
        input_dims = self._automatic_dims(input_dims, input_dim)
        output_dims = self._automatic_dims(output_dims, output_dim)
        super().__init__(ptm, input_dims, output_dims, 'PTM')

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
            PTM: The quantum channel self @ other.

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
            return PTM(
                SuperOp(self).compose(other, qargs=qargs, front=front))

        # Convert other to PTM
        if not isinstance(other, PTM):
            other = PTM(other)
        input_dims, output_dims = self._get_compose_dims(other, qargs, front)
        if front:
            data = np.dot(self._data, other.data)
        else:
            data = np.dot(other.data, self._data)
        return PTM(data, input_dims, output_dims)

    def power(self, n):
        """The matrix power of the channel.

        Args:
            n (int): compute the matrix power of the superoperator matrix.

        Returns:
            PTM: the matrix power of the SuperOp converted to a PTM channel.

        Raises:
            QiskitError: if the input and output dimensions of the
                         QuantumChannel are not equal, or the power is not
                         an integer.
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
        if not isinstance(other, PTM):
            other = PTM(other)
        input_dims = other.input_dims() + self.input_dims()
        output_dims = other.output_dims() + self.output_dims()
        data = np.kron(self._data, other.data)
        return PTM(data, input_dims, output_dims)

    def expand(self, other):
        """Return the tensor product channel other ⊗ self.

        Args:
            other (QuantumChannel): a quantum channel.

        Returns:
            PTM: the tensor product channel other ⊗ self as a PTM object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        if not isinstance(other, PTM):
            other = PTM(other)
        input_dims = self.input_dims() + other.input_dims()
        output_dims = self.output_dims() + other.output_dims()
        data = np.kron(other.data, self._data)
        return PTM(data, input_dims, output_dims)

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
