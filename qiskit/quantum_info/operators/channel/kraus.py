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

# pylint: disable=len-as-condition
"""
Kraus representation of a Quantum Channel.
"""

from numbers import Number
import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.choi import Choi
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.transformations import _to_kraus


class Kraus(QuantumChannel):
    r"""Kraus representation of a quantum channel.

    The Kraus representation for a quantum channel :math:`\mathcal{E}` is a
    set of matrices :math:`[A_0,...,A_{K-1}]` such that

    For a quantum channel :math:`\mathcal{E}`, the Kraus representation is
    given by a set of matrices :math:`[A_0,...,A_{K-1}]` such that the
    evolution of a :class:`~qiskit.quantum_info.DensityMatrix`
    :math:`\rho` is given by

    .. math::

        \mathcal{E}(\rho) = \sum_{i=0}^{K-1} A_i \rho A_i^\dagger

    A general operator map :math:`\mathcal{G}` can also be written using the
    generalized Kraus representation which is given by two sets of matrices
    :math:`[A_0,...,A_{K-1}]`, :math:`[B_0,...,A_{B-1}]` such that

    .. math::

        \mathcal{G}(\rho) = \sum_{i=0}^{K-1} A_i \rho B_i^\dagger

    See reference [1] for further details.

    References:
        1. C.J. Wood, J.D. Biamonte, D.G. Cory, *Tensor networks and graphical calculus
           for open quantum systems*, Quant. Inf. Comp. 15, 0579-0811 (2015).
           `arXiv:1111.6950 [quant-ph] <https://arxiv.org/abs/1111.6950>`_
    """

    def __init__(self, data, input_dims=None, output_dims=None):
        """Initialize a quantum channel Kraus operator.

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
            a list of Kraus matrices.

        Additional Information
        ----------------------
        If the input or output dimensions are None, they will be
        automatically determined from the input data. If the input data is
        a list of Numpy arrays of shape (2**N, 2**N) qubit systems will be used. If
        the input does not correspond to an N-qubit channel, it will assign a
        single subsystem with dimension specified by the shape of the input.
        """
        # If the input is a list or tuple we assume it is a list of Kraus
        # matrices, if it is a numpy array we assume that it is a single Kraus
        # operator
        if isinstance(data, (list, tuple, np.ndarray)):
            # Check if it is a single unitary matrix A for channel:
            # E(rho) = A * rho * A^\dagger
            if isinstance(data, np.ndarray) or np.array(data).ndim == 2:
                # Convert single Kraus op to general Kraus pair
                kraus = ([np.array(data, dtype=complex)], None)
                shape = kraus[0][0].shape

            # Check if single Kraus set [A_i] for channel:
            # E(rho) = sum_i A_i * rho * A_i^dagger
            elif isinstance(data, list) and len(data) > 0:
                # Get dimensions from first Kraus op
                kraus = [np.array(data[0], dtype=complex)]
                shape = kraus[0].shape
                # Iterate over remaining ops and check they are same shape
                for i in data[1:]:
                    op = np.array(i, dtype=complex)
                    if op.shape != shape:
                        raise QiskitError(
                            "Kraus operators are different dimensions.")
                    kraus.append(op)
                # Convert single Kraus set to general Kraus pair
                kraus = (kraus, None)

            # Check if generalized Kraus set ([A_i], [B_i]) for channel:
            # E(rho) = sum_i A_i * rho * B_i^dagger
            elif isinstance(data,
                            tuple) and len(data) == 2 and len(data[0]) > 0:
                kraus_left = [np.array(data[0][0], dtype=complex)]
                shape = kraus_left[0].shape
                for i in data[0][1:]:
                    op = np.array(i, dtype=complex)
                    if op.shape != shape:
                        raise QiskitError(
                            "Kraus operators are different dimensions.")
                    kraus_left.append(op)
                if data[1] is None:
                    kraus = (kraus_left, None)
                else:
                    kraus_right = []
                    for i in data[1]:
                        op = np.array(i, dtype=complex)
                        if op.shape != shape:
                            raise QiskitError(
                                "Kraus operators are different dimensions.")
                        kraus_right.append(op)
                    kraus = (kraus_left, kraus_right)
            else:
                raise QiskitError("Invalid input for Kraus channel.")
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
            # Now that the input is an operator we convert it to a Kraus
            kraus = _to_kraus(data.rep, data._data, input_dim, output_dim)
            if input_dims is None:
                input_dims = data.input_dims()
            if output_dims is None:
                output_dims = data.output_dims()

        output_dim, input_dim = kraus[0][0].shape
        # Check and format input and output dimensions
        input_dims = self._automatic_dims(input_dims, input_dim)
        output_dims = self._automatic_dims(output_dims, output_dim)
        # Initialize either single or general Kraus
        if kraus[1] is None or np.allclose(kraus[0], kraus[1]):
            # Standard Kraus map
            super().__init__('Kraus', (kraus[0], None), input_dims,
                             output_dims)
        else:
            # General (non-CPTP) Kraus map
            super().__init__('Kraus', kraus, input_dims, output_dims)

    @property
    def data(self):
        """Return list of Kraus matrices for channel."""
        if self._data[1] is None:
            # If only a single Kraus set, don't return the tuple
            # Just the fist set
            return self._data[0]
        else:
            # Otherwise return the tuple of both kraus sets
            return self._data

    def is_cptp(self, atol=None, rtol=None):
        """Return True if completely-positive trace-preserving."""
        if self._data[1] is not None:
            return False
        if atol is None:
            atol = self._atol
        if rtol is None:
            rtol = self._rtol
        accum = 0j
        for op in self._data[0]:
            accum += np.dot(np.transpose(np.conj(op)), op)
        return is_identity_matrix(accum, rtol=rtol, atol=atol)

    def conjugate(self):
        """Return the conjugate of the QuantumChannel."""
        kraus_l, kraus_r = self._data
        kraus_l = [k.conj() for k in kraus_l]
        if kraus_r is not None:
            kraus_r = [k.conj() for k in kraus_r]
        return Kraus((kraus_l, kraus_r), self.input_dims(), self.output_dims())

    def transpose(self):
        """Return the transpose of the QuantumChannel."""
        kraus_l, kraus_r = self._data
        kraus_l = [k.T for k in kraus_l]
        if kraus_r is not None:
            kraus_r = [k.T for k in kraus_r]
        return Kraus((kraus_l, kraus_r),
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
            Kraus: The quantum channel self @ other.

        Raises:
            QiskitError: if other cannot be converted to a Kraus or has
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
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].

        Returns:
            Kraus: The quantum channel self * other.

        Raises:
            QiskitError: if other cannot be converted to a Kraus or has
            incompatible dimensions.
        """
        return super().dot(other, qargs=qargs)

    def power(self, n):
        """The matrix power of the channel.

        Args:
            n (int): compute the matrix power of the superoperator matrix.

        Returns:
            Kraus: the matrix power of the SuperOp converted to a Kraus channel.

        Raises:
            QiskitError: if the input and output dimensions of the
            QuantumChannel are not equal, or the power is not an integer.
        """
        if n > 0:
            return super().power(n)
        return Kraus(SuperOp(self).power(n))

    def tensor(self, other):
        """Return the tensor product channel self ⊗ other.

        Args:
            other (QuantumChannel): a quantum channel subclass.

        Returns:
            Kraus: the tensor product channel self ⊗ other as a Kraus
            object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        return self._tensor_product(other, reverse=False)

    def expand(self, other):
        """Return the tensor product channel other ⊗ self.

        Args:
            other (QuantumChannel): a quantum channel subclass.

        Returns:
            Kraus: the tensor product channel other ⊗ self as a Kraus
            object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        return self._tensor_product(other, reverse=True)

    def add(self, other):
        """Return the QuantumChannel self + other.

        Args:
            other (QuantumChannel): a quantum channel subclass.

        Returns:
            Kraus: the linear addition self + other as a Kraus object.

        Raises:
            QiskitError: if other cannot be converted to a channel, or
            has incompatible dimensions.
        """
        # Since we cannot directly add two channels in the Kraus
        # representation we try and use the other channels method
        # or convert to the Choi representation
        return Kraus(Choi(self).add(other))

    def subtract(self, other):
        """Return the QuantumChannel self - other.

        Args:
            other (QuantumChannel): a quantum channel subclass.

        Returns:
            Kraus: the linear subtraction self - other as Kraus object.

        Raises:
            QiskitError: if other cannot be converted to a channel, or
            has incompatible dimensions.
        """
        # Since we cannot directly subtract two channels in the Kraus
        # representation we try and use the other channels method
        # or convert to the Choi representation
        return Kraus(Choi(self).subtract(other))

    def multiply(self, other):
        """Return the QuantumChannel self + other.

        Args:
            other (complex): a complex number.

        Returns:
            Kraus: the scalar multiplication other * self as a Kraus object.

        Raises:
            QiskitError: if other is not a valid scalar.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        # If the number is complex we need to convert to general
        # kraus channel so we multiply via Choi representation
        if isinstance(other, complex) or other < 0:
            # Convert to Choi-matrix
            return Kraus(Choi(self).multiply(other))
        # If the number is real we can update the Kraus operators
        # directly
        val = np.sqrt(other)
        kraus_r = None
        kraus_l = [val * k for k in self._data[0]]
        if self._data[1] is not None:
            kraus_r = [val * k for k in self._data[1]]
        return Kraus((kraus_l, kraus_r), self._input_dim, self._output_dim)

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

    def _tensor_product(self, other, reverse=False):
        """Return the tensor product channel.

        Args:
            other (QuantumChannel): a quantum channel subclass.
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False
        Returns:
            Kraus: the tensor product channel as a Kraus object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        # Convert other to Kraus
        if not isinstance(other, Kraus):
            other = Kraus(other)

        # Get tensor matrix
        ka_l, ka_r = self._data
        kb_l, kb_r = other._data
        if reverse:
            input_dims = self.input_dims() + other.input_dims()
            output_dims = self.output_dims() + other.output_dims()
            kab_l = [np.kron(b_in, a_in) for a_in in ka_l for b_in in kb_l]
        else:
            input_dims = other.input_dims() + self.input_dims()
            output_dims = other.output_dims() + self.output_dims()
            kab_l = [np.kron(a, b) for a in ka_l for b in kb_l]
        if ka_r is None and kb_r is None:
            kab_r = None
        else:
            if ka_r is None:
                ka_r = ka_l
            if kb_r is None:
                kb_r = kb_l
            if reverse:
                kab_r = [np.kron(b_in, a_in) for a_in in ka_r for b_in in kb_r]
            else:
                kab_r = [np.kron(a, b) for a in ka_r for b in kb_r]
        data = (kab_l, kab_r)
        return Kraus(data, input_dims, output_dims)

    def _chanmul(self, other, qargs=None, left_multiply=False):
        """Multiply two quantum channels.

        Args:
            other (QuantumChannel): a quantum channel.
            qargs (list): a list of subsystem positions to compose other on.
            left_multiply (bool): If True return other * self
                                  If False return self * other [Default:False]

        Returns:
            Kraus: The composition channel as a Kraus object.

        Raises:
            QiskitError: if other is not a QuantumChannel subclass, or
            has incompatible dimensions.
        """
        if qargs is not None:
            return Kraus(
                SuperOp(self)._chanmul(other,
                                       qargs=qargs,
                                       left_multiply=left_multiply))

        if not isinstance(other, Kraus):
            other = Kraus(other)
        # Check dimensions match up
        if not left_multiply and self._input_dim != other._output_dim:
            raise QiskitError(
                'input_dim of self must match output_dim of other')
        if left_multiply and self._output_dim != other._input_dim:
            raise QiskitError(
                'input_dim of other must match output_dim of self')

        if left_multiply:
            ka_l, ka_r = other._data
            kb_l, kb_r = self._data
            input_dim = self._input_dim
            output_dim = other._output_dim
        else:
            ka_l, ka_r = self._data
            kb_l, kb_r = other._data
            input_dim = other._input_dim
            output_dim = self._output_dim

        kab_l = [np.dot(a, b) for a in ka_l for b in kb_l]
        if ka_r is None and kb_r is None:
            kab_r = None
        elif ka_r is None:
            kab_r = [np.dot(a, b) for a in ka_l for b in kb_r]
        elif kb_r is None:
            kab_r = [np.dot(a, b) for a in ka_r for b in kb_l]
        else:
            kab_r = [np.dot(a, b) for a in ka_r for b in kb_r]
        return Kraus((kab_l, kab_r), input_dim, output_dim)
