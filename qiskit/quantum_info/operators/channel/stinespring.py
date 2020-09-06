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
Stinespring representation of a Quantum Channel.
"""

import copy
from numbers import Number
import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_identity_matrix
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.kraus import Kraus
from qiskit.quantum_info.operators.channel.choi import Choi
from qiskit.quantum_info.operators.channel.superop import SuperOp
from qiskit.quantum_info.operators.channel.transformations import _to_stinespring


class Stinespring(QuantumChannel):
    r"""Stinespring representation of a quantum channel.

    The Stinespring representation of a quantum channel :math:`\mathcal{E}`
    is a rectangular matrix :math:`A` such that the evolution of a
    :class:`~qiskit.quantum_info.DensityMatrix` :math:`\rho` is given by

    .. math::

        \mathcal{E}(ρ) = \mbox{Tr}_2\left[A ρ A^\dagger\right]

    where :math:`\mbox{Tr}_2` is the :func:`partial_trace` over subsystem 2.

    A general operator map :math:`\mathcal{G}` can also be written using the
    generalized Stinespring representation which is given by two matrices
    :math:`A`, :math:`B` such that

    .. math::

        \mathcal{G}(ρ) = \mbox{Tr}_2\left[A ρ B^\dagger\right]

    See reference [1] for further details.

    References:
        1. C.J. Wood, J.D. Biamonte, D.G. Cory, *Tensor networks and graphical calculus
           for open quantum systems*, Quant. Inf. Comp. 15, 0579-0811 (2015).
           `arXiv:1111.6950 [quant-ph] <https://arxiv.org/abs/1111.6950>`_
    """

    def __init__(self, data, input_dims=None, output_dims=None):
        """Initialize a quantum channel Stinespring operator.

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

        Additional Information:
            If the input or output dimensions are None, they will be
            automatically determined from the input data. This can fail for the
            Stinespring operator if the output dimension cannot be automatically
            determined.
        """
        # If the input is a list or tuple we assume it is a pair of general
        # Stinespring matrices. If it is a numpy array we assume that it is
        # a single Stinespring matrix.
        if isinstance(data, (list, tuple, np.ndarray)):
            if not isinstance(data, tuple):
                # Convert single Stinespring set to length 1 tuple
                stine = (np.asarray(data, dtype=complex), None)
            if isinstance(data, tuple) and len(data) == 2:
                if data[1] is None:
                    stine = (np.asarray(data[0], dtype=complex), None)
                else:
                    stine = (np.asarray(data[0], dtype=complex),
                             np.asarray(data[1], dtype=complex))

            dim_left, dim_right = stine[0].shape
            # If two Stinespring matrices check they are same shape
            if stine[1] is not None:
                if stine[1].shape != (dim_left, dim_right):
                    raise QiskitError("Invalid Stinespring input.")
            input_dim = dim_right
            if output_dims:
                output_dim = np.product(output_dims)
            else:
                output_dim = input_dim
            if dim_left % output_dim != 0:
                raise QiskitError("Invalid output_dim")
        else:
            # Otherwise we initialize by conversion from another Qiskit
            # object into the QuantumChannel.
            if isinstance(data, (QuantumCircuit, Instruction)):
                # If the input is a Terra QuantumCircuit or Instruction we
                # convert it to a SuperOp
                data = SuperOp._init_instruction(data)
            else:
                # We use the QuantumChannel init transform to intialize
                # other objects into a QuantumChannel or Operator object.
                data = self._init_transformer(data)
            data = self._init_transformer(data)
            input_dim, output_dim = data.dim
            # Now that the input is an operator we convert it to a
            # Stinespring operator
            rep = getattr(data, '_channel_rep', 'Operator')
            stine = _to_stinespring(rep, data._data, input_dim, output_dim)
            if input_dims is None:
                input_dims = data.input_dims()
            if output_dims is None:
                output_dims = data.output_dims()

        # Check and format input and output dimensions
        input_dims = self._automatic_dims(input_dims, input_dim)
        output_dims = self._automatic_dims(output_dims, output_dim)
        # Initialize either single or general Stinespring
        if stine[1] is None or (stine[1] == stine[0]).all():
            # Standard Stinespring map
            super().__init__((stine[0], None),
                             input_dims=input_dims,
                             output_dims=output_dims,
                             channel_rep='Stinespring')
        else:
            # General (non-CPTP) Stinespring map
            super().__init__(stine,
                             input_dims=input_dims,
                             output_dims=output_dims,
                             channel_rep='Stinespring')

    @property
    def data(self):
        # Override to deal with data being either tuple or not
        if self._data[1] is None:
            return self._data[0]
        else:
            return self._data

    def is_cptp(self, atol=None, rtol=None):
        """Return True if completely-positive trace-preserving."""
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        if self._data[1] is not None:
            return False
        check = np.dot(np.transpose(np.conj(self._data[0])), self._data[0])
        return is_identity_matrix(check, rtol=self.rtol, atol=self.atol)

    def conjugate(self):
        """Return the conjugate of the QuantumChannel."""
        # pylint: disable=assignment-from-no-return
        stine_l = np.conjugate(self._data[0])
        stine_r = None
        if self._data[1] is not None:
            stine_r = np.conjugate(self._data[1])
        return Stinespring((stine_l, stine_r), self.input_dims(),
                           self.output_dims())

    def transpose(self):
        """Return the transpose of the QuantumChannel."""
        din, dout = self.dim
        dtr = self._data[0].shape[0] // dout
        stine = [None, None]
        for i, mat in enumerate(self._data):
            if mat is not None:
                stine[i] = np.reshape(
                    np.transpose(np.reshape(mat, (dout, dtr, din)), (2, 1, 0)),
                    (din * dtr, dout))
        return Stinespring(tuple(stine),
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
            Stinespring: The quantum channel self @ other.

        Raises:
            QiskitError: if other cannot be converted to a Stinespring or has
                         incompatible dimensions.

        Additional Information:
            Composition (``@``) is defined as `left` matrix multiplication for
            :class:`SuperOp` matrices. That is that ``A @ B`` is equal to ``B * A``.
            Setting ``front=True`` returns `right` matrix multiplication
            ``A * B`` and is equivalent to the :meth:`dot` method.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if qargs is not None:
            return Stinespring(
                SuperOp(self).compose(other, qargs=qargs, front=front))

        # Otherwise we convert via Kraus representation rather than
        # superoperator to avoid unnecessary representation conversions
        return Stinespring(Kraus(self).compose(other, front=front))

    def dot(self, other, qargs=None):
        """Return the right multiplied quantum channel self * other.

        Args:
            other (QuantumChannel): a quantum channel.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].

        Returns:
            Stinespring: The quantum channel self * other.

        Raises:
            QiskitError: if other cannot be converted to a Stinespring or has
                         incompatible dimensions.
        """
        return super().dot(other, qargs=qargs)

    def power(self, n):
        """The matrix power of the channel.

        Args:
            n (int): compute the matrix power of the superoperator matrix.

        Returns:
            Stinespring: the matrix power of the SuperOp converted to a
            Stinespring channel.

        Raises:
            QiskitError: if the input and output dimensions of the
                         QuantumChannel are not equal, or the power is not
                         an integer.
        """
        if n > 0:
            return super().power(n)
        return Stinespring(SuperOp(self).power(n))

    def tensor(self, other):
        """Return the tensor product channel self ⊗ other.

        Args:
            other (QuantumChannel): a quantum channel subclass.

        Returns:
            Stinespring: the tensor product channel other ⊗ self as a
            Stinespring object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        return self._tensor_product(other, reverse=False)

    def expand(self, other):
        """Return the tensor product channel other ⊗ self.

        Args:
            other (QuantumChannel): a quantum channel subclass.

        Returns:
            Stinespring: the tensor product channel other ⊗ self as a
            Stinespring object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        return self._tensor_product(other, reverse=True)

    def _add(self, other, qargs=None):
        """Return the QuantumChannel self + other.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (QuantumChannel): a quantum channel subclass.
            qargs (None or list): optional subsystems to add on
                                  (Default: None)

        Returns:
            Stinespring: the linear addition channel self + other.

        Raises:
            QiskitError: if other cannot be converted to a channel or
                         has incompatible dimensions.
        """
        # Since we cannot directly add two channels in the Stinespring
        # representation we convert to the Choi representation
        return Stinespring(Choi(self)._add(other, qargs=qargs))

    def _multiply(self, other):
        """Return the QuantumChannel other * self.

        Args:
            other (complex): a complex number.

        Returns:
            Stinespring: the scalar multiplication other * self.

        Raises:
            QiskitError: if other is not a valid scalar.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")

        ret = copy.copy(self)
        # If the number is complex or negative we need to convert to
        # general Stinespring representation so we first convert to
        # the Choi representation
        if isinstance(other, complex) or other < 1:
            # Convert to Choi-matrix
            ret._data = Stinespring(Choi(self)._multiply(other))._data
            return ret
        # If the number is real we can update the Kraus operators
        # directly
        num = np.sqrt(other)
        stine_l, stine_r = self._data
        stine_l = num * self._data[0]
        stine_r = None
        if self._data[1] is not None:
            stine_r = num * self._data[1]
        ret._data = (stine_l, stine_r)
        return ret

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
                            if True return (other ⊗ self) [Default: False]
        Returns:
            Stinespring: the tensor product channel as a Stinespring object.

        Raises:
            QiskitError: if other cannot be converted to a channel.
        """
        # Convert other to Stinespring
        if not isinstance(other, Stinespring):
            other = Stinespring(other)

        # Tensor Stinespring ops
        sa_l, sa_r = self._data
        sb_l, sb_r = other._data

        # Reshuffle tensor dimensions
        din_a, dout_a = self.dim
        din_b, dout_b = other.dim
        dtr_a = sa_l.shape[0] // dout_a
        dtr_b = sb_l.shape[0] // dout_b
        if reverse:
            shape_in = (dout_b, dtr_b, dout_a, dtr_a, din_b * din_a)
            shape_out = (dout_b * dtr_b * dout_a * dtr_a, din_b * din_a)
        else:
            shape_in = (dout_a, dtr_a, dout_b, dtr_b, din_a * din_b)
            shape_out = (dout_a * dtr_a * dout_b * dtr_b, din_a * din_b)

        # Compute left Stinespring op
        if reverse:
            input_dims = self.input_dims() + other.input_dims()
            output_dims = self.output_dims() + other.output_dims()
            sab_l = np.kron(sb_l, sa_l)
        else:
            input_dims = other.input_dims() + self.input_dims()
            output_dims = other.output_dims() + self.output_dims()
            sab_l = np.kron(sa_l, sb_l)
        # Reravel indices
        sab_l = np.reshape(
            np.transpose(np.reshape(sab_l, shape_in), (0, 2, 1, 3, 4)),
            shape_out)

        # Compute right Stinespring op
        if sa_r is None and sb_r is None:
            sab_r = None
        else:
            if sa_r is None:
                sa_r = sa_l
            elif sb_r is None:
                sb_r = sb_l
            if reverse:
                sab_r = np.kron(sb_r, sa_r)
            else:
                sab_r = np.kron(sa_r, sb_r)
            # Reravel indices
            sab_r = np.reshape(
                np.transpose(np.reshape(sab_r, shape_in), (0, 2, 1, 3, 4)),
                shape_out)
        return Stinespring((sab_l, sab_r), input_dims, output_dims)
