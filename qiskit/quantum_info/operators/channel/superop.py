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
Superoperator representation of a Quantum Channel."""

from numbers import Number
import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.channel.quantum_channel import QuantumChannel
from qiskit.quantum_info.operators.channel.transformations import _to_superop
from qiskit.quantum_info.operators.channel.transformations import _bipartite_tensor


class SuperOp(QuantumChannel):
    r"""Superoperator representation of a quantum channel.

    The Superoperator representation of a quantum channel :math:`\mathcal{E}`
    is a matrix :math:`S` such that the evolution of a
    :class:`~qiskit.quantum_info.DensityMatrix` :math:`\rho` is given by

    .. math::

        |\mathcal{E}(\rho)\rangle\!\rangle = S |\rho\rangle\!\rangle

    where the double-ket notation :math:`|A\rangle\!\rangle` denotes a vector
    formed by stacking the columns of the matrix :math:`A`
    *(column-vectorization)*.

    See reference [1] for further details.

    References:
        1. C.J. Wood, J.D. Biamonte, D.G. Cory, *Tensor networks and graphical calculus
           for open quantum systems*, Quant. Inf. Comp. 15, 0579-0811 (2015).
           `arXiv:1111.6950 [quant-ph] <https://arxiv.org/abs/1111.6950>`_
    """

    def __init__(self, data, input_dims=None, output_dims=None):
        """Initialize a quantum channel Superoperator operator.

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
            superoperator.

        Additional Information
        ----------------------
        If the input or output dimensions are None, they will be
        automatically determined from the input data. If the input data is
        a Numpy array of shape (4**N, 4**N) qubit systems will be used. If
        the input operator is not an N-qubit operator, it will assign a
        single subsystem with dimension specified by the shape of the input.
        """
        # If the input is a raw list or matrix we assume that it is
        # already a superoperator.
        if isinstance(data, (list, np.ndarray)):
            # We initialize directly from superoperator matrix
            super_mat = np.array(data, dtype=complex)
            # Determine total input and output dimensions
            dout, din = super_mat.shape
            input_dim = int(np.sqrt(din))
            output_dim = int(np.sqrt(dout))
            if output_dim**2 != dout or input_dim**2 != din:
                raise QiskitError("Invalid shape for SuperOp matrix.")
        else:
            # Otherwise we initialize by conversion from another Qiskit
            # object into the QuantumChannel.
            if isinstance(data, (QuantumCircuit, Instruction)):
                # If the input is a Terra QuantumCircuit or Instruction we
                # perform a simulation to construct the circuit superoperator.
                # This will only work if the circuit or instruction can be
                # defined in terms of instructions which have no classical
                # register components. The instructions can be gates, reset,
                # or Kraus instructions. Any conditional gates or measure
                # will cause an exception to be raised.
                data = self._init_instruction(data)
            else:
                # We use the QuantumChannel init transform to initialize
                # other objects into a QuantumChannel or Operator object.
                data = self._init_transformer(data)
            # Now that the input is an operator we convert it to a
            # SuperOp object
            input_dim, output_dim = data.dim
            super_mat = _to_superop(data.rep, data._data, input_dim,
                                    output_dim)
            if input_dims is None:
                input_dims = data.input_dims()
            if output_dims is None:
                output_dims = data.output_dims()
        # Finally we format and validate the channel input and
        # output dimensions
        input_dims = self._automatic_dims(input_dims, input_dim)
        output_dims = self._automatic_dims(output_dims, output_dim)
        super().__init__('SuperOp', super_mat, input_dims, output_dims)

    @property
    def _shape(self):
        """Return the tensor shape of the superoperator matrix"""
        return 2 * tuple(reversed(self.output_dims())) + 2 * tuple(
            reversed(self.input_dims()))

    @property
    def _bipartite_shape(self):
        """Return the shape for bipartite matrix"""
        return (self._output_dim, self._output_dim, self._input_dim,
                self._input_dim)

    def conjugate(self):
        """Return the conjugate of the QuantumChannel."""
        return SuperOp(np.conj(self._data), self.input_dims(),
                       self.output_dims())

    def transpose(self):
        """Return the transpose of the QuantumChannel."""
        return SuperOp(np.transpose(self._data),
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
            SuperOp: The quantum channel self @ other.

        Raises:
            QiskitError: if other cannot be converted to a SuperOp or has
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
            SuperOp: The quantum channel self * other.

        Raises:
            QiskitError: if other cannot be converted to a SuperOp or has
            incompatible dimensions.
        """
        return super().dot(other, qargs=qargs)

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
        return SuperOp(np.linalg.matrix_power(self._data, n),
                       self.input_dims(), self.output_dims())

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
        # Convert other to SuperOp
        if not isinstance(other, SuperOp):
            other = SuperOp(other)

        input_dims = other.input_dims() + self.input_dims()
        output_dims = other.output_dims() + self.output_dims()
        data = _bipartite_tensor(self._data,
                                 other.data,
                                 shape1=self._bipartite_shape,
                                 shape2=other._bipartite_shape)
        return SuperOp(data, input_dims, output_dims)

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
        # Convert other to SuperOp
        if not isinstance(other, SuperOp):
            other = SuperOp(other)

        input_dims = self.input_dims() + other.input_dims()
        output_dims = self.output_dims() + other.output_dims()
        data = _bipartite_tensor(other.data,
                                 self._data,
                                 shape1=other._bipartite_shape,
                                 shape2=self._bipartite_shape)
        return SuperOp(data, input_dims, output_dims)

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
        # Prevent cyclic imports by importing DensityMatrix here
        # pylint: disable=cyclic-import
        from qiskit.quantum_info.states.densitymatrix import DensityMatrix

        if not isinstance(state, DensityMatrix):
            state = DensityMatrix(state)

        if qargs is None:
            # Evolution on full matrix
            if state._dim != self._input_dim:
                raise QiskitError(
                    "Operator input dimension is not equal to density matrix dimension."
                )
            # We reshape in column-major vectorization (Fortran order in Numpy)
            # since that is how the SuperOp is defined
            vec = np.ravel(state.data, order='F')
            mat = np.reshape(np.dot(self.data, vec),
                             (self._output_dim, self._output_dim),
                             order='F')
            return DensityMatrix(mat, dims=self.output_dims())
        # Otherwise we are applying an operator only to subsystems
        # Check dimensions of subsystems match the operator
        if state.dims(qargs) != self.input_dims():
            raise QiskitError(
                "Operator input dimensions are not equal to statevector subsystem dimensions."
            )
        # Reshape statevector and operator
        tensor = np.reshape(state.data, state._shape)
        mat = np.reshape(self.data, self._shape)
        # Construct list of tensor indices of statevector to be contracted
        num_indices = len(state.dims())
        indices = [num_indices - 1 - qubit for qubit in qargs
                   ] + [2 * num_indices - 1 - qubit for qubit in qargs]
        tensor = self._einsum_matmul(tensor, mat, indices)
        # Replace evolved dimensions
        new_dims = list(state.dims())
        for i, qubit in enumerate(qargs):
            new_dims[qubit] = self._output_dims[i]
        new_dim = np.product(new_dims)
        # reshape tensor to density matrix
        tensor = np.reshape(tensor, (new_dim, new_dim))
        return DensityMatrix(tensor, dims=new_dims)

    def _chanmul(self, other, qargs=None, left_multiply=False):
        """Multiply two quantum channels.

        Args:
            other (QuantumChannel): a quantum channel.
            qargs (list): a list of subsystem positions to compose other on.
            left_multiply (bool): If True return other * self
                                  If False return self * other [Default:False]

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
        if left_multiply and self.output_dims(
                qargs=qargs) != other.input_dims():
            raise QiskitError(
                'input_dims of other must match subsystem output_dims')
        if not left_multiply and self.input_dims(
                qargs=qargs) != other.output_dims():
            raise QiskitError(
                'output_dims of other must match subsystem input_dims')

        # Full composition of superoperators
        if qargs is None:
            if left_multiply:
                # other * self
                return SuperOp(np.dot(other.data, self._data),
                               input_dims=self.input_dims(),
                               output_dims=other.output_dims())
            # self * other
            return SuperOp(np.dot(self._data, other.data),
                           input_dims=other.input_dims(),
                           output_dims=self.output_dims())
        # Composition on subsystem
        return self._chanmul_subsystem(other, qargs, left_multiply)

    def _chanmul_subsystem(self, other, qargs, left_multiply=False):
        """Matrix multiply on subsystem."""
        # Compute tensor contraction indices from qargs
        input_dims = list(self.input_dims())
        output_dims = list(self.output_dims())
        if left_multiply:
            num_indices = len(self.output_dims())
            shift = 0
            right_mul = False
            for pos, qubit in enumerate(qargs):
                output_dims[qubit] = other._output_dims[pos]
        else:
            num_indices = len(self.input_dims())
            shift = 2 * len(self.output_dims())
            right_mul = True
            for pos, qubit in enumerate(qargs):
                input_dims[qubit] = other._input_dims[pos]
        # Reshape current matrix
        # Note that we must reverse the subsystem dimension order as
        # qubit 0 corresponds to the right-most position in the tensor
        # product, which is the last tensor wire index.
        tensor = np.reshape(self.data, self._shape)
        mat = np.reshape(other.data, other._shape)
        # Add first set of indices
        indices = [2 * num_indices - 1 - qubit for qubit in qargs
                   ] + [num_indices - 1 - qubit for qubit in qargs]
        final_shape = [np.product(output_dims)**2, np.product(input_dims)**2]
        data = np.reshape(
            self._einsum_matmul(tensor, mat, indices, shift, right_mul),
            final_shape)
        return SuperOp(data, input_dims, output_dims)

    def _compose_subsystem(self, other, qargs, front=False):
        """Return the composition channel."""
        # Compute tensor contraction indices from qargs
        input_dims = list(self.input_dims())
        output_dims = list(self.output_dims())
        if front:
            num_indices = len(self.input_dims())
            shift = 2 * len(self.output_dims())
            right_mul = True
            for pos, qubit in enumerate(qargs):
                input_dims[qubit] = other._input_dims[pos]
        else:
            num_indices = len(self.output_dims())
            shift = 0
            right_mul = False
            for pos, qubit in enumerate(qargs):
                output_dims[qubit] = other._output_dims[pos]
        # Reshape current matrix
        # Note that we must reverse the subsystem dimension order as
        # qubit 0 corresponds to the right-most position in the tensor
        # product, which is the last tensor wire index.
        tensor = np.reshape(self.data, self._shape)
        mat = np.reshape(other.data, other._shape)
        # Add first set of indices
        indices = [2 * num_indices - 1 - qubit for qubit in qargs
                   ] + [num_indices - 1 - qubit for qubit in qargs]
        final_shape = [np.product(output_dims)**2, np.product(input_dims)**2]
        data = np.reshape(
            self._einsum_matmul(tensor, mat, indices, shift, right_mul),
            final_shape)
        return SuperOp(data, input_dims, output_dims)

    @classmethod
    def _init_instruction(cls, instruction):
        """Convert a QuantumCircuit or Instruction to a SuperOp."""
        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        # Initialize an identity superoperator of the correct size
        # of the circuit
        op = SuperOp(np.eye(4**instruction.num_qubits))
        op._append_instruction(instruction)
        return op

    @classmethod
    def _instruction_to_superop(cls, obj):
        """Return superop for instruction if defined or None otherwise."""
        if not isinstance(obj, Instruction):
            raise QiskitError('Input is not an instruction.')
        chan = None
        if obj.name == 'reset':
            # For superoperator evolution we can simulate a reset as
            # a non-unitary superoperator matrix
            chan = SuperOp(
                np.array([[1, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0],
                          [0, 0, 0, 0]]))
        if obj.name == 'kraus':
            kraus = obj.params
            dim = len(kraus[0])
            chan = SuperOp(_to_superop('Kraus', (kraus, None), dim, dim))
        elif hasattr(obj, 'to_matrix'):
            # If instruction is a gate first we see if it has a
            # `to_matrix` definition and if so use that.
            try:
                kraus = [obj.to_matrix()]
                dim = len(kraus[0])
                chan = SuperOp(_to_superop('Kraus', (kraus, None), dim, dim))
            except QiskitError:
                pass
        return chan

    def _append_instruction(self, obj, qargs=None):
        """Update the current Operator by apply an instruction."""
        chan = self._instruction_to_superop(obj)
        if chan is not None:
            # Perform the composition and inplace update the current state
            # of the operator
            op = self.compose(chan, qargs=qargs)
            self._data = op.data
        else:
            # If the instruction doesn't have a matrix defined we use its
            # circuit decomposition definition if it exists, otherwise we
            # cannot compose this gate and raise an error.
            if obj.definition is None:
                raise QiskitError('Cannot apply Instruction: {}'.format(
                    obj.name))
            for instr, qregs, cregs in obj.definition:
                if cregs:
                    raise QiskitError(
                        'Cannot apply instruction with classical registers: {}'
                        .format(instr.name))
                # Get the integer position of the flat register
                if qargs is None:
                    new_qargs = [tup.index for tup in qregs]
                else:
                    new_qargs = [qargs[tup.index] for tup in qregs]
                self._append_instruction(instr, qargs=new_qargs)
