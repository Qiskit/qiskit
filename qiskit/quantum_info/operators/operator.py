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
Matrix Operator class.
"""

import re
from numbers import Number

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.extensions.standard import IdGate, XGate, YGate, ZGate, HGate, SGate, TGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_unitary_matrix, matrix_equal
from qiskit.quantum_info.operators.base_operator import BaseOperator


class Operator(BaseOperator):
    r"""Matrix operator class

    This represents a matrix operator :math:`M` that will
    :meth:`~Statevector.evolve` a :class:`Statevector` :math:`|\psi\rangle`
    by matrix-vector multiplication

    .. math::

        |\psi\rangle \mapsto M|\psi\rangle,

    and will :meth:`~DensityMatrix.evolve` a :class:`DensityMatrix` :math:`\rho`
    by left and right multiplication

    .. math::

        \rho \mapsto M \rho M^\dagger.
    """

    def __init__(self, data, input_dims=None, output_dims=None):
        """Initialize an operator object.

        Args:
            data (QuantumCircuit or
                  Instruction or
                  BaseOperator or
                  matrix): data to initialize operator.
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
        single subsystem with dimension specified by the shape of the input.
        """
        if isinstance(data, (QuantumCircuit, Instruction)):
            # If the input is a Terra QuantumCircuit or Instruction we
            # perform a simulation to construct the unitary operator.
            # This will only work if the circuit or instruction can be
            # defined in terms of unitary gate instructions which have a
            # 'to_matrix' method defined. Any other instructions such as
            # conditional gates, measure, or reset will cause an
            # exception to be raised.
            mat = self._init_instruction(data).data
        elif hasattr(data, 'to_operator'):
            # If the data object has a 'to_operator' attribute this is given
            # higher preference than the 'to_matrix' method for initializing
            # an Operator object.
            data = data.to_operator()
            mat = data.data
            if input_dims is None:
                input_dims = data.input_dims()
            if output_dims is None:
                output_dims = data.output_dims()
        elif hasattr(data, 'to_matrix'):
            # If no 'to_operator' attribute exists we next look for a
            # 'to_matrix' attribute to a matrix that will be cast into
            # a complex numpy matrix.
            mat = np.array(data.to_matrix(), dtype=complex)
        elif isinstance(data, (list, np.ndarray)):
            # Finally we check if the input is a raw matrix in either a
            # python list or numpy array format.
            mat = np.array(data, dtype=complex)
        else:
            raise QiskitError("Invalid input data format for Operator")
        # Determine input and output dimensions
        dout, din = mat.shape
        output_dims = self._automatic_dims(output_dims, dout)
        input_dims = self._automatic_dims(input_dims, din)
        super().__init__('Operator', mat, input_dims, output_dims)

    @classmethod
    def from_label(cls, label):
        """Return a tensor product of single-qubit operators.

        Args:
            label (string): single-qubit operator string.

        Returns:
            Operator: The N-qubit operator.

        Raises:
            QiskitError: if the label contains invalid characters, or the length
            of the label is larger than an explicitly specified num_qubits.

        Additional Information:
            The labels correspond to the single-qubit matrices:
            'I': [[1, 0], [0, 1]]
            'X': [[0, 1], [1, 0]]
            'Y': [[0, -1j], [1j, 0]]
            'Z': [[1, 0], [0, -1]]
            'H': [[1, 1], [1, -1]] / sqrt(2)
            'S': [[1, 0], [0 , 1j]]
            'T': [[1, 0], [0, (1+1j) / sqrt(2)]]
            '0': [[1, 0], [0, 0]]
            '1': [[0, 0], [0, 1]]
            '+': [[0.5, 0.5], [0.5 , 0.5]]
            '-': [[0.5, -0.5], [-0.5 , 0.5]]
            'r': [[0.5, -0.5j], [0.5j , 0.5]]
            'l': [[0.5, 0.5j], [-0.5j , 0.5]]
        """
        # Check label is valid
        label_mats = {
            'I': IdGate().to_matrix(),
            'X': XGate().to_matrix(),
            'Y': YGate().to_matrix(),
            'Z': ZGate().to_matrix(),
            'H': HGate().to_matrix(),
            'S': SGate().to_matrix(),
            'T': TGate().to_matrix(),
            '0': np.array([[1, 0], [0, 0]], dtype=complex),
            '1': np.array([[0, 0], [0, 1]], dtype=complex),
            '+': np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex),
            '-': np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=complex),
            'r': np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=complex),
            'l': np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=complex),
        }
        if re.match(r'^[IXYZHST01rl\-+]+$', label) is None:
            raise QiskitError('Label contains invalid characters.')
        # Initialize an identity matrix and apply each gate
        num_qubits = len(label)
        op = Operator(np.eye(2 ** num_qubits, dtype=complex))
        for qubit, char in enumerate(reversed(label)):
            if char != 'I':
                op = op.compose(label_mats[char], qargs=[qubit])
        return op

    def is_unitary(self, atol=None, rtol=None):
        """Return True if operator is a unitary matrix."""
        if atol is None:
            atol = self._atol
        if rtol is None:
            rtol = self._rtol
        return is_unitary_matrix(self._data, rtol=rtol, atol=atol)

    def to_operator(self):
        """Convert operator to matrix operator class"""
        return self

    def to_instruction(self):
        """Convert to a UnitaryGate instruction."""
        from qiskit.extensions.unitary import UnitaryGate
        return UnitaryGate(self.data)

    def conjugate(self):
        """Return the conjugate of the operator."""
        return Operator(np.conj(self.data),
                        input_dims=self.input_dims(),
                        output_dims=self.output_dims())

    def transpose(self):
        """Return the transpose of the operator."""
        return Operator(np.transpose(self.data),
                        input_dims=self.output_dims(),
                        output_dims=self.input_dims())

    def compose(self, other, qargs=None, front=False):
        """Return the composed operator.

        Args:
            other (Operator): an operator object.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].
            front (bool): If True compose using right operator multiplication,
                          instead of left multiplication [default: False].

        Returns:
            Operator: The operator self @ other.

        Additional Information:
            Composition (``@``) is defined as `left` matrix multiplication for
            matrix operators. That is that ``A @ B`` is equal to ``B * A``.
            Setting ``front=True`` returns `right` matrix multiplication
            ``A * B`` and is equivalent to the :meth:`dot` method.
        """
        if front:
            return self._matmul(other, qargs, left_multiply=False)
        return self._matmul(other, qargs, left_multiply=True)

    def dot(self, other, qargs=None):
        """Return the right multiplied operator self * other.

        Args:
            other (Operator): an operator object.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].

        Returns:
            Operator: The operator self * other.

        Raises:
            QiskitError: if other cannot be converted to an Operator or has
            incompatible dimensions.
        """
        return super().dot(other, qargs=qargs)

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
        if not isinstance(other, Operator):
            other = Operator(other)
        input_dims = other.input_dims() + self.input_dims()
        output_dims = other.output_dims() + self.output_dims()
        data = np.kron(self._data, other._data)
        return Operator(data, input_dims, output_dims)

    def expand(self, other):
        """Return the tensor product operator other ⊗ self.

        Args:
            other (Operator): an operator object.

        Returns:
            Operator: the tensor product operator other ⊗ self.

        Raises:
            QiskitError: if other cannot be converted to an operator.
        """
        if not isinstance(other, Operator):
            other = Operator(other)
        input_dims = self.input_dims() + other.input_dims()
        output_dims = self.output_dims() + other.output_dims()
        data = np.kron(other._data, self._data)
        return Operator(data, input_dims, output_dims)

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

    def equiv(self, other, rtol=None, atol=None):
        """Return True if operators are equivalent up to global phase.

        Args:
            other (Operator): an operator object.
            rtol (float): relative tolerance value for comparison.
            atol (float): absolute tolerance value for comparison.

        Returns:
            bool: True if operators are equivalent up to global phase.
        """
        if not isinstance(other, Operator):
            try:
                other = Operator(other)
            except QiskitError:
                return False
        if self.dim != other.dim:
            return False
        if atol is None:
            atol = self._atol
        if rtol is None:
            rtol = self._rtol
        return matrix_equal(self.data, other.data, ignore_phase=True,
                            rtol=rtol, atol=atol)

    @property
    def _shape(self):
        """Return the tensor shape of the matrix operator"""
        return tuple(reversed(self.output_dims())) + tuple(
            reversed(self.input_dims()))

    def _matmul(self, other, qargs=None, left_multiply=False):
        """Matrix multiply two operators

        Args:
            other (Operator): an operator object.
            qargs (list): a list of subsystem positions to compose other on.
            left_multiply (bool): If True return other * self
                                  If False return self * other [Default:False]
        Returns:
            Operator: The output operator.

        Raises:
            QiskitError: if other cannot be converted to an Operator or has
            incompatible dimensions.
        """
        # Convert to Operator
        if not isinstance(other, Operator):
            other = Operator(other)
        # Check dimensions are compatible
        if not left_multiply and self.input_dims(qargs=qargs) != other.output_dims():
            raise QiskitError(
                'output_dims of other must match subsystem input_dims')
        if left_multiply and self.output_dims(qargs=qargs) != other.input_dims():
            raise QiskitError(
                'input_dims of other must match subsystem output_dims')
        # Full composition of operators
        if qargs is None:
            if left_multiply:
                # Composition other * self
                input_dims = self.input_dims()
                output_dims = other.output_dims()
                data = np.dot(other.data, self._data)
            else:
                # Composition self * other
                input_dims = other.input_dims()
                output_dims = self.output_dims()
                data = np.dot(self._data, other.data)
            return Operator(data, input_dims, output_dims)
        # Compose with other on subsystem
        return self._matmul_subsystem(other, qargs, left_multiply)

    def _matmul_subsystem(self, other, qargs, left_multiply=False):
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
            shift = len(self.output_dims())
            right_mul = True
            for pos, qubit in enumerate(qargs):
                input_dims[qubit] = other._input_dims[pos]
        # Reshape current matrix
        # Note that we must reverse the subsystem dimension order as
        # qubit 0 corresponds to the right-most position in the tensor
        # product, which is the last tensor wire index.
        tensor = np.reshape(self.data, self._shape)
        mat = np.reshape(other.data, other._shape)
        indices = [num_indices - 1 - qubit for qubit in qargs]
        final_shape = [np.product(output_dims), np.product(input_dims)]
        data = np.reshape(
            self._einsum_matmul(tensor, mat, indices, shift, right_mul),
            final_shape)
        return Operator(data, input_dims, output_dims)

    def _compose_subsystem(self, other, qargs, front=False):
        """Return the composition channel."""
        # Compute tensor contraction indices from qargs
        input_dims = list(self.input_dims())
        output_dims = list(self.output_dims())
        if front:
            num_indices = len(self.input_dims())
            shift = len(self.output_dims())
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
        indices = [num_indices - 1 - qubit for qubit in qargs]
        final_shape = [np.product(output_dims), np.product(input_dims)]
        data = np.reshape(
            self._einsum_matmul(tensor, mat, indices, shift, right_mul),
            final_shape)
        return Operator(data, input_dims, output_dims)

    @classmethod
    def _init_instruction(cls, instruction):
        """Convert a QuantumCircuit or Instruction to an Operator."""
        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        # Initialize an identity operator of the correct size of the circuit
        op = Operator(np.eye(2 ** instruction.num_qubits))
        op._append_instruction(instruction)
        return op

    @classmethod
    def _instruction_to_matrix(cls, obj):
        """Return Operator for instruction if defined or None otherwise."""
        if not isinstance(obj, Instruction):
            raise QiskitError('Input is not an instruction.')
        mat = None
        if hasattr(obj, 'to_matrix'):
            # If instruction is a gate first we see if it has a
            # `to_matrix` definition and if so use that.
            try:
                mat = obj.to_matrix()
            except QiskitError:
                pass
        return mat

    def _append_instruction(self, obj, qargs=None):
        """Update the current Operator by apply an instruction."""
        mat = self._instruction_to_matrix(obj)
        if mat is not None:
            # Perform the composition and inplace update the current state
            # of the operator
            op = self.compose(mat, qargs=qargs)
            self._data = op.data
        else:
            # If the instruction doesn't have a matrix defined we use its
            # circuit decomposition definition if it exists, otherwise we
            # cannot compose this gate and raise an error.
            if obj.definition is None:
                raise QiskitError('Cannot apply Instruction: {}'.format(obj.name))
            for instr, qregs, cregs in obj.definition:
                if cregs:
                    raise QiskitError(
                        'Cannot apply instruction with classical registers: {}'.format(
                            instr.name))
                # Get the integer position of the flat register
                if qargs is None:
                    new_qargs = [tup.index for tup in qregs]
                else:
                    new_qargs = [qargs[tup.index] for tup in qregs]
                self._append_instruction(instr, qargs=new_qargs)
