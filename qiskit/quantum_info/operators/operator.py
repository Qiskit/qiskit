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

import copy
import re
from numbers import Number

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate, HGate, SGate, TGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.predicates import is_unitary_matrix, matrix_equal
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.mixins import generate_apidocs


class Operator(LinearOp):
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

        Additional Information:
            If the input or output dimensions are None, they will be
            automatically determined from the input data. If the input data is
            a Numpy array of shape (2**N, 2**N) qubit systems will be used. If
            the input operator is not an N-qubit operator, it will assign a
            single subsystem with dimension specified by the shape of the input.
        """
        op_shape = None
        if isinstance(data, (list, np.ndarray)):
            # Default initialization from list or numpy array matrix
            self._data = np.asarray(data, dtype=complex)
        elif isinstance(data, (QuantumCircuit, Instruction)):
            # If the input is a Terra QuantumCircuit or Instruction we
            # perform a simulation to construct the unitary operator.
            # This will only work if the circuit or instruction can be
            # defined in terms of unitary gate instructions which have a
            # 'to_matrix' method defined. Any other instructions such as
            # conditional gates, measure, or reset will cause an
            # exception to be raised.
            self._data = self._init_instruction(data).data
        elif hasattr(data, "to_operator"):
            # If the data object has a 'to_operator' attribute this is given
            # higher preference than the 'to_matrix' method for initializing
            # an Operator object.
            data = data.to_operator()
            self._data = data.data
            op_shape = data._op_shape
        elif hasattr(data, "to_matrix"):
            # If no 'to_operator' attribute exists we next look for a
            # 'to_matrix' attribute to a matrix that will be cast into
            # a complex numpy matrix.
            self._data = np.asarray(data.to_matrix(), dtype=complex)
        else:
            raise QiskitError("Invalid input data format for Operator")

        super().__init__(
            op_shape=op_shape,
            input_dims=input_dims,
            output_dims=output_dims,
            shape=self._data.shape,
        )

    def __array__(self, dtype=None):
        if dtype:
            return np.asarray(self.data, dtype=dtype)
        return self.data

    def __repr__(self):
        prefix = "Operator("
        pad = len(prefix) * " "
        return "{}{},\n{}input_dims={}, output_dims={})".format(
            prefix,
            np.array2string(self.data, separator=", ", prefix=prefix),
            pad,
            self.input_dims(),
            self.output_dims(),
        )

    def __eq__(self, other):
        """Test if two Operators are equal."""
        if not super().__eq__(other):
            return False
        return np.allclose(self.data, other.data, rtol=self.rtol, atol=self.atol)

    @property
    def data(self):
        """Return data."""
        return self._data

    @property
    def settings(self):
        """Return operator settings."""
        return {
            "data": self._data,
            "input_dims": self.input_dims(),
            "output_dims": self.output_dims(),
        }

    @classmethod
    def from_label(cls, label):
        """Return a tensor product of single-qubit operators.

        Args:
            label (string): single-qubit operator string.

        Returns:
            Operator: The N-qubit operator.

        Raises:
            QiskitError: if the label contains invalid characters, or the
                         length of the label is larger than an explicitly
                         specified num_qubits.

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
            "I": IGate().to_matrix(),
            "X": XGate().to_matrix(),
            "Y": YGate().to_matrix(),
            "Z": ZGate().to_matrix(),
            "H": HGate().to_matrix(),
            "S": SGate().to_matrix(),
            "T": TGate().to_matrix(),
            "0": np.array([[1, 0], [0, 0]], dtype=complex),
            "1": np.array([[0, 0], [0, 1]], dtype=complex),
            "+": np.array([[0.5, 0.5], [0.5, 0.5]], dtype=complex),
            "-": np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=complex),
            "r": np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=complex),
            "l": np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=complex),
        }
        if re.match(r"^[IXYZHST01rl\-+]+$", label) is None:
            raise QiskitError("Label contains invalid characters.")
        # Initialize an identity matrix and apply each gate
        num_qubits = len(label)
        op = Operator(np.eye(2 ** num_qubits, dtype=complex))
        for qubit, char in enumerate(reversed(label)):
            if char != "I":
                op = op.compose(label_mats[char], qargs=[qubit])
        return op

    def is_unitary(self, atol=None, rtol=None):
        """Return True if operator is a unitary matrix."""
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        return is_unitary_matrix(self._data, rtol=rtol, atol=atol)

    def to_operator(self):
        """Convert operator to matrix operator class"""
        return self

    def to_instruction(self):
        """Convert to a UnitaryGate instruction."""
        # pylint: disable=cyclic-import
        from qiskit.extensions.unitary import UnitaryGate

        return UnitaryGate(self.data)

    def conjugate(self):
        # Make a shallow copy and update array
        ret = copy.copy(self)
        ret._data = np.conj(self._data)
        return ret

    def transpose(self):
        # Make a shallow copy and update array
        ret = copy.copy(self)
        ret._data = np.transpose(self._data)
        ret._op_shape = self._op_shape.transpose()
        return ret

    def compose(self, other, qargs=None, front=False):
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if not isinstance(other, Operator):
            other = Operator(other)

        # Validate dimensions are compatible and return the composed
        # operator dimensions
        new_shape = self._op_shape.compose(other._op_shape, qargs, front)
        input_dims = new_shape.dims_r()
        output_dims = new_shape.dims_l()

        # Full composition of operators
        if qargs is None:
            if front:
                # Composition self * other
                data = np.dot(self._data, other.data)
            else:
                # Composition other * self
                data = np.dot(other.data, self._data)
            ret = Operator(data, input_dims, output_dims)
            ret._op_shape = new_shape
            return ret

        # Compose with other on subsystem
        num_qargs_l, num_qargs_r = self._op_shape.num_qargs
        if front:
            num_indices = num_qargs_r
            shift = num_qargs_l
            right_mul = True
        else:
            num_indices = num_qargs_l
            shift = 0
            right_mul = False

        # Reshape current matrix
        # Note that we must reverse the subsystem dimension order as
        # qubit 0 corresponds to the right-most position in the tensor
        # product, which is the last tensor wire index.
        tensor = np.reshape(self.data, self._op_shape.tensor_shape)
        mat = np.reshape(other.data, other._op_shape.tensor_shape)
        indices = [num_indices - 1 - qubit for qubit in qargs]
        final_shape = [np.product(output_dims), np.product(input_dims)]
        data = np.reshape(
            Operator._einsum_matmul(tensor, mat, indices, shift, right_mul), final_shape
        )
        ret = Operator(data, input_dims, output_dims)
        ret._op_shape = new_shape
        return ret

    def power(self, n):
        """Return the matrix power of the operator.

        Args:
            n (float): the power to raise the matrix to.

        Returns:
            Operator: the resulting operator ``O ** n``.

        Raises:
            QiskitError: if the input and output dimensions of the operator
                         are not equal.
        """
        if self.input_dims() != self.output_dims():
            raise QiskitError("Can only power with input_dims = output_dims.")
        ret = copy.copy(self)
        ret._data = np.linalg.matrix_power(self.data, n)
        return ret

    def tensor(self, other):
        if not isinstance(other, Operator):
            other = Operator(other)
        return self._tensor(self, other)

    def expand(self, other):
        if not isinstance(other, Operator):
            other = Operator(other)
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        ret = copy.copy(a)
        ret._op_shape = a._op_shape.tensor(b._op_shape)
        ret._data = np.kron(a.data, b.data)
        return ret

    def _add(self, other, qargs=None):
        """Return the operator self + other.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (Operator): an operator object.
            qargs (None or list): optional subsystems to add on
                                  (Default: None)

        Returns:
            Operator: the operator self + other.

        Raises:
            QiskitError: if other is not an operator, or has incompatible
                         dimensions.
        """
        # pylint: disable=cyclic-import
        from qiskit.quantum_info.operators.scalar_op import ScalarOp

        if qargs is None:
            qargs = getattr(other, "qargs", None)

        if not isinstance(other, Operator):
            other = Operator(other)

        self._op_shape._validate_add(other._op_shape, qargs)
        other = ScalarOp._pad_with_identity(self, other, qargs)

        ret = copy.copy(self)
        ret._data = self.data + other.data
        return ret

    def _multiply(self, other):
        """Return the operator self * other.

        Args:
            other (complex): a complex number.

        Returns:
            Operator: the operator other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        ret = copy.copy(self)
        ret._data = other * self._data
        return ret

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
            atol = self.atol
        if rtol is None:
            rtol = self.rtol
        return matrix_equal(self.data, other.data, ignore_phase=True, rtol=rtol, atol=atol)

    def reverse_qargs(self):
        r"""Return an Operator with reversed subsystem ordering.

        For a tensor product operator this is equivalent to reversing
        the order of tensor product subsystems. For an operator
        :math:`A = A_{n-1} \otimes ... \otimes A_0`
        the returned operator will be
        :math:`A_0 \otimes ... \otimes A_{n-1}`.

        Returns:
            Operator: the operator with reversed subsystem order.
        """
        ret = copy.copy(self)
        axes = tuple(range(self._op_shape._num_qargs_l - 1, -1, -1))
        axes = axes + tuple(len(axes) + i for i in axes)
        ret._data = np.reshape(
            np.transpose(np.reshape(self.data, self._op_shape.tensor_shape), axes),
            self._op_shape.shape,
        )
        ret._op_shape = self._op_shape.reverse()
        return ret

    @classmethod
    def _einsum_matmul(cls, tensor, mat, indices, shift=0, right_mul=False):
        """Perform a contraction using Numpy.einsum

        Args:
            tensor (np.array): a vector or matrix reshaped to a rank-N tensor.
            mat (np.array): a matrix reshaped to a rank-2M tensor.
            indices (list): tensor indices to contract with mat.
            shift (int): shift for indices of tensor to contract [Default: 0].
            right_mul (bool): if True right multiply tensor by mat
                              (else left multiply) [Default: False].

        Returns:
            Numpy.ndarray: the matrix multiplied rank-N tensor.

        Raises:
            QiskitError: if mat is not an even rank tensor.
        """
        rank = tensor.ndim
        rank_mat = mat.ndim
        if rank_mat % 2 != 0:
            raise QiskitError("Contracted matrix must have an even number of indices.")
        # Get einsum indices for tensor
        indices_tensor = list(range(rank))
        for j, index in enumerate(indices):
            indices_tensor[index + shift] = rank + j
        # Get einsum indices for mat
        mat_contract = list(reversed(range(rank, rank + len(indices))))
        mat_free = [index + shift for index in reversed(indices)]
        if right_mul:
            indices_mat = mat_contract + mat_free
        else:
            indices_mat = mat_free + mat_contract
        return np.einsum(tensor, indices_tensor, mat, indices_mat)

    @classmethod
    def _init_instruction(cls, instruction):
        """Convert a QuantumCircuit or Instruction to an Operator."""
        # Initialize an identity operator of the correct size of the circuit
        if hasattr(instruction, "__array__"):
            return Operator(np.array(instruction, dtype=complex))

        dimension = 2 ** instruction.num_qubits
        op = Operator(np.eye(dimension))
        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        op._append_instruction(instruction)
        return op

    @classmethod
    def _instruction_to_matrix(cls, obj):
        """Return Operator for instruction if defined or None otherwise."""
        if not isinstance(obj, Instruction):
            raise QiskitError("Input is not an instruction.")
        mat = None
        if hasattr(obj, "to_matrix"):
            # If instruction is a gate first we see if it has a
            # `to_matrix` definition and if so use that.
            try:
                mat = obj.to_matrix()
            except QiskitError:
                pass
        return mat

    def _append_instruction(self, obj, qargs=None):
        """Update the current Operator by apply an instruction."""
        from qiskit.circuit.barrier import Barrier
        from .scalar_op import ScalarOp

        mat = self._instruction_to_matrix(obj)
        if mat is not None:
            # Perform the composition and inplace update the current state
            # of the operator
            op = self.compose(mat, qargs=qargs)
            self._data = op.data
        elif isinstance(obj, Barrier):
            return
        else:
            # If the instruction doesn't have a matrix defined we use its
            # circuit decomposition definition if it exists, otherwise we
            # cannot compose this gate and raise an error.
            if obj.definition is None:
                raise QiskitError(f"Cannot apply Instruction: {obj.name}")
            if not isinstance(obj.definition, QuantumCircuit):
                raise QiskitError(
                    'Instruction "{}" '
                    "definition is {} but expected QuantumCircuit.".format(
                        obj.name, type(obj.definition)
                    )
                )
            if obj.definition.global_phase:
                dimension = 2 ** obj.num_qubits
                op = self.compose(
                    ScalarOp(dimension, np.exp(1j * float(obj.definition.global_phase))),
                    qargs=qargs,
                )
                self._data = op.data
            flat_instr = obj.definition
            bit_indices = {
                bit: index
                for bits in [flat_instr.qubits, flat_instr.clbits]
                for index, bit in enumerate(bits)
            }

            for instr, qregs, cregs in flat_instr:
                if cregs:
                    raise QiskitError(
                        f"Cannot apply instruction with classical registers: {instr.name}"
                    )
                # Get the integer position of the flat register
                if qargs is None:
                    new_qargs = [bit_indices[tup] for tup in qregs]
                else:
                    new_qargs = [qargs[bit_indices[tup]] for tup in qregs]
                self._append_instruction(instr, qargs=new_qargs)


# Update docstrings for API docs
generate_apidocs(Operator)
