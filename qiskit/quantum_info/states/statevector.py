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
Statevector quantum state class.
"""

import re
from numbers import Number

import numpy as np

from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.instruction import Instruction
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.quantum_info.states.counts import state_to_counts
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.predicates import matrix_equal


class Statevector(QuantumState):
    """Statevector class"""

    def __init__(self, data, dims=None):
        """Initialize a state object."""
        if isinstance(data, Statevector):
            # Shallow copy constructor
            vec = data.data
            if dims is None:
                dims = data.dims()
        elif isinstance(data, Operator):
            # We allow conversion of column-vector operators to Statevectors
            input_dim, output_dim = data.dim
            if input_dim != 1:
                raise QiskitError("Input Operator is not a column-vector.")
            vec = np.reshape(data.data, output_dim)
        elif isinstance(data, (list, np.ndarray)):
            # Finally we check if the input is a raw vector in either a
            # python list or numpy array format.
            vec = np.array(data, dtype=complex)
        else:
            raise QiskitError("Invalid input data format for Statevector")
        # Check that the input is a numpy vector or column-vector numpy
        # matrix. If it is a column-vector matrix reshape to a vector.
        if vec.ndim not in [1, 2] or (vec.ndim == 2 and vec.shape[1] != 1):
            raise QiskitError("Invalid input: not a vector or column-vector.")
        if vec.ndim == 2 and vec.shape[1] == 1:
            vec = np.reshape(vec, vec.shape[0])
        dim = vec.shape[0]
        subsystem_dims = self._automatic_dims(dims, dim)
        super().__init__('Statevector', vec, subsystem_dims)

    def is_valid(self, atol=None, rtol=None):
        """Return True if a Statevector has norm 1."""
        if atol is None:
            atol = self._atol
        if rtol is None:
            rtol = self._rtol
        norm = np.linalg.norm(self.data)
        return np.allclose(norm, 1, rtol=rtol, atol=atol)

    def to_operator(self):
        """Convert state to a rank-1 projector operator"""
        mat = np.outer(self.data, np.conj(self.data))
        return Operator(mat, input_dims=self.dims(), output_dims=self.dims())

    def conjugate(self):
        """Return the conjugate of the operator."""
        return Statevector(np.conj(self.data), dims=self.dims())

    def trace(self):
        """Return the trace of the quantum state as a density matrix."""
        return np.sum(np.abs(self.data) ** 2)

    def purity(self):
        """Return the purity of the quantum state."""
        # For a valid statevector the purity is always 1, however if we simply
        # have an arbitrary vector (not correctly normalized) then the
        # purity is equivalent to the trace squared:
        # P(|psi>) = Tr[|psi><psi|psi><psi|] = |<psi|psi>|^2
        return self.trace() ** 2

    def tensor(self, other):
        """Return the tensor product state self ⊗ other.

        Args:
            other (Statevector): a quantum state object.

        Returns:
            Statevector: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        if not isinstance(other, Statevector):
            other = Statevector(other)
        dims = other.dims() + self.dims()
        data = np.kron(self._data, other._data)
        return Statevector(data, dims)

    def expand(self, other):
        """Return the tensor product state other ⊗ self.

        Args:
            other (Statevector): a quantum state object.

        Returns:
            Statevector: the tensor product state other ⊗ self.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        if not isinstance(other, Statevector):
            other = Statevector(other)
        dims = self.dims() + other.dims()
        data = np.kron(other._data, self._data)
        return Statevector(data, dims)

    def add(self, other):
        """Return the linear combination self + other.

        Args:
            other (Statevector): a quantum state object.

        Returns:
            LinearOperator: the linear combination self + other.

        Raises:
            QiskitError: if other is not a quantum state, or has
            incompatible dimensions.
        """
        if not isinstance(other, Statevector):
            other = Statevector(other)
        if self.dim != other.dim:
            raise QiskitError("other Statevector has different dimensions.")
        return Statevector(self.data + other.data, self.dims())

    def subtract(self, other):
        """Return the linear operator self - other.

        Args:
            other (Statevector): a quantum state object.

        Returns:
            LinearOperator: the linear combination self - other.

        Raises:
            QiskitError: if other is not a quantum state, or has
            incompatible dimensions.
        """
        if not isinstance(other, Statevector):
            other = Statevector(other)
        if self.dim != other.dim:
            raise QiskitError("other Statevector has different dimensions.")
        return Statevector(self.data - other.data, self.dims())

    def multiply(self, other):
        """Return the linear operator self * other.

        Args:
            other (complex): a complex number.

        Returns:
            Operator: the linear combination other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        return Statevector(other * self.data, self.dims())

    def evolve(self, other, qargs=None):
        """Evolve a quantum state by the operator.

        Args:
            other (Operator): The operator to evolve by.
            qargs (list): a list of Statevector subsystem positions to apply
                           the operator on.

        Returns:
            Statevector: the output quantum state.

        Raises:
            QiskitError: if the operator dimension does not match the
            specified Statevector subsystem dimensions.
        """
        # Evolution by a circuit or instruction
        if isinstance(other, (QuantumCircuit, Instruction)):
            return self._evolve_instruction(other, qargs=qargs)
        # Evolution by an Operator
        if not isinstance(other, Operator):
            other = Operator(other)
        if qargs is None:
            # Evolution on full statevector
            if self._dim != other._input_dim:
                raise QiskitError(
                    "Operator input dimension is not equal to statevector dimension."
                )
            return Statevector(np.dot(other.data, self.data), dims=other.output_dims())
        # Otherwise we are applying an operator only to subsystems
        # Check dimensions of subsystems match the operator
        if self.dims(qargs) != other.input_dims():
            raise QiskitError(
                "Operator input dimensions are not equal to statevector subsystem dimensions."
            )
        # Reshape statevector and operator
        tensor = np.reshape(self.data, self._shape)
        mat = np.reshape(other.data, other._shape)
        # Construct list of tensor indices of statevector to be contracted
        num_indices = len(self.dims())
        indices = [num_indices - 1 - qubit for qubit in qargs]
        tensor = Operator._einsum_matmul(tensor, mat, indices)
        new_dims = list(self.dims())
        for i, qubit in enumerate(qargs):
            new_dims[qubit] = other._output_dims[i]
        # Replace evolved dimensions
        return Statevector(np.reshape(tensor, np.product(new_dims)), dims=new_dims)

    def equiv(self, other, rtol=None, atol=None):
        """Return True if statevectors are equivalent up to global phase.

        Args:
            other (Statevector): a statevector object.
            rtol (float): relative tolerance value for comparison.
            atol (float): absolute tolerance value for comparison.

        Returns:
            bool: True if statevectors are equivalent up to global phase.
        """
        if not isinstance(other, Statevector):
            try:
                other = Statevector(other)
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

    def to_counts(self):
        """Returns the statevector as a counts dict
        of probabilities.

        Returns:
            dict: Counts of probabilities.
        """
        return state_to_counts(self.data.ravel(), self._atol)

    @classmethod
    def from_label(cls, label):
        """Return a tensor product of Pauli X,Y,Z eigenstates.

        Args:
            label (string): a eigenstate string ket label 0,1,+,-,r,l.

        Returns:
            Statevector: The N-qubit basis state density matrix.

        Raises:
            QiskitError: if the label contains invalid characters, or the length
            of the label is larger than an explicitly specified num_qubits.

        Additional Information:
            The labels correspond to the single-qubit states:
            '0': [1, 0]
            '1': [0, 1]
            '+': [1 / sqrt(2), 1 / sqrt(2)]
            '-': [1 / sqrt(2), -1 / sqrt(2)]
            'r': [1 / sqrt(2), 1j / sqrt(2)]
            'l': [1 / sqrt(2), -1j / sqrt(2)]
        """
        # Check label is valid
        if re.match(r'^[01rl\-+]+$', label) is None:
            raise QiskitError('Label contains invalid characters.')
        # We can prepare Z-eigenstates by converting the computational
        # basis bit-string to an integer and preparing that unit vector
        # However, for X-basis states, we will prepare a Z-eigenstate first
        # then apply Hadamard gates to rotate 0 and 1s to + and -.
        z_label = label
        xy_states = False
        if re.match('^[01]+$', label) is None:
            # We have X or Y eigenstates so replace +,r with 0 and
            # -,l with 1 and prepare the corresponding Z state
            xy_states = True
            z_label = z_label.replace('+', '0')
            z_label = z_label.replace('r', '0')
            z_label = z_label.replace('-', '1')
            z_label = z_label.replace('l', '1')
        # Initialize Z eigenstate vector
        num_qubits = len(label)
        data = np.zeros(1 << num_qubits, dtype=complex)
        pos = int(z_label, 2)
        data[pos] = 1
        state = Statevector(data)
        if xy_states:
            # Apply hadamards to all qubits in X eigenstates
            x_mat = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
            # Apply S.H to qubits in Y eigenstates
            y_mat = np.dot(np.diag([1, 1j]), x_mat)
            for qubit, char in enumerate(reversed(label)):
                if char in ['+', '-']:
                    state = state.evolve(x_mat, qargs=[qubit])
                elif char in ['r', 'l']:
                    state = state.evolve(y_mat, qargs=[qubit])
        return state

    @classmethod
    def from_instruction(cls, instruction):
        """Return the output statevector of an instruction.

        The statevector is initialized in the state |0,...,0> of the same
        number of qubits as the input instruction or circuit, evolved
        by the input instruction, and the output statevector returned.

        Args:
            instruction (Instruction or QuantumCircuit): instruction or circuit

        Returns:
            Statevector: The final statevector.

        Raises:
            QiskitError: if the instruction contains invalid instructions for
            the statevector simulation.
        """
        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()
        # Initialize an the statevector in the all |0> state
        init = np.zeros(2 ** instruction.num_qubits, dtype=complex)
        init[0] = 1
        vec = Statevector(init, dims=instruction.num_qubits * [2])
        vec._append_instruction(instruction)
        return vec

    @property
    def _shape(self):
        """Return the tensor shape of the matrix operator"""
        return tuple(reversed(self.dims()))

    def _append_instruction(self, obj, qargs=None):
        """Update the current Statevector by applying an instruction."""
        mat = Operator._instruction_to_matrix(obj)
        if mat is not None:
            # Perform the composition and inplace update the current state
            # of the operator
            self._data = self.evolve(mat, qargs=qargs).data
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

    def _evolve_instruction(self, obj, qargs=None):
        """Return a new statevector by applying an instruction."""
        if isinstance(obj, QuantumCircuit):
            obj = obj.to_instruction()
        vec = Statevector(self.data, dims=self.dims())
        vec._append_instruction(obj, qargs=qargs)
        return vec
