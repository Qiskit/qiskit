# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Clifford operator class.
"""
# pylint: disable=invalid-name, abstract-method

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from .stabilizer_table import StabilizerTable
from .clifford_circuits import append_gate, decompose_clifford


class Clifford(BaseOperator):
    """Clifford table operator class"""

    def __init__(self, data, validate=True):
        """Initialize an operator object."""

        # Initialize from another Clifford by sharing the underlying
        # StabilizerTable
        if isinstance(data, Clifford):
            self._table = data._table

        # Initialize from ScalarOp as N-qubit identity discarding any global phase
        elif isinstance(data, ScalarOp):
            if not data.is_unitary() or set(data._input_dims) != set([2]):
                raise QiskitError("Can only initalize from N-qubit identity ScalarOp.")
            self._table = StabilizerTable(
                np.eye(2 * len(data._input_dims), dtype=np.bool))

        # Initialize from a QuantumCircuit or Instruction object
        elif isinstance(data, (QuantumCircuit, Instruction)):
            self._table = Clifford.from_instruction(data)._table

        # Initialize StabilizerTable directly from the data
        else:
            self._table = StabilizerTable(data)

            # Validate table is a symplectic matrix
            if validate and not Clifford._is_symplectic(self._table.array):
                raise QiskitError(
                    'Invalid Clifford. Input StabilizerTable is not a valid'
                    ' symplectic matrix.')

        # Initialize BaseOperator
        dims = self._table.num_qubits * (2,)
        super().__init__(dims, dims)

    def __repr__(self):
        return 'Clifford({})'.format(repr(self.table))

    def __str__(self):
        return 'Clifford: Stabilizer = {}, Destabilizer = {}'.format(
            str(self.stabilizer.to_labels()),
            str(self.destabilizer.to_labels()))

    def __eq__(self, other):
        """Check if two Clifford tables are equal"""
        return super().__eq__(other) and self._table == other._table

    # ---------------------------------------------------------------------
    # Attributes
    # ---------------------------------------------------------------------
    def __getitem__(self, key):
        """Return a stabilizer Pauli row"""
        return self._table.__getitem__(key)

    def __setitem__(self, key, value):
        """Set a stabilizer Pauli row"""
        self._table.__setitem__(key, value)

    @property
    def table(self):
        """Return StabilizerTable"""
        return self._table

    @table.setter
    def table(self, value):
        """Set the stabilizer table"""
        # Note that is setup so it can't change the size of the Clifford
        # It can only replace the contents of the StabilizerTable with
        # another StabilizerTable of the same size.
        if not isinstance(value, StabilizerTable):
            value = StabilizerTable(value)
        self._table._array[:, :] = value._table._array
        self._table._phase[:] = value._table._phase

    @property
    def stabilizer(self):
        """Return the stabilizer block of the StabilizerTable."""
        return StabilizerTable(self._table[self.num_qubits:2*self.num_qubits])

    @stabilizer.setter
    def stabilizer(self, value):
        """Set the value of stabilizer block of the StabilizerTable"""
        inds = slice(self.num_qubits, 2*self.num_qubits)
        self._table.__setitem__(inds, value)

    @property
    def destabilizer(self):
        """Return the destabilizer block of the StabilizerTable."""
        return StabilizerTable(self._table[0:self.num_qubits])

    @destabilizer.setter
    def destabilizer(self, value):
        """Set the value of destabilizer block of the StabilizerTable"""
        inds = slice(0, self.num_qubits)
        self._table.__setitem__(inds, value)

    # ---------------------------------------------------------------------
    # Utility Operator methods
    # ---------------------------------------------------------------------

    def is_unitary(self, atol=None, rtol=None):
        """Return True if the Clifford table is valid."""
        # pylint: disable=unused-argument
        # A valid Clifford is always unitary, so this function is really
        # checking that the underlying Stabilizer table array is a valid
        # Clifford array.
        return Clifford._is_symplectic(self.table.array)

    # ---------------------------------------------------------------------
    # BaseOperator Abstract Methods
    # ---------------------------------------------------------------------

    def conjugate(self):
        """Return the conjugate of the Clifford."""
        return Clifford._conjugate_transpose(self, 'C')

    def adjoint(self):
        """Return the conjugate transpose of the Clifford"""
        return Clifford._conjugate_transpose(self, 'A')

    def transpose(self):
        """Return the transpose of the Clifford."""
        return Clifford._conjugate_transpose(self, 'T')

    def compose(self, other, qargs=None, front=False):
        """Return the composed operator.

        Args:
            other (Clifford): an operator object.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].
            front (bool): If True compose using right operator multiplication,
                          instead of left multiplication [default: False].

        Returns:
            Clifford: The operator self @ other.

        Raise:
            QiskitError: if operators have incompatible dimensions for
                         composition.

        Additional Information:
            Composition (``@``) is defined as `left` matrix multiplication for
            matrix operators. That is that ``A @ B`` is equal to ``B * A``.
            Setting ``front=True`` returns `right` matrix multiplication
            ``A * B`` and is equivalent to the :meth:`dot` method.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        if not isinstance(other, Clifford):
            other = Clifford(other)

        # Validate dimensions. Note we don't need to get updated input or
        # output dimensions from `_get_compose_dims` as the dimensions of the
        # Clifford object can't be changed by composition
        self._get_compose_dims(other, qargs, front)

        # Pad other with identities if composeing on subsystem
        other = self._pad_with_identity(other, qargs)
        return self._compose_clifford(other, front=front)

    def dot(self, other, qargs=None):
        """Return the right multiplied operator self * other.

        Args:
            other (Clifford): an operator object.
            qargs (list or None): a list of subsystem positions to apply
                                  other on. If None apply on all
                                  subsystems [default: None].

        Returns:
            Clifford: The operator self * other.

        Raises:
            QiskitError: if operators have incompatible dimensions for
                         composition.
        """
        return super().dot(other, qargs=qargs)

    def tensor(self, other):
        """Return the tensor product operator self ⊗ other.

        Args:
            other (Clifford): a operator subclass object.

        Returns:
            Clifford: the tensor product operator self ⊗ other.
        """
        return self._tensor_product(other, reverse=False)

    def expand(self, other):
        """Return the tensor product operator other ⊗ self.

        Args:
            other (Clifford): an operator object.

        Returns:
            Clifford: the tensor product operator other ⊗ self.
        """
        return self._tensor_product(other, reverse=True)

    # ---------------------------------------------------------------------
    # Representation conversions
    # ---------------------------------------------------------------------

    def to_dict(self):
        """Return dictionary represenation of Clifford object"""
        return {
            "stabilizer": self.stabilizer.to_labels(),
            "destabilizer": self.destabilizer.to_labels()
        }

    @staticmethod
    def from_dict(obj):
        """Load a Clifford from a dictionary"""
        destabilizer = StabilizerTable.from_labels(obj.get('destabilizer'))
        stabilizer = StabilizerTable.from_labels(obj.get('stabilizer'))
        return Clifford(destabilizer + stabilizer)

    def to_matrix(self):
        """Convert operator to Numpy matrix."""
        return self.to_operator().data

    def to_operator(self):
        """Convert to an Operator object."""
        return Operator(self.to_gate())

    def to_circuit(self):
        """Return a QuantumCircuit implementing the Clifford."""
        return decompose_clifford(self)

    def to_gate(self):
        """Return a Gate instruction implementing the Clifford."""
        return self.to_circuit().to_gate()

    @staticmethod
    def from_instruction(instruction):
        """Initialize from a QuantumCircuit or Instruction.

        Args:
            instruction (QuantumCircuit or Instruction): instruction to
                                                         initialize.

        Returns:
            Clifford: the Clifford object for the instruction.

        Raises:
            QiskitError: if the input instruction is non-Clifford or contains
                         classical register instruction.
        """
        if not isinstance(instruction, (QuantumCircuit, Instruction)):
            raise QiskitError("Input must be a QuantumCircuit or Instruction")

        # Convert circuit to an instruction
        if isinstance(instruction, QuantumCircuit):
            instruction = instruction.to_instruction()

        # Initialize an identity Clifford
        clifford = Clifford(np.eye(2 * instruction.num_qubits), validate=False)
        append_gate(clifford, instruction)
        return clifford

    # ---------------------------------------------------------------------
    # Internal helper functions
    # ---------------------------------------------------------------------

    @staticmethod
    def _is_symplectic(mat):
        """Return True if input is symplectic matrix."""
        # Condition is
        # table.T * [[0, 1], [1, 0]] * table = [[0, 1], [1, 0]]
        # where we are block matrix multiplying using symplectic product

        dim = len(mat) // 2
        if mat.shape != (2 * dim, 2 * dim):
            return False

        one = np.eye(dim, dtype=np.int)
        zero = np.zeros((dim, dim), dtype=np.int)
        seye = np.block([[zero, one], [one, zero]])
        arr = mat.astype(np.int)
        return np.array_equal(np.mod(arr.T.dot(seye).dot(arr), 2), seye)

    @staticmethod
    def _conjugate_transpose(clifford, method):
        """Return the adjoint, conjugate, or transpose of the Clifford.

        Args:
            method (str): what function to apply 'A', 'C', or 'T'.

        Returns:
            Clifford: the modified clifford.
        """
        ret = clifford.copy()
        if method in ['A', 'T']:
            # Apply inverse
            # Update table
            tmp = ret.destabilizer.X.copy()
            ret.destabilizer.X = ret.stabilizer.Z.T
            ret.destabilizer.Z = ret.destabilizer.Z.T
            ret.stabilizer.X = ret.stabilizer.X.T
            ret.stabilizer.Z = tmp.T
            # Update phase
            ret.table.phase ^= clifford.dot(ret).table.phase
        if method in ['C', 'T']:
            # Apply conjugate
            ret.table.phase ^= np.mod(np.sum(
                ret.table.X & ret.table.Z, axis=1), 2).astype(np.bool)
        return ret

    def _tensor_product(self, other, reverse=False):
        """Return the tensor product operator.

        Args:
            other (Clifford): another Clifford operator.
            reverse (bool): If False return self ⊗ other, if True return
                            if True return (other ⊗ self) [Default: False
        Returns:
            Clifford: the tensor product operator.

        Raises:
            QiskitError: if other cannot be converted into an Clifford.
        """
        if not isinstance(other, Clifford):
            other = Clifford(other)

        if reverse:
            cliff0 = self
            cliff1 = other
        else:
            cliff0 = other
            cliff1 = self

        # Pad stabilizers and destabilizers
        destab = (cliff0.destabilizer.expand(cliff1.num_qubits * 'I') +
                  cliff1.destabilizer.tensor(cliff0.num_qubits * 'I'))
        stab = (cliff0.stabilizer.expand(cliff1.num_qubits * 'I') +
                cliff1.stabilizer.tensor(cliff0.num_qubits * 'I'))

        # Add the padded table
        return Clifford(destab + stab, validate=False)

    def _pad_with_identity(self, clifford, qargs):
        """Pad Clifford with identities on other subsystems."""
        if qargs is None:
            return clifford

        padded = Clifford(StabilizerTable(
            np.eye(2 * self.num_qubits, dtype=np.bool)), validate=False)

        inds = list(qargs) + [self.num_qubits + i for i in qargs]

        # Pad Pauli array
        pauli = clifford.table.array
        for i, pos in enumerate(qargs):
            padded.table.array[inds, pos] = pauli[:, i]
            padded.table.array[inds, self.num_qubits + pos] = pauli[:, clifford.num_qubits + i]

        # Pad phase
        padded.table.phase[inds] = clifford.table.phase

        return padded

    def _compose_clifford(self, other, front=False):
        """Return the composition channel assume other is Clifford of same size as self."""
        if front:
            table1 = self.table
            table2 = other.table
        else:
            table1 = other.table
            table2 = self.table

        num_qubits = self.num_qubits

        array1 = table1.array.astype(int)
        phase1 = table1.phase.astype(int)

        array2 = table2.array.astype(int)
        phase2 = table2.phase.astype(int)

        # Update Pauli table
        pauli = StabilizerTable(array2.dot(array1) % 2)

        # Add phases
        phase = np.mod(array2.dot(phase1) + phase2, 2)

        # Correcting for phase due to Pauli multiplicatio
        ifacts = np.zeros(2 * num_qubits, dtype=np.int)

        for k in range(2 * num_qubits):

            row2 = array2[k]
            x2 = table2.X[k]
            z2 = table2.Z[k]

            # Adding a factor of i for each Y in the image of an operator under the
            # first operation, since Y=iXZ

            ifacts[k] += np.sum(x2 & z2)

            # Adding factors of i due to qubit-wise Pauli multiplication

            for j in range(num_qubits):
                x = 0
                z = 0
                for i in range(2 * num_qubits):
                    if row2[i]:
                        x1 = array1[i, j]
                        z1 = array1[i, j + num_qubits]
                        if (x | z) & (x1 | z1):
                            val = np.mod(np.abs(3 * z1 - x1) - np.abs(3 * z - x) - 1, 3)
                            if val == 0:
                                ifacts[k] += 1
                            elif val == 1:
                                ifacts[k] -= 1
                        x = np.mod(x + x1, 2)
                        z = np.mod(z + z1, 2)

        p = np.mod(ifacts, 4) // 2

        phase = np.mod(phase + p, 2)

        return Clifford(StabilizerTable(pauli, phase), validate=False)
