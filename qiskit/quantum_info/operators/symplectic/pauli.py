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
Symplectic Pauli Operator Class
"""
# pylint: disable=invalid-name, abstract-method

import re
from warnings import warn
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.circuit.barrier import Barrier


class Pauli(BasePauli):
    r"""N-qubit Pauli group operator.

    **Symplectic Representation**

    The symplectic representation of a single-qubit Pauli matrix
    is a pair of boolean values :math:`[x, z]` and phase `q` such that the
    Pauli matrix is given by :math:`P = (-i)^{q + x.z} \sigma_z^z.\sigma_x^x`.
    Where :math:`z, x \in \mathbb{Z}_2` and :math:`q \in \mathbb{Z}_4`.

    **Data Access**

    The individual qubit Paulis can be accessed using the list access ``[]`` operator.
    The underlying Numpy array can be directly accessed using the :attr:`array` property,
    and the sub-arrays for the `X` or `Z` part can be accessed using the
    :attr:`X` and :attr:`Z` properties respectively.
    """
    # pylint: disable = missing-param-doc, missing-type-doc
    def __init__(self, z=None, x=None, phase=None, *, label=None):
        """Initialize the Pauli.

        When using the symplectic array input data both z and x arguments must
        be provided, however the first (z) argument can be used alone for string
        label, Pauli operator, or ScalarOp input data.

        Args:
            z (array or str or ScalarOp or Pauli): input data or symplectic z vector.
            x (array): Optional, symplectic x vector.
            phase (int or None): Optional, phase exponent from Z_4.
            label (str): DEPRECATED, string label.

        Raises:
            QiskitError: if input array is invalid shape.
        """
        if isinstance(z, BasePauli):
            # BasePauli initialization
            base_z, base_x, base_phase = z._z, z._x, z._phase
            if z._num_paulis != 1:
                raise QiskitError("Input is not a single Pauli")
        elif x is not None:
            base_z, base_x, base_phase = self._from_array(z, x, phase)
        elif isinstance(z, str):
            base_z, base_x, base_phase = self._from_label(z)
        elif label is not None:
            # Check for deprecated initialization from legacy Pauli
            warn('Initializing Pauli from label kwarg is deprecated '
                 'and will be removed no earlier than 3 months after the release date. '
                 'Use `Pauli(str)` instead.', DeprecationWarning)
            base_z, base_x, base_phase = self._from_label(label)
        elif isinstance(z, ScalarOp):
            base_z, base_x, base_phase = self._from_scalar_op(z)
        elif isinstance(z, (QuantumCircuit, Instruction)):
            base_z, base_x, base_phase = self._from_circuit(z)
        else:
            raise QiskitError("Invalid input data for Pauli.")

        # Add phase
        if phase is not None:
            base_phase = np.mod(int(phase) + np.sum(
                np.logical_and(base_z, base_x), axis=1), 4)

        # Set size properties
        super().__init__(base_z, base_x, base_phase)

    def __repr__(self):
        """Display representation."""
        if len(self) > 100:
            return "Pauli('{}...{}'[{}])".format(
                self[:3].__str__(), self[-3:].__str__(), self.num_qubits)
        return "Pauli('{}')".format(self.__str__())

    def __eq__(self, other):
        """Test if two Paulis are equal."""
        return (isinstance(other, Pauli)
                and self._phase[0] == other._phase[0]
                and np.all(self._z == other._z)
                and np.all(self._x == other._x))

    def equiv(self, other):
        """Return True if Pauli's are equivalent up to global phase.

        Args:
            other (Pauli): an operator object.

        Returns:
            bool: True if the Pauli's are equivalent up to global phase.
        """
        if not isinstance(other, Pauli):
            try:
                other = Pauli(other)
            except QiskitError:
                return False
        return np.all(self._z == other._z) and np.all(self._x == other._x)

    # ---------------------------------------------------------------------
    # Direct array access
    # ---------------------------------------------------------------------
    @property
    def phase(self):
        """Return the phase exponent relative to the unsigned Pauli group element."""
        return np.mod(self._phase - self._count_y(), 4)[0]

    @phase.setter
    def phase(self, value):
        """Set the phase exponent."""
        # Convert group phase convetion to internal ZX-phase convention
        self._phase[:] = np.mod(value + self._count_y(), 4)

    @property
    def x(self):
        """The x vector for the symplectic representation."""
        return self._x[0]

    @x.setter
    def x(self, val):
        self._x[0, :] = val

    @property
    def z(self):
        """The z vector for the symplectic representation."""
        return self._z[0]

    @z.setter
    def z(self, val):
        self._z[0, :] = val

    # ---------------------------------------------------------------------
    # Pauli Array methods
    # ---------------------------------------------------------------------

    def __len__(self):
        """Return the number of qubits in the Pauli."""
        return self.num_qubits

    def __getitem__(self, qubits):
        """Return the unsigned Pauli group Pauli for subset of qubits."""
        # Set group phase to 0 so returned Pauli is always +1 coeff
        if isinstance(qubits, (int, np.int)):
            qubits = [qubits]
        return Pauli(self.z[qubits], self.x[qubits])

    def __setitem__(self, qubits, value):
        """Update the Pauli for a subset of qubits."""
        if not isinstance(value, Pauli):
            value = Pauli(value)
        self._z[0, qubits] = value.z
        self._x[0, qubits] = value.x
        # Add extra phase from new Pauli to current
        self._phase += value._phase

    def delete(self, qubits):
        """Return a Pauli with qubits deleted.

        Args:
            qubits (int or list): qubits to delete from Pauli.

        Returns:
            Pauli: the resulting Pauli with the specified qubits removed.

        Raises:
            QiskitError: if ind is out of bounds for the array size or
                         number of qubits.
        """
        if isinstance(qubits, (int, np.int)):
            qubits = [qubits]
        if max(qubits) > self.num_qubits - 1:
            raise QiskitError(
                "Qubit index is larger than the number of qubits "
                "({}>{}).".format(max(qubits), self.num_qubits - 1))
        if len(qubits) == self.num_qubits:
            raise QiskitError("Cannot delete all qubits of Pauli")
        z = np.delete(self._z, qubits, axis=1)
        x = np.delete(self._x, qubits, axis=1)
        return Pauli(BasePauli(z, x, phase=self._phase))

    def insert(self, qubits, value):
        """Insert a Pauli at specific qubit value.

        Args:
            qubits (int or list): qubits index to insert at.
            value (Pauli): value to insert.

        Returns:
            Pauli: the resulting Pauli with the entries inserted.

        Raises:
            QiskitError: if the insertion qubits are invalid.
        """
        if not isinstance(value, Pauli):
            value = Pauli(value)

        # Initialize empty operator
        ret_qubits = self.num_qubits + value.num_qubits
        ret = Pauli(np.zeros(ret_qubits, dtype=np.bool),
                    np.zeros(ret_qubits, dtype=np.bool))
        if isinstance(qubits, (int, np.int)):
            if value.num_qubits == 1:
                qubits = [qubits]
            else:
                qubits = list(range(qubits, qubits + value.num_qubits))
        if len(qubits) != value.num_qubits:
            raise QiskitError(
                "Number of indices does not match number of qubits for "
                "the inserted Pauli ({}!={})".format(len(qubits), value.num_qubits))
        if max(qubits) > ret.num_qubits - 1:
            raise QiskitError(
                "Index is too larger for combined Pauli number of qubits "
                "({}>{}).".format(max(qubits), ret.num_qubits - 1))
        # Qubit positions for original op
        self_qubits = [i for i in range(ret.num_qubits) if i not in qubits]
        ret[self_qubits] = self
        ret[qubits] = value
        return ret

    # ---------------------------------------------------------------------
    # Representation conversions
    # ---------------------------------------------------------------------

    def __str__(self):
        """String representation."""
        return self._to_label(self.z, self.x, self._phase[0])

    def __hash__(self):
        """Make hashable based on string representation."""
        return hash(self.__str__())

    def to_matrix(self, sparse=False):
        r"""Convert to a Numpy array.

        Args:
            sparse (bool): if True return sparse CSR matrices, otherwise
                           return dense Numpy arrays (default: False).

        Returns:
            array: The Pauli matrix.
        """
        return self._to_matrix(self.z, self.x, self._phase[0], sparse=sparse)

    def to_operator(self):
        """Convert to a matrix Operator object"""
        from qiskit.quantum_info.operators.operator import Operator
        return Operator(self.to_matrix())

    def to_instruction(self):
        """Convert to Pauli circuit instruction."""
        from math import pi
        pauli, phase = self._to_label(self.z, self.x, self._phase[0],
                                      full_group=False,
                                      return_phase=True)
        gate = PauliGate(pauli)
        if not phase:
            return gate
        # Add global phase
        circuit = QuantumCircuit(self.num_qubits, name=str(self))
        circuit.global_phase = -phase * pi / 2
        circuit.append(gate, range(self.num_qubits))
        return circuit.to_instruction()

    # ---------------------------------------------------------------------
    # BaseOperator methods
    # ---------------------------------------------------------------------

    def tensor(self, other):
        """Return the tensor product Pauli self ⊗ other.

        Args:
            other (Pauli): another Pauli.

        Returns:
            Pauli: the tensor product Pauli.
        """
        if not isinstance(other, Pauli):
            other = Pauli(other)
        return Pauli(super().tensor(other))

    def expand(self, other):
        """Return the tensor product Pauli other ⊗ self.

        Args:
            other (Pauli): another Pauli.

        Returns:
            Pauli: the tensor product Pauli.
        """
        if not isinstance(other, Pauli):
            other = Pauli(other)
        return Pauli(super().expand(other))

    def compose(self, other, qargs=None, front=False, inplace=False):
        """Return the composed Pauli self∘other.

        Args:
            other (Pauli): another Pauli.
            qargs (None or list): qubits to apply dot product on (default: None).
            front (bool): If True use `dot` composition method (default: False).
            inplace (bool): If True update in-place (default: False).

        Returns:
            Pauli: the output Pauli.

        Raises:
            QiskitError: if other cannot be converted to a Pauli.
        """
        # pylint: disable=unused-argument
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if not isinstance(other, Pauli):
            other = Pauli(other)
        return Pauli(super().compose(
            other, qargs=qargs, front=front, inplace=inplace))

    def dot(self, other, qargs=None, inplace=False):
        """Return the dot product Pauli self∘other.

        Args:
            other (Pauli): another Pauli.
            qargs (None or list): qubits to apply dot product on (default: None).
            inplace (bool): If True update in-place (default: False).

        Returns:
            Pauli: the dot outer product table.

        Raises:
            QiskitError: if other cannot be converted to a Pauli.
        """
        return Pauli(super().dot(other, qargs=qargs, inplace=inplace))

    def _multiply(self, other):
        """Multiply Pauli by a phase.

        Args:
            other (complex): a complex number in [1, -1j, -1, 1j]

        Returns:
            Pauli: the Pauli other * self.

        Raises:
            QiskitError: if the phase is not in the set [1, -1j, -1, 1j].
        """
        return Pauli(super()._multiply(other))

    def conjugate(self):
        """Return the conjugated Pauli."""
        return Pauli(super().conjugate())

    def transpose(self):
        """Return the transposed Pauli."""
        return Pauli(super().transpose())

    def adjoint(self):
        """Return the adjoint Pauli."""
        return Pauli(super().adjoint())

    def inverse(self):
        """Return the inverse Pauli."""
        return Pauli(super().adjoint())

    # ---------------------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------------------

    def commutes(self, other, qargs=None):
        """Return True if the Pauli commutes with other.

        Args:
            other (Pauli or PauliList): another Pauli operator.
            qargs (list): qubits to apply dot product on (default: None).

        Returns:
            bool: True if Pauli's commute, False if they anti-commute.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if not isinstance(other, BasePauli):
            other = Pauli(other)
        ret = super().commutes(other, qargs=qargs)
        if len(ret) == 1:
            return ret[0]
        return ret

    def anticommutes(self, other, qargs=None):
        """Return True if other Pauli anticommutes with self.

        Args:
            other (Pauli): another Pauli operator.
            qargs (list): qubits to apply dot product on (default: None).

        Returns:
            bool: True if Pauli's anticommute, False if they commute.
        """
        return np.logical_not(self.commutes(other, qargs=qargs))

    def evolve(self, other, qargs=None):
        r"""Evolve the Pauli by a Clifford.

        This returns the Pauli :math:`P^\prime = C.P.C^\dagger`.

        Args:
            other (Pauli or Clifford or QuantumCircuit): The Clifford operator to evolve by.
            qargs (list): a list of qubits to apply the Clifford to.

        Returns:
            Pauli: the Pauli :math:`C.P.C^\dagger`.

        Raises:
            QiskitError: if the Clifford number of qubits and qargs don't match.
        """
        # pylint: disable=cyclic-import
        from qiskit.quantum_info.operators.symplectic.clifford import Clifford

        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        # Convert quantum circuits to Clifford
        if isinstance(other, Clifford):
            other = other.to_circuit()

        if not isinstance(other, (Pauli, Instruction, QuantumCircuit)):
            # Convert to a Pauli
            other = Pauli(other)

        return Pauli(super().evolve(other, qargs=qargs))

    # ---------------------------------------------------------------------
    # Initialization helper functions
    # ---------------------------------------------------------------------

    @staticmethod
    def _from_label(label):
        """Return the symplectic representation of Pauli string.

        Args:
            label (str): the Pauli string label.

        Returns:
            BasePauli: the BasePauli corresponding to the label.

        Raises:
            QiskitError: if Pauli string is not valid.
        """
        # Split string into coefficient and Pauli
        span = re.search(r'[IXYZ]+', label).span()
        pauli, coeff = _split_pauli_label(label)
        coeff = label[:span[0]]

        # Convert coefficient to phase
        phase = 0 if not coeff else _phase_from_label(coeff)
        if phase is None:
            raise QiskitError('Pauli string is not valid.')

        # Convert to Symplectic representation
        num_qubits = len(pauli)
        base_z = np.zeros((1, num_qubits), dtype=np.bool)
        base_x = np.zeros((1, num_qubits), dtype=np.bool)
        base_phase = np.array([phase], dtype=np.int)
        for i, char in enumerate(pauli):
            if char == 'X':
                base_x[0, num_qubits - 1 - i] = True
            elif char == 'Z':
                base_z[0, num_qubits - 1 - i] = True
            elif char == 'Y':
                base_x[0, num_qubits - 1 - i] = True
                base_z[0, num_qubits - 1 - i] = True
                base_phase += 1
        return base_z, base_x, base_phase % 4

    @staticmethod
    def _from_array(z, x, phase):
        """Convert array data to BasePauli data."""
        if isinstance(z, np.ndarray) and z.dtype == np.bool:
            base_z = z
        else:
            base_z = np.asarray(z, dtype=np.bool)
        if base_z.ndim == 1:
            base_z = base_z.reshape((1, base_z.size))
        elif base_z.ndim != 2 or base_z.shape[0] != 1:
            raise QiskitError("Invalid Pauli z vector shape.")

        if isinstance(x, np.ndarray) and x.dtype == np.bool:
            base_x = x
        else:
            base_x = np.asarray(x, dtype=np.bool)
        if base_x.ndim == 1:
            base_x = base_x.reshape((1, base_x.size))
        elif base_x.ndim != 2 or base_x.shape[0] != 1:
            raise QiskitError("Invalid Pauli x vector shape.")

        if base_z.shape != base_x.shape:
            raise QiskitError("z and x vectors are different size.")

        base_phase = None
        if phase is None:
            # Convert group phase convention to internal ZX-phase convertion.
            # If phase is not None it will be converted later.
            base_phase = np.mod(
                np.sum(np.logical_and(base_x, base_z), axis=1, dtype=np.int), 4)
        return base_z, base_x, base_phase

    @classmethod
    def _from_scalar_op(cls, op):
        """Convert a ScalarOp to BasePauli data."""
        if op.num_qubits is None:
            raise QiskitError('{} is not an N-qubit identity'.format(op))
        base_z = np.zeros((1, op.num_qubits), dtype=np.bool)
        base_x = np.zeros((1, op.num_qubits), dtype=np.bool)
        base_phase = np.mod(cls._phase_from_complex(op.coeff) + np.sum(
            np.logical_and(base_z, base_x), axis=1), 4)
        return base_z, base_x, base_phase

    @classmethod
    def _from_pauli_instruction(cls, instr):
        """Convert a Pauli instruction to BasePauli data."""
        if isinstance(instr, PauliGate):
            return cls._from_label(instr.params[0])
        if isinstance(instr, IGate):
            return np.array([[False]]), np.array([[False]]), np.array([0])
        if isinstance(instr, XGate):
            return np.array([[False]]), np.array([[True]]), np.array([0])
        if isinstance(instr, YGate):
            return np.array([[True]]), np.array([[True]]), np.array([1])
        if isinstance(instr, ZGate):
            return np.array([[True]]), np.array([[False]]), np.array([0])
        raise QiskitError("Invalid Pauli instruction.")

    @classmethod
    def _from_circuit(cls, instr):
        """Convert a Pauli circuit to BasePauli data."""
        # Try and convert single instruction
        if isinstance(instr, (PauliGate, IGate, XGate, YGate, ZGate)):
            return cls._from_pauli_instruction(instr)

        if isinstance(instr, Instruction):
            # Convert other instructions to circuit definition
            if instr.definition is None:
                raise QiskitError('Cannot apply Instruction: {}'.format(
                    instr.name))
            # Convert to circuit
            instr = instr.definition

        # Initialize identity Pauli
        ret = Pauli(BasePauli(
            np.zeros((1, instr.num_qubits), dtype=np.bool),
            np.zeros((1, instr.num_qubits), dtype=np.bool),
            np.zeros(1, dtype=np.int)))

        # Add circuit global phase if specified
        if instr.global_phase:
            ret.phase = cls._phase_from_complex(
                np.exp(1j * float(instr.global_phase)))

        # Recursively apply instructions
        for dinstr, qregs, cregs in instr.data:
            if cregs:
                raise QiskitError(
                    'Cannot apply instruction with classical registers: {}'.
                    format(dinstr.name))
            if not isinstance(dinstr, Barrier):
                next_instr = BasePauli(*cls._from_circuit(dinstr))
                if next_instr is not None:
                    qargs = [tup.index for tup in qregs]
                    ret = ret.compose(next_instr, qargs=qargs)
        return ret._z, ret._x, ret._phase

    # ---------------------------------------------------------------------
    # DEPRECATED methods from old Pauli class
    # ---------------------------------------------------------------------

    @staticmethod
    def _make_np_bool(arr):
        if not isinstance(arr, (list, np.ndarray, tuple)):
            arr = [arr]
        arr = np.asarray(arr).astype(np.bool)
        return arr

    @staticmethod
    def from_label(label):
        """DEPRECATED: Construct a Pauli from a string label.

        This function is deprecated use ``Pauli(label)`` instead.

        Args:
            label (str): Pauli string label.

        Returns:
            Pauli: the constructed Pauli.

        Raises:
            QiskitError: If the input list is empty or contains invalid
            Pauli strings.
        """
        warn('`from_label` is deprecated and will be removed no earlier than '
             '3 months after the release date. Use Pauli(label) instead.',
             DeprecationWarning, stacklevel=2)
        return Pauli(label)

    def to_label(self):
        """DEPRECATED: Convert a Pauli to an unsigned string label.

        This function is deprecated use ``str(pauli)`` to convert to the
        full Pauli group label, or ``str(pauli[:])`` to convert to the
        unsigned Pauli group label (coeff = +1)

        Returns:
            str: the Pauli string label.
        """
        warn('`to_label` is deprecated and will be removed no earlier than '
             '3 months after the release date. Use str(Pauli) to convert to '
             'full Pauli group label, or str(Pauli[:]) to convert to the '
             'unsigned Pauli group label (coeff = +1)',
             DeprecationWarning, stacklevel=2)
        return str(self[:])

    @staticmethod
    def sgn_prod(p1, p2):
        r"""
        DEPRECATED: Multiply two Paulis and track the phase.

        This function is deprecated. The Pauli class now handles full
        Pauli group multiplication using :meth:`compose` or :meth:`dot`.

        $P_3 = P_1 \otimes P_2$: X*Y

        Args:
            p1 (Pauli): pauli 1
            p2 (Pauli): pauli 2

        Returns:
            Pauli: the multiplied pauli
            complex: the sign of the multiplication, 1, -1, 1j or -1j
        """
        warn('sgn_prod.x is deprecated and will be removed no earlier than '
             '3 months after the release date. Use `dot` instead.',
             DeprecationWarning, stacklevel=2)
        pauli = p1.dot(p2)
        return pauli, (-1j) ** pauli.phase

    def to_spmatrix(self):
        r"""
        DEPRECATED Convert Pauli to a sparse matrix representation (CSR format).

        This function is deprecated. Use :meth:`to_matrix` with kwarg
        ``sparse=True`` instead.

        Returns:
            scipy.sparse.csr_matrix: a sparse matrix with CSR format that
            represents the pauli.
        """
        warn('`to_spmatrix` is deprecated and will be removed no earlier than '
             '3 months after the release date. Use `to_matrix(sparse=True)` instead.',
             DeprecationWarning, stacklevel=2)
        return self.to_matrix(sparse=True)

    def kron(self, other):
        r"""DEPRECATED: Kronecker product of two paulis.

        This function is deprecated. Use :meth:`expand` instead.

        Order is $P_2 (other) \otimes P_1 (self)$

        Args:
            other (Pauli): P2

        Returns:
            Pauli: self
        """
        warn('`kron` is deprecated and will be removed no earlier than '
             '3 months after the release date. Use `tensor` instead.',
             DeprecationWarning, stacklevel=2)
        pauli = self.expand(other)
        self._z = pauli._z
        self._x = pauli._x
        self._phase = pauli._phase
        self._set_dims(pauli.num_qubits * (2,), pauli.num_qubits * (2,))
        return self

    def update_z(self, z, indices=None):
        """
        DEPRECATED: Update partial or entire z.

        This function is deprecated. Use the setter for :attr:`Z` instead.

        Args:
            z (numpy.ndarray or list): to-be-updated z
            indices (numpy.ndarray or list or optional): to-be-updated qubit indices

        Returns:
            Pauli: self

        Raises:
            QiskitError: when updating whole z, the number of qubits must be the same.
        """
        warn('`update_z` is deprecated and will be removed no earlier than '
             '3 months after the release date. Use `Pauli.z = val` or '
             '`Pauli.z[indices] = val` instead.', DeprecationWarning, stacklevel=2)
        z = self._make_np_bool(z)
        if indices is None:
            if len(self.z) != len(z):
                raise QiskitError("During updating whole z, you can not "
                                  "change the number of qubits.")
            self.z = z
        else:
            if not isinstance(indices, list) and not isinstance(indices, np.ndarray):
                indices = [indices]
            for p, idx in enumerate(indices):
                self.z[idx] = z[p]
        return self

    def update_x(self, x, indices=None):
        """
        DEPRECATED: Update partial or entire x.

        This function is deprecated. Use the setter for :attr:`X` instead.

        Args:
            x (numpy.ndarray or list): to-be-updated x
            indices (numpy.ndarray or list or optional): to-be-updated qubit indices

        Returns:
            Pauli: self

        Raises:
            QiskitError: when updating whole x, the number of qubits must be the same.
        """
        warn('`update_z` is deprecated and will be removed no earlier than '
             '3 months after the release date. Use `Pauli.x = val` or '
             '`Pauli.x[indices] = val` instead.', DeprecationWarning, stacklevel=2)
        x = self._make_np_bool(x)
        if indices is None:
            if len(self.x) != len(x):
                raise QiskitError("During updating whole x, you can not change "
                                  "the number of qubits.")
            self.x = x
        else:
            if not isinstance(indices, list) and not isinstance(indices, np.ndarray):
                indices = [indices]
            for p, idx in enumerate(indices):
                self.x[idx] = x[p]

        return self

    def insert_paulis(self, indices=None, paulis=None, pauli_labels=None):
        """
        DEPRECATED: Insert or append pauli to the targeted indices.

        This function is deprecated. Similar functionality can be obtained
        using the :meth:`insert` method.

        If indices is None, it means append at the end.

        Args:
            indices (list[int]): the qubit indices to be inserted
            paulis (Pauli): the to-be-inserted or appended pauli
            pauli_labels (list[str]): the to-be-inserted or appended pauli label

        Note:
            the indices refers to the location of original paulis,
            e.g. if indices = [0, 2], pauli_labels = ['Z', 'I'] and original pauli = 'ZYXI'
            the pauli will be updated to ZY'I'XI'Z'
            'Z' and 'I' are inserted before the qubit at 0 and 2.

        Returns:
            Pauli: self

        Raises:
            QiskitError: provide both `paulis` and `pauli_labels` at the same time
        """
        warn('`insert_paulis` is deprecated and will be removed no earlier than '
             '3 months after the release date. For similar functionality use '
             '`Pauli.insert` instead.',
             DeprecationWarning, stacklevel=2)
        if pauli_labels is not None:
            if paulis is not None:
                raise QiskitError("Please only provide either `paulis` or `pauli_labels`")
            if isinstance(pauli_labels, str):
                pauli_labels = list(pauli_labels)
            # since pauli label is in reversed order.
            label = ''.join(pauli_labels[::-1])
            paulis = self.from_label(label)

        # Insert and update self
        if indices is None:  # append
            z = np.concatenate((self.z, paulis.z))
            x = np.concatenate((self.x, paulis.x))
        else:
            if not isinstance(indices, list):
                indices = [indices]
            z = np.insert(self.z, indices, paulis.z)
            x = np.insert(self.x, indices, paulis.x)
        pauli = Pauli(z, x, self.phase + paulis.phase)
        self._z = pauli._z
        self._x = pauli._x
        self._phase = pauli._phase
        self._num_qubits = pauli._num_qubits
        return self

    def append_paulis(self, paulis=None, pauli_labels=None):
        """
        DEPRECATED: Append pauli at the end.

        Args:
            paulis (Pauli): the to-be-inserted or appended pauli
            pauli_labels (list[str]): the to-be-inserted or appended pauli label

        Returns:
            Pauli: self
        """
        warn('`append_paulis` is deprecated and will be removed no earlier than '
             '3 months after the release date. Use `Pauli.expand` instead.',
             DeprecationWarning, stacklevel=2)
        return self.insert_paulis(None, paulis=paulis, pauli_labels=pauli_labels)

    def delete_qubits(self, indices):
        """
        DEPRECATED: Delete pauli at the indices.

        This function is deprecated. Equivalent functionality can be obtained
        using the :meth:`delete` method.

        Args:
            indices(list[int]): the indices of to-be-deleted paulis

        Returns:
            Pauli: self
        """
        warn('`append_paulis` is deprecated and will be removed no earlier than '
             '3 months after the release date. For equivalent functionality '
             'use `Pauli.delete` instead.',
             DeprecationWarning, stacklevel=2)
        pauli = self.delete(indices)
        self._z = pauli._z
        self._x = pauli._x
        self._phase = pauli._phase
        self._num_qubits = pauli._num_qubits
        return self

    @classmethod
    def pauli_single(cls, num_qubits, index, pauli_label):
        """
        DEPRECATED: Generate single qubit pauli at index with pauli_label with length num_qubits.

        Args:
            num_qubits (int): the length of pauli
            index (int): the qubit index to insert the single qubit
            pauli_label (str): pauli

        Returns:
            Pauli: single qubit pauli
        """
        warn('`pauli_single` is deprecated and will be removed no earlier than '
             '3 months after the release date.',
             DeprecationWarning, stacklevel=2)
        tmp = Pauli(pauli_label)
        ret = Pauli(np.zeros(num_qubits, dtype=np.bool),
                    np.zeros(num_qubits, dtype=np.bool))
        ret.x[index] = tmp.x[0]
        ret.z[index] = tmp.z[0]
        ret.phase = tmp.phase
        return ret

    @classmethod
    def random(cls, num_qubits, seed=None):
        """DEPRECATED: Return a random Pauli on number of qubits.

        This function is deprecated use
        :func:`~qiskit.quantum_info.random_pauli` instead.

        Args:
            num_qubits (int): the number of qubits
            seed (int): Optional. To set a random seed.
        Returns:
            Pauli: the random pauli
        """
        warn('`random` is deprecated and will be removed no earlier than '
             '3 months after the release date. '
             'Use `qiskit.quantum_info.random_pauli` instead',
             DeprecationWarning, stacklevel=2)
        # pylint: disable=cyclic-import
        from qiskit.quantum_info.operators.symplectic.random import random_pauli
        return random_pauli(num_qubits, group_phase=False, seed=seed)


# ---------------------------------------------------------------------
# Label parsing helper functions
# ---------------------------------------------------------------------

def _split_pauli_label(label):
    """Split Pauli label into unsigned group label and coefficient label"""
    span = re.search(r'[IXYZ]+', label).span()
    pauli = label[span[0]:]
    coeff = label[:span[0]]
    if span[1] != len(label):
        raise QiskitError('Pauli string is not valid.')
    return pauli, coeff

def _phase_from_label(label):
    """Return the phase from a label"""
    # Returns None if label is invalid
    label = label.replace('+', '', 1).replace('1', '', 1).replace('j', 'i', 1)
    phases = {'': 0, '-i': 1, '-': 2, 'i': 3}
    return phases.get(label)
