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

from warnings import warn
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.circuit import QuantumCircuit, Instruction
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.circuit.barrier import Barrier
from qiskit.quantum_info.operators.symplectic.pauli_tools import (
    evolve_pauli, pauli_from_label, pauli_to_label, pauli_to_matrix,
    coeff_phase_from_complex)


class Pauli(BaseOperator):
    r"""N-qubit Pauli group operator.

    **Symplectic Representation**

    The symplectic representation of a single-qubit Pauli matrix
    is a pair of boolean values :math:`[x, z]` and phase `q` such that the
    Pauli matrix is given by :math:`P = (-i)^{q} \sigma_z^z.\sigma_x^x`.
    Where :math:`z, x \in \mathbb{Z}_2` and :math:`q \in \mathbb{Z}_4`.

    **Data Access**

    The individual qubit Paulis can be accessed using the list access ``[]`` operator.
    The underlying Numpy array can be directly accessed using the :attr:`array` property,
    and the sub-arrays for the `X` or `Z` part can be accessed using the
    :attr:`X` and :attr:`Z` properties respectively.
    """
    # pylint: disable = missing-param-doc, missing-type-doc
    def __init__(self, data=None, phase=0, *, z=None, x=None, label=None):
        """Initialize the Pauli.

        Args:
            data (array or str or ScalarOp or Pauli): input data.
            phase (int): phase exponent from Z_4.

        Raises:
            QiskitError: if input array is invalid shape.

        Additional Information:
            The input array is not copied so Paulis can share the same array
        """
        # Check for deprecated initialization from legacy Pauli
        if isinstance(phase, (list, tuple, np.ndarray)):
            # Positional use of deprecated kwargs z, x
            z, x = data, phase
            data = None
            phase = 0
        if x is not None or z is not None or label is not None:
            warn('Initializing Pauli from z, x or label kwargs is deprecated '
                 'and will be removed no earlier than 3 months after the release date. '
                 'See Pauli API documentation new use.',
                 DeprecationWarning, stacklevel=2)
            data = self._deprecated_init(z=z, x=x, label=label)
            phase = 0

        # New operator initialization
        self._phase = int(phase) % 4
        if isinstance(data, (np.ndarray, list)):
            self._array = np.asarray(data, dtype=np.bool)
            if self._array.ndim != 1 or self._array.shape[0] % 2 != 0:
                raise QiskitError("Invalid array shape for Pauli.")
        elif isinstance(data, Pauli):
            # Share underlying array
            self._array = data._array
            self._phase = data._phase
        elif isinstance(data, str):
            self._array, self._phase = pauli_from_label(data)
        elif isinstance(data, ScalarOp):
            # Initialize an N-qubit identity
            if data.num_qubits is None:
                raise QiskitError('{} is not an N-qubit identity'.format(data))
            self._phase = coeff_phase_from_complex(data.coeff)
            self._array = np.zeros(2 * data.num_qubits, dtype=np.bool)
        elif isinstance(data, (QuantumCircuit, Instruction)):
            data = self._from_circuit(data)
            self._phase = data._phase
            self._array = data._array
        else:
            raise QiskitError("Invalid input data for Pauli.")

        # Set size properties
        dims = (self._array.shape[0] // 2) * (2, )
        super().__init__(dims, dims)

    def __repr__(self):
        """Display representation."""
        prefix = 'Pauli('
        arr_str = np.array2string(self._array, separator=', ', prefix=prefix)
        if self._phase:
            return '{}{}, phase={})'.format(prefix, arr_str, self._phase)
        return '{}{})'.format(prefix, arr_str)

    def __eq__(self, other):
        """Test if two Paulis are equal."""
        if isinstance(other, Pauli):
            return self._phase == other._phase and np.all(
                self._array == other._array)
        return False

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
        return np.all(self._array == other._array)

    # ---------------------------------------------------------------------
    # Representation conversions
    # ---------------------------------------------------------------------

    def __str__(self):
        """String representation."""
        return pauli_to_label(self._array, self._phase)

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
        return pauli_to_matrix(self._array, self._phase, sparse=sparse)

    def to_operator(self):
        """Convert to a matrix Operator object"""
        from qiskit.quantum_info.operators.operator import Operator
        return Operator(self.to_matrix())

    # ---------------------------------------------------------------------
    # Direct array access
    # ---------------------------------------------------------------------

    @property
    def phase(self):
        """Return the phase exponent of the Pauli."""
        return self._phase

    @phase.setter
    def phase(self, value):
        """Set the phase exponent."""
        self._phase = int(value) % 4

    @property
    def group_phase(self):
        """Return the phase exponent relative to the unsigned Pauli group element."""
        return (self._phase - self._count_y()) % 4

    @group_phase.setter
    def group_phase(self, phase):
        """Set the phase exponent relative to the unsigned Pauli group element."""
        self._phase = int(phase + self._count_y()) % 4

    @property
    def array(self):
        """The underlying boolean array."""
        return self._array

    @array.setter
    def array(self, value):
        """Set the underlying boolean array."""
        # We use [:] array view so that setting the array cannot
        # change the arrays shape.
        self._array[:] = value

    @property
    def X(self):
        """The X block of the :attr:`array`."""
        return self._array[0:self._num_qubits]

    @X.setter
    def X(self, val):
        self._array[0:self._num_qubits] = val

    @property
    def Z(self):
        """The Z block of the :attr:`array`."""
        return self._array[self._num_qubits:2 * self._num_qubits]

    @Z.setter
    def Z(self, val):
        self._array[self._num_qubits:2 * self._num_qubits] = val

    def __len__(self):
        """Return the number of qubits in the Pauli."""
        return self.num_qubits

    # ---------------------------------------------------------------------
    # Pauli Array methods
    # ---------------------------------------------------------------------

    def __getitem__(self, qubits):
        """Return the unsigned Pauli group Pauli for subset of qubits."""
        pauli = Pauli(self._array[self._qubit_indices(qubits)])
        # Set group phase to 0 so returned Pauli is always +1 coeff
        pauli.group_phase = 0
        return pauli

    def __setitem__(self, qubits, value):
        """Update the Pauli for a subset of qubits."""
        if not isinstance(value, Pauli):
            value = Pauli(value)
        self._array[self._qubit_indices(qubits)] = value.array
        # Add extra phase from new Pauli to current
        self.phase = self.phase + value.phase

    def _qubit_indices(self, qubits):
        """Return indices for qubit parts of the underlying array"""
        nq = self._num_qubits
        if isinstance(qubits, (int, np.int)):
            return [qubits, nq + qubits]
        if isinstance(qubits, slice):
            return list(range(nq)[qubits]) + list(range(nq, 2*nq)[qubits])
        return list(qubits) + [self._num_qubits + i for i in qubits]

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
        indices = self._qubit_indices(qubits)

        if max(indices) > len(self._array) - 1:
            raise QiskitError(
                "Qubit index is larger than the number of qubits "
                "({}>{}).".format(max(indices) - self.num_qubits,
                                  self.num_qubits - 1))

        if len(indices) == len(self._array):
            raise QiskitError("Cannot delete all qubits of Pauli")

        ret = Pauli(np.delete(self._array, indices))
        ret.group_phase = self.group_phase
        return ret

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
        ret = Pauli(np.zeros(2 * (self.num_qubits + value.num_qubits),
                             dtype=np.bool))
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
        array = np.concatenate([other.X, self.X, other.Z, self.Z])
        phase = self._phase + other._phase
        return Pauli(array, phase=phase)

    def expand(self, other):
        """Return the tensor product Pauli other ⊗ self.

        Args:
            other (Pauli): another Pauli.

        Returns:
            Pauli: the tensor product Pauli.
        """
        if not isinstance(other, Pauli):
            other = Pauli(other)
        array = np.concatenate([self.X, other.X, self.Z, other.Z])
        phase = self._phase + other._phase
        return Pauli(array, phase=phase)

    # pylint: disable=arguments-differ
    def compose(self, other, qargs=None, front=False, inplace=False):
        """Return the composition channel self∘other.

        Note that this discards phases.

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

        if qargs is None and other.num_qubits != self.num_qubits:
            raise QiskitError(
                "other Pauli must be on the same number of qubits.")

        if qargs and other.num_qubits != len(qargs):
            raise QiskitError(
                "Number of qubits in the other Pauli does not match qargs.")

        ret = self if inplace else self.copy()

        # Compute Pauli array
        if qargs is not None:
            inds = list(qargs)
            X, Z = ret.X[inds], ret.Z[inds]
        else:
            X, Z = ret.X, ret.Z

        # Update phase
        if front:
            phase_shift = 2 * np.sum(1 & (X & other.Z))
        else:
            phase_shift = 2 * np.sum(1 & (Z & other.X))
        ret.phase += other.phase + phase_shift

        # Update Pauli
        X ^= other.X
        Z ^= other.Z

        return ret

    # pylint: disable=arguments-differ
    def dot(self, other, qargs=None, inplace=False):
        """Return the composition channel self∘other.

        Note that this discards phases.

        Args:
            other (Pauli): another Pauli.
            qargs (None or list): qubits to apply dot product on (default: None).
            inplace (bool): If True update in-place (default: False).

        Returns:
            Pauli: the dot outer product table.

        Raises:
            QiskitError: if other cannot be converted to a Pauli.
        """
        return self.compose(other, qargs=qargs, front=True, inplace=inplace)

    def __imul__(self, other):
        return self.dot(other, inplace=True)

    def conjugate(self):
        """Return the conjugate of the operator."""
        if self.phase % 2 == 0:
            return self
        if self.phase == 1:
            return Pauli(self._array, phase=3)
        # Phase is 3
        return Pauli(self._array, phase=1)

    def transpose(self):
        """Return the transpose of the operator."""
        # Transpose sets Y -> -Y. This has effect on changing the phase
        parity_y = self._count_y() % 2
        if parity_y == 0:
            return self
        return Pauli(self._array, phase=self._phase + 2 * parity_y)

    def inverse(self):
        """Return the inverse of the operator."""
        return self.adjoint()

    def _multiply(self, other):
        """Multiply a Pauli by a phase.

        Args:
            other (complex): a complex number in [1, -1j, -1, 1j]

        Returns:
            Pauli: the linear operator other * self.

        Raises:
            QiskitError: if the phase is not in the set [1, -1j, -1, 1j].
        """
        return Pauli(self._array, phase=self._phase + coeff_phase_from_complex(other))

    def to_instruction(self):
        """Convert to Pauli circuit instruction."""
        from math import pi
        pauli, phase = pauli_to_label(self._array,
                                      self._phase,
                                      full_group=False,
                                      return_phase=True)
        gate = PauliGate(pauli)
        if not phase:
            return gate
        # Add global phase
        circuit = QuantumCircuit(self.num_qubits, name=self.to_label())
        circuit.global_phase = -phase * pi / 2
        circuit.append(gate, range(self.num_qubits))
        return circuit.to_instruction()

    @classmethod
    def _from_instruction(cls, instr):
        """Convert compatible instruction to Pauli"""
        from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
        if isinstance(instr, PauliGate):
            return Pauli.from_label(instr.params[0])
        if isinstance(instr, IGate):
            return Pauli.from_label('I')
        if isinstance(instr, XGate):
            return Pauli.from_label('X')
        if isinstance(instr, YGate):
            return Pauli.from_label('Y')
        if isinstance(instr, ZGate):
            return Pauli.from_label('Z')
        return None

    @classmethod
    def _from_circuit(cls, instr):
        """Initialize Pauli from compatible instruction by simulation"""
        # Try and convert single instruction
        if isinstance(instr, Instruction):
            # Check if Pauli instruction
            pauli = cls._from_instruction(instr)
            if pauli:
                return pauli
            # If not check definition for unrolling
            if instr.definition is None:
                raise QiskitError('Cannot apply Instruction: {}'.format(
                    instr.name))
            # Convert to circuit
            instr = instr.definition

        # Recursively convert circuit
        ret = Pauli(np.zeros(2 * instr.num_qubits, dtype=np.bool))
        if instr.global_phase:
            ret.phase = coeff_phase_from_complex(
                np.exp(1j * float(instr.global_phase)))

        for dinstr, qregs, cregs in instr.data:
            if cregs:
                raise QiskitError(
                    'Cannot apply instruction with classical registers: {}'.
                    format(dinstr.name))
            if not isinstance(dinstr, Barrier):
                next_instr = cls._from_circuit(dinstr)
                if next_instr is not None:
                    qargs = [tup.index for tup in qregs]
                    ret = ret.compose(next_instr, qargs=qargs)
        return ret

    # ---------------------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------------------

    def commutes(self, other, qargs=None):
        """Return True if other Pauli commutes with self.

        Args:
            other (Pauli): another Pauli operator.
            qargs (list): qubits to apply dot product on (default: None).

        Returns:
            bool: True if Pauli's commute, False if they anti-commute.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if not isinstance(other, Pauli):
            other = Pauli(other)
        if qargs is not None:
            inds = list(qargs)
            X, Z = self.X[inds], self.Z[inds]
        else:
            X, Z = self.X, self.Z
        return np.sum(X & other.Z) % 2 == np.sum((Z & other.X)) % 2

    def anticommutes(self, other, qargs=None):
        """Return True if other Pauli anticommutes with self.

        Args:
            other (Pauli): another Pauli operator.
            qargs (list): qubits to apply dot product on (default: None).

        Returns:
            bool: True if Pauli's anticommute, False if they commute.
        """
        return not self.commutes(other, qargs=qargs)

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
        from qiskit.quantum_info.operators.symplectic.clifford import Clifford

        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        # Get return vector
        ret = self.copy()

        # Convert quantum circuits to Cliffords
        if isinstance(other, Clifford):
            other = other.to_circuit()
        if not isinstance(other, (Pauli, Instruction, QuantumCircuit)):
            # Convert to a Pauli
            other = Pauli(other)

        # Check dimension
        if len(self.input_dims(qargs)) != other.num_qubits:
            raise QiskitError(
                "Incorrect number of qubits for Clifford {} != {}.".format(
                    other.num_qubits, len(self.input_dims(qargs))))

        # Evolve via Pauli
        if isinstance(other, Pauli):
            ret = self.compose(other, qargs=qargs)
            ret = ret.dot(other.adjoint(), qargs=qargs)
            return ret

        # Otherwise evolve by circuit evolution
        return evolve_pauli(self.copy(), other)

    def _count_y(self):
        """Count the number of Y Pauli's"""
        return np.sum(self.X & self.Z)

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
    def _deprecated_init(z=None, x=None, label=None):
        if label is not None:
            return Pauli(label)
        if z is None:
            raise QiskitError("z vector must not be None.")
        if x is None:
            raise QiskitError("x vector must not be None.")
        if len(z) != len(x):
            raise QiskitError("length of z and x vectors must be "
                              "the same. (z: {} vs x: {})".format(len(z), len(x)))
        x = Pauli._make_np_bool(x)
        z = Pauli._make_np_bool(z)
        phase = np.sum(x & z)
        return Pauli(np.concatenate([x, z]), phase)

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
        pauli, phase = pauli_from_label(label)
        return Pauli(pauli, phase)

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

    @property
    def z(self):
        """DEPRECATED: Getter of z. Use :attr:`Z` instead."""
        warn('Pauli.z is deprecated and will be removed no earlier than '
             '3 months after the release date. Use Pauli.Z instead.',
             DeprecationWarning, stacklevel=2)
        return self.Z

    @property
    def x(self):
        """DEPRECATED: Getter of x. Use :attr:`X` instead."""
        warn('Pauli.x is deprecated and will be removed no earlier than '
             '3 months after the release date. Use Pauli.X instead.',
             DeprecationWarning, stacklevel=2)
        return self.X

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
        return pauli, (-1j) ** pauli.group_phase

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
        self._array = pauli._array
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
             '3 months after the release date. Use `Pauli.Z = val` or '
             '`Pauli.Z[indices] = val` instead.', DeprecationWarning, stacklevel=2)
        z = self._make_np_bool(z)
        if indices is None:
            if len(self.Z) != len(z):
                raise QiskitError("During updating whole z, you can not "
                                  "change the number of qubits.")
            self.Z = z
        else:
            if not isinstance(indices, list) and not isinstance(indices, np.ndarray):
                indices = [indices]
            for p, idx in enumerate(indices):
                self.Z[idx] = z[p]
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
             '3 months after the release date. Use `Pauli.X = val` or '
             '`Pauli.X[indices] = val` instead.', DeprecationWarning, stacklevel=2)
        x = self._make_np_bool(x)
        if indices is None:
            if len(self.X) != len(x):
                raise QiskitError("During updating whole x, you can not change "
                                  "the number of qubits.")
            self.X = x
        else:
            if not isinstance(indices, list) and not isinstance(indices, np.ndarray):
                indices = [indices]
            for p, idx in enumerate(indices):
                self.X[idx] = x[p]

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
            z = np.concatenate((self.Z, paulis.Z))
            x = np.concatenate((self.X, paulis.X))
        else:
            if not isinstance(indices, list):
                indices = [indices]
            z = np.insert(self.Z, indices, paulis.Z)
            x = np.insert(self.X, indices, paulis.X)
        pauli = Pauli(np.concatenate([x, z]), self.phase + paulis.phase)
        self._array = pauli._array
        self._phase = pauli._phase
        self._set_dims(pauli.num_qubits * (2,), pauli.num_qubits * (2,))
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
        self._array = pauli._array
        self._phase = pauli._phase
        self._set_dims(pauli.num_qubits * (2,), pauli.num_qubits * (2,))
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
        ret = Pauli(np.zeros(2 * num_qubits, dtype=np.bool))
        ret.X[index] = tmp.X[0]
        ret.Z[index] = tmp.Z[0]
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
        pauli = random_pauli(num_qubits, group_phase=False, seed=seed)
        return cls(pauli.array, pauli.phase)
