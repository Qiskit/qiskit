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
N-qubit Pauli Operator Class
"""
# pylint: disable=invalid-name
# pylint: disable=bad-docstring-quotes  # for deprecate_function decorator

from typing import Dict
import re

import numpy as np

from qiskit.circuit import Instruction, QuantumCircuit
from qiskit.circuit.barrier import Barrier
from qiskit.circuit.library.generalized_gates import PauliGate
from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.symplectic.base_pauli import BasePauli
from qiskit.utils.deprecation import deprecate_function


class Pauli(BasePauli):
    r"""N-qubit Pauli operator.

    This class represents an operator :math:`P` from the full :math:`n`-qubit
    *Pauli* group

    .. math::

        P = (-i)^{q} P_{n-1} \otimes ... \otimes P_{0}

    where :math:`q\in \mathbb{Z}_4` and :math:`P_i \in \{I, X, Y, Z\}`
    are single-qubit Pauli matrices:

    .. math::

        I = \begin{pmatrix} 1 & 0  \\ 0 & 1  \end{pmatrix},
        X = \begin{pmatrix} 0 & 1  \\ 1 & 0  \end{pmatrix},
        Y = \begin{pmatrix} 0 & -i \\ i & 0  \end{pmatrix},
        Z = \begin{pmatrix} 1 & 0  \\ 0 & -1 \end{pmatrix}.

    **Initialization**

    A Pauli object can be initialized in several ways:

        ``Pauli(obj)``
            where ``obj`` is a Pauli string, ``Pauli`` or
            :class:`~qiskit.quantum_info.ScalarOp` operator, or a Pauli
            gate or :class:`~qiskit.QuantumCircuit` containing only
            Pauli gates.

        ``Pauli((z, x, phase))``
            where ``z`` and ``x`` are boolean ``numpy.ndarrays`` and ``phase`` is
            an integer in ``[0, 1, 2, 3]``.

        ``Pauli((z, x))``
            equivalent to ``Pauli((z, x, 0))`` with trivial phase.

    **String representation**

    An :math:`n`-qubit Pauli may be represented by a string consisting of
    :math:`n` characters from ``['I', 'X', 'Y', 'Z']``, and optionally phase
    coefficient in :math:`['', '-i', '-', 'i']`. For example: ``XYZ`` or
    ``'-iZIZ'``.

    In the string representation qubit-0 corresponds to the right-most
    Pauli character, and qubit-:math:`(n-1)` to the left-most Pauli
    character. For example ``'XYZ'`` represents
    :math:`X\otimes Y \otimes Z` with ``'Z'`` on qubit-0,
    ``'Y'`` on qubit-1, and ``'X'`` on qubit-3.

    The string representation can be converted to a ``Pauli`` using the
    class initialization (``Pauli('-iXYZ')``). A ``Pauli`` object can be
    converted back to the string representation using the
    :meth:`to_label` method or ``str(pauli)``.

    .. note::

        Using ``str`` to convert a ``Pauli`` to a string will truncate the
        returned string for large numbers of qubits while :meth:`to_label`
        will return the full string with no truncation. The default
        truncation length is 50 characters. The default value can be
        changed by setting the class `__truncate__` attribute to an integer
        value. If set to ``0`` no truncation will be performed.

    **Array Representation**

    The internal data structure of an :math:`n`-qubit Pauli is two
    length-:math:`n` boolean vectors :math:`z \in \mathbb{Z}_2^N`,
    :math:`x \in \mathbb{Z}_2^N`, and an integer :math:`q \in \mathbb{Z}_4`
    defining the Pauli operator

    .. math::

        P &= (-i)^{q + z\cdot x} Z^z \cdot X^x.

    The :math:`k`th qubit corresponds to the :math:`k`th entry in the
    :math:`z` and :math:`x` arrays

    .. math::

        P &= P_{n-1} \otimes ... \otimes P_{0} \\
        P_k &= (-i)^{z[k] * x[k]} Z^{z[k]}\cdot X^{x[k]}

    where ``z[k] = P.z[k]``, ``x[k] = P.x[k]`` respectively.

    The :math:`z` and :math:`x` arrays can be accessed and updated using
    the :attr:`z` and :attr:`x` properties respectively. The phase integer
    :math:`q` can be accessed and updated using the :attr:`phase` property.

    **Matrix Operator Representation**

    Pauli's can be converted to :math:`(2^n, 2^n)`
    :class:`~qiskit.quantum_info.Operator` using the :meth:`to_operator` method,
    or to a dense or sparse complex matrix using the :meth:`to_matrix` method.

    **Data Access**

    The individual qubit Paulis can be accessed and updated using the ``[]``
    operator which accepts integer, lists, or slices for selecting subsets
    of Paulis. Note that selecting subsets of Pauli's will discard the
    phase of the current Pauli.

    For example

    .. code:

        p = Pauli('-iXYZ')

        print('P[0] =', repr(P[0]))
        print('P[1] =', repr(P[1]))
        print('P[2] =', repr(P[2]))
        print('P[:] =', repr(P[:]))
        print('P[::-1] =, repr(P[::-1]))
    """
    # Set the max Pauli string size before truncation
    __truncate__ = 50

    _VALID_LABEL_PATTERN = re.compile(r"^[+-]?1?[ij]?[IXYZ]+$")

    def __init__(self, data=None, x=None, *, z=None, label=None):
        """Initialize the Pauli.

        When using the symplectic array input data both z and x arguments must
        be provided, however the first (z) argument can be used alone for string
        label, Pauli operator, or ScalarOp input data.

        Args:
            data (str or tuple or Pauli or ScalarOp): input data for Pauli. If input is
                a tuple it must be of the form ``(z, x)`` or (z, x, phase)`` where
                ``z`` and ``x`` are boolean Numpy arrays, and phase is an integer from Z_4.
                If input is a string, it must be a concatenation of a phase and a Pauli string
                (e.g. 'XYZ', '-iZIZ') where a phase string is a combination of at most three
                characters from ['+', '-', ''], ['1', ''], and ['i', 'j', ''] in this order,
                e.g. '', '-1j' while a Pauli string is 1 or more characters of 'I', 'X', 'Y' or 'Z',
                e.g. 'Z', 'XIYY'.
            x (np.ndarray): DEPRECATED, symplectic x vector.
            z (np.ndarray): DEPRECATED, symplectic z vector.
            label (str): DEPRECATED, string label.

        Raises:
            QiskitError: if input array is invalid shape.
        """
        if isinstance(data, BasePauli):
            base_z, base_x, base_phase = data._z, data._x, data._phase
        elif isinstance(data, tuple):
            if len(data) not in [2, 3]:
                raise QiskitError(
                    "Invalid input tuple for Pauli, input tuple must be"
                    " `(z, x, phase)` or `(z, x)`"
                )
            base_z, base_x, base_phase = self._from_array(*data)
        elif isinstance(data, str):
            base_z, base_x, base_phase = self._from_label(data)
        elif isinstance(data, ScalarOp):
            base_z, base_x, base_phase = self._from_scalar_op(data)
        elif isinstance(data, (QuantumCircuit, Instruction)):
            base_z, base_x, base_phase = self._from_circuit(data)
        elif x is not None:  # DEPRECATED
            if z is None:
                # Using old Pauli initialization with positional args instead of kwargs
                z = data
            base_z, base_x, base_phase = self._from_array_deprecated(z, x)
        elif label is not None:  # DEPRECATED
            base_z, base_x, base_phase = self._from_label_deprecated(label)
        else:
            raise QiskitError("Invalid input data for Pauli.")

        # Initialize BasePauli
        if base_z.shape[0] != 1:
            raise QiskitError("Input is not a single Pauli")
        super().__init__(base_z, base_x, base_phase)

    def __repr__(self):
        """Display representation."""
        return f"Pauli('{self.__str__()}')"

    def __str__(self):
        """Print representation."""
        if self.__truncate__ and self.num_qubits > self.__truncate__:
            front = self[-self.__truncate__ :].to_label()
            return front + "..."
        return self.to_label()

    def __array__(self, dtype=None):
        if dtype:
            return np.asarray(self.to_matrix(), dtype=dtype)
        return self.to_matrix()

    @classmethod
    def set_truncation(cls, val):
        """Set the max number of Pauli characters to display before truncation/

        Args:
            val (int): the number of characters.

        .. note::

            Truncation will be disabled if the truncation value is set to 0.
        """
        cls.__truncate__ = int(val)

    def __eq__(self, other):
        """Test if two Paulis are equal."""
        if not isinstance(other, BasePauli):
            return False
        return self._eq(other)

    def equiv(self, other):
        """Return True if Pauli's are equivalent up to group phase.

        Args:
            other (Pauli): an operator object.

        Returns:
            bool: True if the Pauli's are equivalent up to group phase.
        """
        if not isinstance(other, Pauli):
            try:
                other = Pauli(other)
            except QiskitError:
                return False
        return np.all(self._z == other._z) and np.all(self._x == other._x)

    @property
    def settings(self) -> Dict:
        """Return settings."""
        return {"data": self.to_label()}

    # ---------------------------------------------------------------------
    # Direct array access
    # ---------------------------------------------------------------------
    @property
    def phase(self):
        """Return the group phase exponent for the Pauli."""
        # Convert internal ZX-phase convention of BasePauli to group phase
        return np.mod(self._phase - self._count_y(), 4)[0]

    @phase.setter
    def phase(self, value):
        # Convert group phase convention to internal ZX-phase convention
        self._phase[:] = np.mod(value + self._count_y(), 4)

    @property
    def x(self):
        """The x vector for the Pauli."""
        return self._x[0]

    @x.setter
    def x(self, val):
        self._x[0, :] = val

    @property
    def z(self):
        """The z vector for the Pauli."""
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
        if isinstance(qubits, (int, np.integer)):
            qubits = [qubits]
        return Pauli((self.z[qubits], self.x[qubits]))

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
        if isinstance(qubits, (int, np.integer)):
            qubits = [qubits]
        if max(qubits) > self.num_qubits - 1:
            raise QiskitError(
                "Qubit index is larger than the number of qubits "
                "({}>{}).".format(max(qubits), self.num_qubits - 1)
            )
        if len(qubits) == self.num_qubits:
            raise QiskitError("Cannot delete all qubits of Pauli")
        z = np.delete(self._z, qubits, axis=1)
        x = np.delete(self._x, qubits, axis=1)
        return Pauli((z, x, self.phase))

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
        ret = Pauli((np.zeros(ret_qubits, dtype=bool), np.zeros(ret_qubits, dtype=bool)))
        if isinstance(qubits, (int, np.integer)):
            if value.num_qubits == 1:
                qubits = [qubits]
            else:
                qubits = list(range(qubits, qubits + value.num_qubits))
        if len(qubits) != value.num_qubits:
            raise QiskitError(
                "Number of indices does not match number of qubits for "
                "the inserted Pauli ({}!={})".format(len(qubits), value.num_qubits)
            )
        if max(qubits) > ret.num_qubits - 1:
            raise QiskitError(
                "Index is too larger for combined Pauli number of qubits "
                "({}>{}).".format(max(qubits), ret.num_qubits - 1)
            )
        # Qubit positions for original op
        self_qubits = [i for i in range(ret.num_qubits) if i not in qubits]
        ret[self_qubits] = self
        ret[qubits] = value
        return ret

    # ---------------------------------------------------------------------
    # Representation conversions
    # ---------------------------------------------------------------------

    def __hash__(self):
        """Make hashable based on string representation."""
        return hash(self.to_label())

    def to_label(self):
        """Convert a Pauli to a string label.

        .. note::

            The difference between `to_label` and :meth:`__str__` is that
            the later will truncate the output for large numbers of qubits.

        Returns:
            str: the Pauli string label.
        """
        return self._to_label(self.z, self.x, self._phase[0])

    def to_matrix(self, sparse=False):
        r"""Convert to a Numpy array or sparse CSR matrix.

        Args:
            sparse (bool): if True return sparse CSR matrices, otherwise
                           return dense Numpy arrays (default: False).

        Returns:
            array: The Pauli matrix.
        """
        return self._to_matrix(self.z, self.x, self._phase[0], sparse=sparse)

    def to_instruction(self):
        """Convert to Pauli circuit instruction."""
        from math import pi

        pauli, phase = self._to_label(
            self.z, self.x, self._phase[0], full_group=False, return_phase=True
        )
        if len(pauli) == 1:
            gate = {"I": IGate(), "X": XGate(), "Y": YGate(), "Z": ZGate()}[pauli]
        else:
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

    def compose(self, other, qargs=None, front=False, inplace=False):
        """Return the operator composition with another Pauli.

        Args:
            other (Pauli): a Pauli object.
            qargs (list or None): Optional, qubits to apply dot product
                                  on (default: None).
            front (bool): If True compose using right operator multiplication,
                          instead of left multiplication [default: False].
            inplace (bool): If True update in-place (default: False).

        Returns:
            Pauli: The composed Pauli.

        Raises:
            QiskitError: if other cannot be converted to an operator, or has
                         incompatible dimensions for specified subsystems.

        .. note::
            Composition (``&``) by default is defined as `left` matrix multiplication for
            matrix operators, while :meth:`dot` is defined as `right` matrix
            multiplication. That is that ``A & B == A.compose(B)`` is equivalent to
            ``B.dot(A)`` when ``A`` and ``B`` are of the same type.

            Setting the ``front=True`` kwarg changes this to `right` matrix
            multiplication and is equivalent to the :meth:`dot` method
            ``A.dot(B) == A.compose(B, front=True)``.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if not isinstance(other, Pauli):
            other = Pauli(other)
        return Pauli(super().compose(other, qargs=qargs, front=front, inplace=inplace))

    # pylint: disable=arguments-differ
    def dot(self, other, qargs=None, inplace=False):
        """Return the right multiplied operator self * other.

        Args:
            other (Pauli): an operator object.
            qargs (list or None): Optional, qubits to apply dot product
                                  on (default: None).
            inplace (bool): If True update in-place (default: False).

        Returns:
            Pauli: The operator self * other.
        """
        return self.compose(other, qargs=qargs, front=True, inplace=inplace)

    def tensor(self, other):
        if not isinstance(other, Pauli):
            other = Pauli(other)
        return Pauli(super().tensor(other))

    def expand(self, other):
        if not isinstance(other, Pauli):
            other = Pauli(other)
        return Pauli(super().expand(other))

    def _multiply(self, other):
        return Pauli(super()._multiply(other))

    def conjugate(self):
        return Pauli(super().conjugate())

    def transpose(self):
        return Pauli(super().transpose())

    def adjoint(self):
        return Pauli(super().adjoint())

    def inverse(self):
        """Return the inverse of the Pauli."""
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
            qargs = getattr(other, "qargs", None)
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
        r"""Heisenberg picture evolution of a Pauli by a Clifford.

        This returns the Pauli :math:`P^\prime = C^\dagger.P.C`.

        Args:
            other (Pauli or Clifford or QuantumCircuit): The Clifford operator to evolve by.
            qargs (list): a list of qubits to apply the Clifford to.

        Returns:
            Pauli: the Pauli :math:`C^\dagger.P.C`.

        Raises:
            QiskitError: if the Clifford number of qubits and qargs don't match.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)

        # pylint: disable=cyclic-import
        from qiskit.quantum_info.operators.symplectic.clifford import Clifford

        if not isinstance(other, (Pauli, Instruction, QuantumCircuit, Clifford)):
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
        if Pauli._VALID_LABEL_PATTERN.match(label) is None:
            raise QiskitError(f'Pauli string label "{label}" is not valid.')

        # Split string into coefficient and Pauli
        pauli, coeff = _split_pauli_label(label)

        # Convert coefficient to phase
        phase = 0 if not coeff else _phase_from_label(coeff)

        # Convert to Symplectic representation
        num_qubits = len(pauli)
        base_z = np.zeros((1, num_qubits), dtype=bool)
        base_x = np.zeros((1, num_qubits), dtype=bool)
        base_phase = np.array([phase], dtype=int)
        for i, char in enumerate(pauli):
            if char == "X":
                base_x[0, num_qubits - 1 - i] = True
            elif char == "Z":
                base_z[0, num_qubits - 1 - i] = True
            elif char == "Y":
                base_x[0, num_qubits - 1 - i] = True
                base_z[0, num_qubits - 1 - i] = True
                base_phase += 1
        return base_z, base_x, base_phase % 4

    @classmethod
    def _from_scalar_op(cls, op):
        """Convert a ScalarOp to BasePauli data."""
        if op.num_qubits is None:
            raise QiskitError(f"{op} is not an N-qubit identity")
        base_z = np.zeros((1, op.num_qubits), dtype=bool)
        base_x = np.zeros((1, op.num_qubits), dtype=bool)
        base_phase = np.mod(
            cls._phase_from_complex(op.coeff) + np.sum(np.logical_and(base_z, base_x), axis=1), 4
        )
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
                raise QiskitError(f"Cannot apply Instruction: {instr.name}")
            # Convert to circuit
            instr = instr.definition

        # Initialize identity Pauli
        ret = Pauli(
            BasePauli(
                np.zeros((1, instr.num_qubits), dtype=bool),
                np.zeros((1, instr.num_qubits), dtype=bool),
                np.zeros(1, dtype=int),
            )
        )

        # Add circuit global phase if specified
        if instr.global_phase:
            ret.phase = cls._phase_from_complex(np.exp(1j * float(instr.global_phase)))

        # Recursively apply instructions
        for dinstr, qregs, cregs in instr.data:
            if cregs:
                raise QiskitError(
                    f"Cannot apply instruction with classical registers: {dinstr.name}"
                )
            if not isinstance(dinstr, Barrier):
                next_instr = BasePauli(*cls._from_circuit(dinstr))
                if next_instr is not None:
                    qargs = [tup.index for tup in qregs]
                    ret = ret.compose(next_instr, qargs=qargs)
        return ret._z, ret._x, ret._phase

    # ---------------------------------------------------------------------
    # DEPRECATED methods from old Pauli class
    # ---------------------------------------------------------------------

    @classmethod
    @deprecate_function(
        "Initializing Pauli from `Pauli(label=l)` kwarg is deprecated as of "
        "version 0.17.0 and will be removed no earlier than 3 months after "
        "the release date. Use `Pauli(l)` instead."
    )
    def _from_label_deprecated(cls, label):
        # Deprecated wrapper of `_from_label` so that a deprecation warning
        # can be displaced during initialization with deprecated kwarg
        return cls._from_label(label)

    @classmethod
    @deprecate_function(
        "Initializing Pauli from `Pauli(z=z, x=x)` kwargs is deprecated as of "
        "version 0.17.0 and will be removed no earlier than 3 months after "
        "the release date. Use tuple initialization `Pauli((z, x))` instead."
    )
    def _from_array_deprecated(cls, z, x):
        # Deprecated wrapper of `_from_array` so that a deprecation warning
        # can be displaced during initialization with deprecated kwarg
        return cls._from_array(z, x)

    @staticmethod
    def _make_np_bool(arr):
        if not isinstance(arr, (list, np.ndarray, tuple)):
            arr = [arr]
        arr = np.asarray(arr).astype(bool)
        return arr

    @staticmethod
    @deprecate_function(
        "`from_label` is deprecated and will be removed no earlier than "
        "3 months after the release date. Use Pauli(label) instead."
    )
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
        if isinstance(label, tuple):
            # Legacy usage from aqua
            label = "".join(label)
        return Pauli(label)

    @staticmethod
    @deprecate_function(
        "sgn_prod is deprecated and will be removed no earlier than "
        "3 months after the release date. Use `dot` instead."
    )
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
            Pauli: the multiplied pauli (without phase)
            complex: the sign of the multiplication, 1, -1, 1j or -1j
        """
        pauli = p1.dot(p2)
        return pauli[:], (-1j) ** pauli.phase

    @deprecate_function(
        "`to_spmatrix` is deprecated and will be removed no earlier than "
        "3 months after the release date. Use `to_matrix(sparse=True)` instead."
    )
    def to_spmatrix(self):
        r"""
        DEPRECATED Convert Pauli to a sparse matrix representation (CSR format).

        This function is deprecated. Use :meth:`to_matrix` with kwarg
        ``sparse=True`` instead.

        Returns:
            scipy.sparse.csr_matrix: a sparse matrix with CSR format that
            represents the pauli.
        """
        return self.to_matrix(sparse=True)

    @deprecate_function(
        "`kron` is deprecated and will be removed no earlier than "
        "3 months after the release date of Qiskit Terra 0.17.0. "
        "Use `expand` instead, but note this does not change "
        "the operator in-place."
    )
    def kron(self, other):
        r"""DEPRECATED: Kronecker product of two paulis.

        This function is deprecated. Use :meth:`expand` instead.

        Order is $P_2 (other) \otimes P_1 (self)$

        Args:
            other (Pauli): P2

        Returns:
            Pauli: self
        """
        pauli = self.expand(other)
        self._z = pauli._z
        self._x = pauli._x
        self._phase = pauli._phase
        self._op_shape = self._op_shape.expand(other._op_shape)
        return self

    @deprecate_function(
        "`update_z` is deprecated and will be removed no earlier than "
        "3 months after the release date. Use `Pauli.z = val` or "
        "`Pauli.z[indices] = val` instead."
    )
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
        phase = self.phase
        z = self._make_np_bool(z)
        if indices is None:
            if len(self.z) != len(z):
                raise QiskitError(
                    "During updating whole z, you can not " "change the number of qubits."
                )
            self.z = z
        else:
            if not isinstance(indices, list) and not isinstance(indices, np.ndarray):
                indices = [indices]
            for p, idx in enumerate(indices):
                self.z[idx] = z[p]
        self.phase = phase
        return self

    @deprecate_function(
        "`update_z` is deprecated and will be removed no earlier than "
        "3 months after the release date. Use `Pauli.x = val` or "
        "`Pauli.x[indices] = val` instead."
    )
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
        phase = self.phase
        x = self._make_np_bool(x)
        if indices is None:
            if len(self.x) != len(x):
                raise QiskitError(
                    "During updating whole x, you can not change " "the number of qubits."
                )
            self.x = x
        else:
            if not isinstance(indices, list) and not isinstance(indices, np.ndarray):
                indices = [indices]
            for p, idx in enumerate(indices):
                self.x[idx] = x[p]
        self.phase = phase
        return self

    @deprecate_function(
        "`insert_paulis` is deprecated and will be removed no earlier than "
        "3 months after the release date. For similar functionality use "
        "`Pauli.insert` instead."
    )
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
        if pauli_labels is not None:
            if paulis is not None:
                raise QiskitError("Please only provide either `paulis` or `pauli_labels`")
            if isinstance(pauli_labels, str):
                pauli_labels = list(pauli_labels)
            # since pauli label is in reversed order.
            label = "".join(pauli_labels[::-1])
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
        pauli = Pauli((z, x, self.phase + paulis.phase))
        self._z = pauli._z
        self._x = pauli._x
        self._phase = pauli._phase
        self._op_shape = pauli._op_shape
        return self

    @deprecate_function(
        "`append_paulis` is deprecated and will be removed no earlier than "
        "3 months after the release date. Use `Pauli.expand` instead."
    )
    def append_paulis(self, paulis=None, pauli_labels=None):
        """
        DEPRECATED: Append pauli at the end.

        Args:
            paulis (Pauli): the to-be-inserted or appended pauli
            pauli_labels (list[str]): the to-be-inserted or appended pauli label

        Returns:
            Pauli: self
        """
        return self.insert_paulis(None, paulis=paulis, pauli_labels=pauli_labels)

    @deprecate_function(
        "`append_paulis` is deprecated and will be removed no earlier than "
        "3 months after the release date. For equivalent functionality "
        "use `Pauli.delete` instead."
    )
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
        pauli = self.delete(indices)
        self._z = pauli._z
        self._x = pauli._x
        self._phase = pauli._phase
        self._op_shape = pauli._op_shape
        return self

    @classmethod
    @deprecate_function(
        "`pauli_single` is deprecated and will be removed no earlier than "
        "3 months after the release date."
    )
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
        tmp = Pauli(pauli_label)
        ret = Pauli((np.zeros(num_qubits, dtype=bool), np.zeros(num_qubits, dtype=bool)))
        ret.x[index] = tmp.x[0]
        ret.z[index] = tmp.z[0]
        ret.phase = tmp.phase
        return ret

    @classmethod
    @deprecate_function(
        "`random` is deprecated and will be removed no earlier than "
        "3 months after the release date. "
        "Use `qiskit.quantum_info.random_pauli` instead"
    )
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
        # pylint: disable=cyclic-import
        from qiskit.quantum_info.operators.symplectic.random import random_pauli

        return random_pauli(num_qubits, group_phase=False, seed=seed)


# ---------------------------------------------------------------------
# Label parsing helper functions
# ---------------------------------------------------------------------


def _split_pauli_label(label):
    """Split Pauli label into unsigned group label and coefficient label"""
    span = re.search(r"[IXYZ]+", label).span()
    pauli = label[span[0] :]
    coeff = label[: span[0]]
    if span[1] != len(label):
        invalid = set(re.sub(r"[IXYZ]+", "", label[span[0] :]))
        raise QiskitError(
            "Pauli string contains invalid characters " "{} ∉ ['I', 'X', 'Y', 'Z']".format(invalid)
        )
    return pauli, coeff


def _phase_from_label(label):
    """Return the phase from a label"""
    # Returns None if label is invalid
    label = label.replace("+", "", 1).replace("1", "", 1).replace("j", "i", 1)
    phases = {"": 0, "-i": 1, "-": 2, "i": 3}
    if label not in phases:
        raise QiskitError("Invalid Pauli phase label '{}'".format(label))
    return phases[label]


# Update docstrings for API docs
generate_apidocs(Pauli)
