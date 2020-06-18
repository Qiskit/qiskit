# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name,assignment-from-no-return

"""
Tools for working with Pauli Operators.

A simple pauli class and some tools.
"""

import warnings

import numpy as np
from scipy import sparse

from qiskit.exceptions import QiskitError


def _make_np_bool(arr):
    if not isinstance(arr, (list, np.ndarray, tuple)):
        arr = [arr]
    arr = np.asarray(arr).astype(np.bool)
    return arr


def _count_set_bits(i):
    """
    Counts the number of set bits in a uint (or a numpy array of uints).
    """
    i = i - ((i >> 1) & 0x55555555)
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
    return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24


class Pauli:
    """A simple class representing Pauli Operators.

    The form is P_zx = (-i)^dot(z,x) Z^z X^x where z and x are elements of Z_2^n.
    That is, there are 4^n elements (no phases in this group).

    For example, for 1 qubit
    P_00 = Z^0 X^0 = I
    P_01 = X
    P_10 = Z
    P_11 = -iZX = (-i) iY = Y

    The overload __mul__ does not track the sign: P1*P2 = Z^(z1+z2) X^(x1+x2) but
    sgn_prod does __mul__ and track the phase: P1*P2 = (-i)^dot(z1+z2,x1+x2) Z^(z1+z2) X^(x1+x2)
    where the sums are taken modulo 2.

    Pauli vectors z and x are supposed to be defined as boolean numpy arrays.

    Ref.
    Jeroen Dehaene and Bart De Moor
    Clifford group, stabilizer states, and linear and quadratic operations
    over GF(2)
    Phys. Rev. A 68, 042318 – Published 20 October 2003
    """

    def __init__(self, z=None, x=None, label=None):
        r"""Make the Pauli object.

        Note that, for the qubit index:
            - Order of z, x vectors is q_0 ... q_{n-1},
            - Order of pauli label is q_{n-1} ... q_0

        E.g.,
            - z and x vectors: z = [z_0 ... z_{n-1}], x = [x_0 ... x_{n-1}]
            - a pauli is $P_{n-1} \otimes ... \otimes P_0$

        Args:
            z (numpy.ndarray): boolean, z vector
            x (numpy.ndarray): boolean, x vector
            label (str): pauli label
        """
        if label is not None:
            a = Pauli.from_label(label)
            self._z = a.z
            self._x = a.x
        else:
            self._init_from_bool(z, x)

    @classmethod
    def from_label(cls, label):
        r"""Take pauli string to construct pauli.

        The qubit index of pauli label is q_{n-1} ... q_0.
        E.g., a pauli is $P_{n-1} \otimes ... \otimes P_0$

        Args:
            label (str): pauli label

        Returns:
            Pauli: the constructed pauli

        Raises:
            QiskitError: invalid character in the label
        """
        z = np.zeros(len(label), dtype=np.bool)
        x = np.zeros(len(label), dtype=np.bool)
        for i, char in enumerate(label):
            if char == 'X':
                x[-i - 1] = True
            elif char == 'Z':
                z[-i - 1] = True
            elif char == 'Y':
                z[-i - 1] = True
                x[-i - 1] = True
            elif char != 'I':
                raise QiskitError("Pauli string must be only consisted of 'I', 'X', "
                                  "'Y' or 'Z' but you have {}.".format(char))
        return cls(z=z, x=x)

    def _init_from_bool(self, z, x):
        """Construct pauli from boolean array.

        Args:
            z (numpy.ndarray): boolean, z vector
            x (numpy.ndarray): boolean, x vector

        Returns:
            Pauli: self

        Raises:
            QiskitError: if z or x are None or the length of z and x are different.
        """
        if z is None:
            raise QiskitError("z vector must not be None.")
        if x is None:
            raise QiskitError("x vector must not be None.")
        if len(z) != len(x):
            raise QiskitError("length of z and x vectors must be "
                              "the same. (z: {} vs x: {})".format(len(z), len(x)))

        z = _make_np_bool(z)
        x = _make_np_bool(x)
        self._z = z
        self._x = x

        return self

    def __len__(self):
        """Return number of qubits."""
        return len(self._z)

    def __repr__(self):
        """Return the representation of self."""
        z = list(self._z)
        x = list(self._x)

        ret = self.__class__.__name__ + "(z={}, x={})".format(z, x)
        return ret

    def __str__(self):
        """Output the Pauli label."""
        label = ''
        for z, x in zip(self._z[::-1], self._x[::-1]):
            if not z and not x:
                label = ''.join([label, 'I'])
            elif not z and x:
                label = ''.join([label, 'X'])
            elif z and not x:
                label = ''.join([label, 'Z'])
            else:
                label = ''.join([label, 'Y'])
        return label

    def __eq__(self, other):
        """Return True if all Pauli terms are equal.

        Args:
            other (Pauli): other pauli

        Returns:
            bool: are self and other equal.
        """
        res = False
        if len(self) == len(other):
            if np.all(self._z == other.z) and np.all(self._x == other.x):
                res = True
        return res

    def __mul__(self, other):
        """Multiply two Paulis.

        Returns:
            Pauli: the multiplied pauli.

        Raises:
            QiskitError: if the number of qubits of two paulis are different.
        """
        if len(self) != len(other):
            raise QiskitError("These Paulis cannot be multiplied - different "
                              "number of qubits. ({} vs {})".format(len(self), len(other)))
        z_new = np.logical_xor(self._z, other.z)
        x_new = np.logical_xor(self._x, other.x)
        return Pauli(z_new, x_new)

    def __imul__(self, other):
        """Multiply two Paulis.

        Returns:
            Pauli: the multiplied pauli and save to itself, in-place computation.

        Raises:
            QiskitError: if the number of qubits of two paulis are different.
        """
        if len(self) != len(other):
            raise QiskitError("These Paulis cannot be multiplied - different "
                              "number of qubits. ({} vs {})".format(len(self), len(other)))
        self._z = np.logical_xor(self._z, other.z)
        self._x = np.logical_xor(self._x, other.x)
        return self

    def __hash__(self):
        """Make object is hashable, based on the pauli label to hash."""
        return hash(str(self))

    @property
    def z(self):
        """Getter of z."""
        return self._z

    @property
    def x(self):
        """Getter of x."""
        return self._x

    @staticmethod
    def sgn_prod(p1, p2):
        r"""
        Multiply two Paulis and track the phase.

        $P_3 = P_1 \otimes P_2$: X*Y

        Args:
            p1 (Pauli): pauli 1
            p2 (Pauli): pauli 2

        Returns:
            Pauli: the multiplied pauli
            complex: the sign of the multiplication, 1, -1, 1j or -1j
        """
        phase = Pauli._prod_phase(p1, p2)
        new_pauli = p1 * p2
        return new_pauli, phase

    @property
    def num_qubits(self):
        """Number of qubits."""
        return len(self)

    @property
    def numberofqubits(self):
        """Deprecated, use ``num_qubits`` instead. Number of qubits."""
        warnings.warn('The Pauli.numberofqubits method is deprecated as of 0.13.0, and '
                      'will be removed no earlier than 3 months after that release date. '
                      'You should use the Pauli.num_qubits method instead.',
                      DeprecationWarning, stacklevel=2)
        return self.num_qubits

    def to_label(self):
        """Present the pauli labels in I, X, Y, Z format.

        Order is $q_{n-1} .... q_0$

        Returns:
            str: pauli label
        """
        return str(self)

    def to_matrix(self):
        r"""
        Convert Pauli to a matrix representation.

        Order is q_{n-1} .... q_0, i.e., $P_{n-1} \otimes ... P_0$

        Returns:
            numpy.array: a matrix that represents the pauli.
        """
        mat = self.to_spmatrix()
        return mat.toarray()

    def to_spmatrix(self):
        r"""
        Convert Pauli to a sparse matrix representation (CSR format).

        Order is q_{n-1} .... q_0, i.e., $P_{n-1} \otimes ... P_0$

        Returns:
            scipy.sparse.csr_matrix: a sparse matrix with CSR format that
            represents the pauli.
        """
        _x, _z = self._x, self._z
        n = 2**len(_x)
        twos_array = 1 << np.arange(len(_x))
        xs = np.array(_x).dot(twos_array)
        zs = np.array(_z).dot(twos_array)
        rows = np.arange(n+1, dtype=np.uint)
        columns = rows ^ xs
        global_factor = (-1j)**np.dot(np.array(_x, dtype=np.uint), _z)
        data = global_factor*(-1)**np.mod(_count_set_bits(zs & rows), 2)
        return sparse.csr_matrix((data, columns, rows), shape=(n, n))

    def to_operator(self):
        """Convert to Operator object."""
        # Place import here to avoid cyclic import from circuit visualization
        from qiskit.quantum_info.operators.operator import Operator
        return Operator(self.to_matrix())

    def to_instruction(self):
        """Convert to Pauli circuit instruction."""
        from qiskit.circuit import QuantumCircuit, QuantumRegister
        from qiskit.circuit.library.standard_gates import IGate, XGate, YGate, ZGate
        gates = {'I': IGate(), 'X': XGate(), 'Y': YGate(), 'Z': ZGate()}
        label = self.to_label()
        num_qubits = self.num_qubits
        qreg = QuantumRegister(num_qubits)
        circuit = QuantumCircuit(qreg, name='Pauli:{}'.format(label))
        for i, pauli in enumerate(reversed(label)):
            circuit.append(gates[pauli], [qreg[i]])
        return circuit.to_instruction()

    def update_z(self, z, indices=None):
        """
        Update partial or entire z.

        Args:
            z (numpy.ndarray or list): to-be-updated z
            indices (numpy.ndarray or list or optional): to-be-updated qubit indices

        Returns:
            Pauli: self

        Raises:
            QiskitError: when updating whole z, the number of qubits must be the same.
        """
        z = _make_np_bool(z)
        if indices is None:
            if len(self._z) != len(z):
                raise QiskitError("During updating whole z, you can not "
                                  "change the number of qubits.")
            self._z = z
        else:
            if not isinstance(indices, list) and not isinstance(indices, np.ndarray):
                indices = [indices]
            for p, idx in enumerate(indices):
                self._z[idx] = z[p]

        return self

    def update_x(self, x, indices=None):
        """
        Update partial or entire x.

        Args:
            x (numpy.ndarray or list): to-be-updated x
            indices (numpy.ndarray or list or optional): to-be-updated qubit indices

        Returns:
            Pauli: self

        Raises:
            QiskitError: when updating whole x, the number of qubits must be the same.
        """
        x = _make_np_bool(x)
        if indices is None:
            if len(self._x) != len(x):
                raise QiskitError("During updating whole x, you can not change "
                                  "the number of qubits.")
            self._x = x
        else:
            if not isinstance(indices, list) and not isinstance(indices, np.ndarray):
                indices = [indices]
            for p, idx in enumerate(indices):
                self._x[idx] = x[p]

        return self

    def insert_paulis(self, indices=None, paulis=None, pauli_labels=None):
        """
        Insert or append pauli to the targeted indices.

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
            paulis = Pauli.from_label(pauli_labels[::-1])

        if indices is None:  # append
            self._z = np.concatenate((self._z, paulis.z))
            self._x = np.concatenate((self._x, paulis.x))
        else:
            if not isinstance(indices, list):
                indices = [indices]
            self._z = np.insert(self._z, indices, paulis.z)
            self._x = np.insert(self._x, indices, paulis.x)

        return self

    def append_paulis(self, paulis=None, pauli_labels=None):
        """
        Append pauli at the end.

        Args:
            paulis (Pauli): the to-be-inserted or appended pauli
            pauli_labels (list[str]): the to-be-inserted or appended pauli label

        Returns:
            Pauli: self
        """
        return self.insert_paulis(None, paulis=paulis, pauli_labels=pauli_labels)

    def delete_qubits(self, indices):
        """
        Delete pauli at the indices.

        Args:
            indices(list[int]): the indices of to-be-deleted paulis

        Returns:
            Pauli: self
        """
        if not isinstance(indices, list):
            indices = [indices]

        self._z = np.delete(self._z, indices)
        self._x = np.delete(self._x, indices)

        return self

    @classmethod
    def random(cls, num_qubits, seed=None):
        """Return a random Pauli on number of qubits.

        Args:
            num_qubits (int): the number of qubits
            seed (int): Optional. To set a random seed.
        Returns:
            Pauli: the random pauli
        """
        rng = np.random.default_rng(seed)
        z = rng.integers(2, size=num_qubits).astype(np.bool)
        x = rng.integers(2, size=num_qubits).astype(np.bool)
        return cls(z, x)

    @classmethod
    def pauli_single(cls, num_qubits, index, pauli_label):
        """
        Generate single qubit pauli at index with pauli_label with length num_qubits.

        Args:
            num_qubits (int): the length of pauli
            index (int): the qubit index to insert the single qubit
            pauli_label (str): pauli

        Returns:
            Pauli: single qubit pauli
        """
        tmp = Pauli.from_label(pauli_label)
        z = np.zeros(num_qubits, dtype=np.bool)
        x = np.zeros(num_qubits, dtype=np.bool)

        z[index] = tmp.z[0]
        x[index] = tmp.x[0]

        return cls(z, x)

    def kron(self, other):
        r"""Kronecker product of two paulis.

        Order is $P_2 (other) \otimes P_1 (self)$

        Args:
            other (Pauli): P2

        Returns:
            Pauli: self
        """
        self.insert_paulis(indices=None, paulis=other)
        return self

    @staticmethod
    def _prod_phase(p1, p2):
        phase_changes = 0
        for z1, x1, z2, x2 in zip(p1.z, p1.x, p2.z, p2.x):
            if z1 and not x1:  # Z
                if x2:
                    phase_changes = phase_changes - 1 if z2 else phase_changes + 1
            elif not z1 and x1:  # X
                if z2:
                    phase_changes = phase_changes + 1 if x2 else phase_changes - 1
            elif z1 and x1:  # Y
                if not z2 and x2:  # X
                    phase_changes -= 1
                elif z2 and not x2:  # Z
                    phase_changes += 1
        phase = (1j) ** (phase_changes % 4)

        return phase


def pauli_group(number_of_qubits, case='weight'):
    """Return the Pauli group with 4^n elements.

    The phases have been removed.
    case 'weight' is ordered by Pauli weights and
    case 'tensor' is ordered by I,X,Y,Z counting lowest qubit fastest.

    Args:
        number_of_qubits (int): number of qubits
        case (str): determines ordering of group elements ('weight' or 'tensor')

    Returns:
        list: list of Pauli objects

    Raises:
        QiskitError: case is not 'weight' or 'tensor'
        QiskitError: number_of_qubits is larger than 4
    """
    if number_of_qubits < 5:
        temp_set = []

        if case == 'weight':
            tmp = pauli_group(number_of_qubits, case='tensor')
            # sort on the weight of the Pauli operator
            return sorted(tmp, key=lambda x: -np.count_nonzero(
                np.array(x.to_label(), 'c') == b'I'))
        elif case == 'tensor':
            # the Pauli set is in tensor order II IX IY IZ XI ...
            for k in range(4 ** number_of_qubits):
                z = np.zeros(number_of_qubits, dtype=np.bool)
                x = np.zeros(number_of_qubits, dtype=np.bool)
                # looping over all the qubits
                for j in range(number_of_qubits):
                    # making the Pauli for each j fill it in from the
                    # end first
                    element = (k // (4 ** j)) % 4
                    if element == 1:
                        x[j] = True
                    elif element == 2:
                        z[j] = True
                        x[j] = True
                    elif element == 3:
                        z[j] = True
                temp_set.append(Pauli(z, x))
            return temp_set
        else:
            raise QiskitError("Only support 'weight' or 'tensor' cases "
                              "but you have {}.".format(case))

    raise QiskitError("Only support number of qubits is less than 5")
