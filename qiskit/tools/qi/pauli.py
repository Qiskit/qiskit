# -*- coding: utf-8 -*-

# Copyright 2017, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name

"""
Tools for working with Pauli Operators.

A simple pauli class and some tools.
"""
import numpy as np
from scipy import sparse

from qiskit import QISKitError


class Pauli:
    """A simple class representing Pauli Operators.

    The form is P = (-i)^dot(v,w) Z^v X^w where v and w are elements of Z_2^n.
    That is, there are 4^n elements (no phases in this group).

    For example, for 1 qubit
    P_00 = Z^0 X^0 = I
    P_01 = X
    P_10 = Z
    P_11 = -iZX = (-i) iY = Y

    Multiplication is P1*P2 = (-i)^dot(v1+v2,w1+w2) Z^(v1+v2) X^(w1+w2)
    where the sums are taken modulo 2.

    Pauli vectors v and w are supposed to be defined as numpy arrays.

    Ref.
    Jeroen Dehaene and Bart De Moor
    Clifford group, stabilizer states, and linear and quadratic operations
    over GF(2)
    Phys. Rev. A 68, 042318 – Published 20 October 2003
    """

    def __init__(self, v, w):
        """Make the Pauli object."""
        if isinstance(v, list) and isinstance(w, list):
            v = np.asarray(v).astype(np.bool)
            w = np.asarray(w).astype(np.bool)

        if v.dtype != np.bool or w.dtype != np.bool:
            v = v.astype(np.bool)
            w = w.astype(np.bool)

        self._v = v
        self._w = w

    @classmethod
    def from_label(cls, labels):
        """Take pauli string to construct pauli."""
        v = np.zeros(len(labels), dtype=np.bool)
        w = np.zeros(len(labels), dtype=np.bool)
        for i, label in enumerate(labels):
            if label == 'X':
                w[i] = True
            elif label == 'Z':
                v[i] = True
            elif label == 'Y':
                v[i] = True
                w[i] = True
            elif label != 'I':
                raise QISKitError("Pauli string must be only consisted of 'I', 'X', "
                                  "'Y' or 'Z' but you have {}.".format(label))
        return cls(v, w)

    def __len__(self):
        """Return number of qubits."""
        return len(self._v)

    def __repr__(self):
        """Return the representation of self."""
        v = [x for x in self._v.astype(np.int)]
        w = [x for x in self._w.astype(np.int)]

        ret = "Pauli(v={}, w={})".format(v, w)
        return ret

    def __str__(self):
        """Output the Pauli label."""
        label = ''
        for v, w in zip(self._v, self._w):
            if not v and not w:
                label += 'I'
            elif not v and w:
                label += 'X'
            elif v and not w:
                label += 'Z'
            else:
                label += 'Y'

        return label

    def __eq__(self, other):
        """Return True if all Pauli terms are equal."""
        res = False
        if len(self) == len(other):
            if np.all(self._v == other.v) and np.all(self._w == other.w):
                res = True
        return res

    def __ne__(self, other):
        """Return True if all Pauli terms are not equal."""
        return not self.__eq__(other)

    def __mul__(self, other):
        """Multiply two Paulis."""
        if len(self) != len(other):
            raise QISKitError('These Paulis cannot be multiplied - different number '
                              'of qubits')
        v_new = np.logical_xor(self._v, other.v)
        w_new = np.logical_xor(self._w, other.w)
        return Pauli(v_new, w_new)

    @property
    def v(self):
        """Getter of v."""
        return self._v

    @property
    def w(self):
        """Getter of w."""
        return self._w

    @staticmethod
    def sgn_prod(p1, p2):
        """
        Multiply two Paulis p1*p2 and track the sign.

        p3 = p1*p2: X*Y

        Args:
            p1 (Pauli): pauli 1
            p2 (Pauli): pauli 2

        Returns:
            Pauli: the multiplied pauli
            phase: the sign of the multiplication
        """
        new_pauli = p1 * p2
        phase_changes = 0
        for v1, w1, v2, w2 in zip(p1._v, p1._w, p2._v, p2._w):
            if v1 and not w1:  # Z
                if w2:
                    phase_changes = phase_changes - 1 if v2 else phase_changes + 1
            elif not v1 and w1:  # X
                if v2:
                    phase_changes = phase_changes + 1 if w2 else phase_changes - 1
            elif v1 and w1:  # Y
                if not v2 and w2:  # X
                    phase_changes -= 1
                elif v2 and not w2:  # Z
                    phase_changes += 1
        phase = (1j) ** (phase_changes % 4)
        return new_pauli, phase

    @property
    def numberofqubits(self):
        """Number of qubits."""
        return len(self)

    def to_label(self):
        # TODO: the order of qubits.
        """Print out the labels in X, Y, Z format.

        Returns:
            str: pauli label
        """
        return str(self)

    def to_matrix(self):
        """
        Convert Pauli to a matrix representation.

        Order is q_n x q_{n-1} .... q_0

        Returns:
            numpy.array: a matrix that represents the pauli.
        """
        mat = self.to_spmatrix
        return mat.toarray()

    def to_spmatrix(self):
        # TODO: the order of qubits.
        """
        Convert Pauli to a sparse matrix representation (CSR format).

        Order is q_n x q_{n-1} .... q_0

        Returns:
            scipy.sparse.csr_matrix: a sparse matrix with CSR format that
            represnets the pauli.
        """
        mat = sparse.coo_matrix(1)
        for v, w in zip(self._v, self._w):
            if not v and not w:  # I
                mat = sparse.bmat([[mat, None], [None, mat]], format='coo')
            elif v and not w:  # Z
                mat = sparse.bmat([[mat, None], [None, -mat]], format='coo')
            elif not v and w:  # X
                mat = sparse.bmat([[None, mat], [mat, None]], format='coo')
            else:  # Y
                mat = mat * 1j
                mat = sparse.bmat([[None, -mat], [mat, None]], format='coo')

        return mat.tocsr()

    def update_v(self, v, pos=None):
        # TODO: the order of qubits.
        """
        Update partial of entire v.

        Args:
            v (numpy.ndarray): to-be-updated v.
            pos (numpy.ndarray or list or None): to-be-updated position
        """
        if pos is None:
            self._v = v
        else:
            if not isinstance(pos, list) or not isinstance(pos, np.ndarray):
                pos = [pos]
            for p, idx in enumerate(pos):
                self._v[idx] = v[p]

        return self

    def update_w(self, w, pos=None):
        # TODO: the order of qubits.
        """
        Update partial of entire w.

        Args:
            w (numpy.ndarray): to-be-updated w.
            pos (numpy.ndarray or list or None): to-be-updated position
        """
        if pos is None:
            self._w = w
        else:
            if not isinstance(pos, list) or not isinstance(pos, np.ndarray):
                pos = [pos]
            for p, idx in enumerate(pos):
                self._w[idx] = w[p]

        return self

    def insert_qubits(self, pos, pauli_labels):
        # TODO: the order of qubits.
        """
        Insert pauli to the targeted positions.

        Args:
            pos ([int]): the position to be inserted.
            paulis_label([str]): to-be-inserted pauli

        Note:
            the pos refers to the localion of original paulis,
            e.g. if pos = [0, 2], pauli_labels = ['Z', 'I'] and original pauli = 'IXYZ'
            the pauli will be updated to 'Z'IX'I'YZ.
            'Z' and 'I' are inserted before the index at 0 and 2.
        """
        if not isinstance(pos, list):
            pos = [pos]

        if not isinstance(pauli_labels, list):
            pauli_labels = [pauli_labels]

        tmp = Pauli.from_pauli_string(pauli_labels)

        new_v = self._v
        new_w = self._w
        self._v = np.insert(new_v, pos, tmp.v)
        self._w = np.insert(new_w, pos, tmp.w)

        return self

    def append_qubits(self, pauli_labels):
        # TODO: the order of qubits.
        """
        Append pauli to the end.

        Args:
            paulis_labels([str]): to-be-inserted pauli
        """
        if not isinstance(pauli_labels, list):
            pauli_labels = [pauli_labels]

        tmp = Pauli.from_pauli_string(pauli_labels)

        self._v = np.concatenate((self._v, tmp.v))
        self._w = np.concatenate((self._w, tmp.w))

        return self

    def delete_qubits(self, pos):
        # TODO: the order of qubits.
        """
        Delete pauli at the position.

        Args:
            pos([int]): the positions of to-be-deleted paulis.
        """
        if not isinstance(pos, list):
            pos = [pos]

        self._v = np.delete(self._v, pos)
        self._w = np.delete(self._w, pos)

        return self

    @staticmethod
    def generate_random_pauli(num_qubits):
        """Return a random Pauli on numberofqubits.

        Args:
            num_qubits (int): the number of qubits.

        Returns:
            Pauli: the random pauli
        """
        v = np.random.randint(2, size=num_qubits).astype(np.bool)
        w = np.random.randint(2, size=num_qubits).astype(np.bool)
        return Pauli(v, w)

    @staticmethod
    def generate_single_qubit_pauli(pos, pauli_label, num_qubits):
        # TODO: the order of qubits.
        """
        Generate single qubit pauli at pos with pauli_label with length num_qubits.

        Args:
            pos (int): the position to insert the single qubii
            pauli_label (str): pauli
            num_qubits (int): the length of pauli

        """
        tmp = Pauli.from_pauli_string(pauli_label)
        v = np.zeros(num_qubits, dtype=np.bool)
        w = np.zeros(num_qubits, dtype=np.bool)

        v[pos] = tmp._v[0]
        w[pos] = tmp._w[0]

        return Pauli(v, w)

    def kron(self, pauli):
        # TODO: the order of qubits.
        """kron product of two paulis"""

        self._v = np.concatenate((self._v, pauli.v))
        self._w = np.concatenate((self._w, pauli.w))

        return self


def pauli_group(number_of_qubits, case='weight'):
    """Return the Pauli group with 4^n elements.

    The phases have been removed.
    case 'weight' is ordered by Pauli weights and
    case 'tensor' is ordered by I,X,Y,Z counting last qubit fastest.

    Args:
        number_of_qubits (int): number of qubits
        case (str): determines ordering of group elements ('weight' or 'tensor')

    Returns:
        list: list of Pauli objects

    Raises:
        QISKitError: case is not 'weight' or 'tensor'
        QISKitError: number_of_qubits is larger than 4
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
                v = np.zeros(number_of_qubits, dtype=np.bool)
                w = np.zeros(number_of_qubits, dtype=np.bool)
                # looping over all the qubits
                for j in range(number_of_qubits):
                    # making the Pauli for each j fill it in from the
                    # end first
                    element = (k // (4 ** j)) % 4
                    if element == 1:
                        w[j] = True
                    elif element == 2:
                        v[j] = True
                        w[j] = True
                    elif element == 3:
                        v[j] = True
                temp_set.append(Pauli(v, w))
            return temp_set
        else:
            raise QISKitError("Only support 'weight' or 'tensor' cases")

    raise QISKitError("Only support number of qubits is less than 5")


def pauli_singles(j_index, number_qubits):
    """Return the single qubit pauli in number_qubits."""
    # looping over all the qubits
    tempset = []
    v = np.zeros(number_qubits, dtype=np.bool)
    w = np.zeros(number_qubits, dtype=np.bool)
    w[j_index] = True
    tempset.append(Pauli(v, w))
    v = np.zeros(number_qubits, dtype=np.bool)
    w = np.zeros(number_qubits, dtype=np.bool)
    v[j_index] = True
    w[j_index] = True
    tempset.append(Pauli(v, w))
    v = np.zeros(number_qubits, dtype=np.bool)
    w = np.zeros(number_qubits, dtype=np.bool)
    v[j_index] = True
    tempset.append(Pauli(v, w))
    return tempset


if __name__ == '__main__':
    p = Pauli.from_label('IIII')
    # p.insert_qubits([0, 2], ['Z', 'X'])
    # p.append_qubits(list('ZX'))
    print(p)
    print(repr(p))
    a = np.zeros(4)
    b = np.ones(4)
    c = Pauli(a, b)
    print(c._v.dtype)
