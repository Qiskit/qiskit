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

    The form is P_vw = (-i)^dot(v,w) Z^v X^w where v and w are elements of Z_2^n.
    That is, there are 4^n elements (no phases in this group).

    For example, for 1 qubit
    P_00 = Z^0 X^0 = I
    P_01 = X
    P_10 = Z
    P_11 = -iZX = (-i) iY = Y

    Multiplication is P1*P2 = (-i)^dot(v1+v2,w1+w2) Z^(v1+v2) X^(w1+w2)
    where the sums are taken modulo 2.

    Pauli vectors v and w are supposed to be defined as boolean numpy arrays.

    Ref.
    Jeroen Dehaene and Bart De Moor
    Clifford group, stabilizer states, and linear and quadratic operations
    over GF(2)
    Phys. Rev. A 68, 042318 – Published 20 October 2003
    """

    def __init__(self, v=None, w=None, label=None):
        r"""Make the Pauli object.

        Note that, for the qubit index:
            - Order of v, w vectors is q_0 ... q_{n-1},
            - Order of pauli label is q_{n-1} ... q_0

        E.g.,
            - v and w vectors: v = [v_0 ... v_{n-1}], w = [w_0 ... w_{n-1}]
            - a pauli is $P_{n-1} \otimes ... \otimes P_0$

        Args:
            v (numpy.ndarray): boolean, v vector
            w (numpy.ndarray): boolean, w vector
            label (str): pauli label
        """
        if label is not None:
            Pauli.from_label(label)
        else:
            self._init_from_bool(v, w)

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
            QISKitError: invalid character in the label
        """
        v = np.zeros(len(label), dtype=np.bool)
        w = np.zeros(len(label), dtype=np.bool)
        for i, char in enumerate(label):
            if char == 'X':
                w[-i - 1] = True
            elif char == 'Z':
                v[-i - 1] = True
            elif char == 'Y':
                v[-i - 1] = True
                w[-i - 1] = True
            elif char != 'I':
                raise QISKitError("Pauli string must be only consisted of 'I', 'X', "
                                  "'Y' or 'Z' but you have {}.".format(char))
        return cls(v=v, w=w)

    def _init_from_bool(self, v, w):
        """Construct pauli from boolean array.

        Args:
            v (numpy.ndarray): boolean, v vector
            w (numpy.ndarray): boolean, w vector
        """
        if v is None:
            raise QISKitError("v vector must not be None.")
        if w is None:
            raise QISKitError("w vector must not be None.")
        if len(v) != len(w):
            raise QISKitError("length of v and w vectors must be "
                              "the same. (v: {} vs w: {})".format(len(v), len(w)))

        if isinstance(v, list) and isinstance(w, list):
            v = np.asarray(v).astype(np.bool)
            w = np.asarray(w).astype(np.bool)

        if v.dtype != np.bool or w.dtype != np.bool:
            v = v.astype(np.bool)
            w = w.astype(np.bool)

        self._v = v
        self._w = w

        return self

    def __len__(self):
        """Return number of qubits."""
        return len(self._v)

    def __repr__(self):
        """Return the representation of self."""
        v = [x for x in self._v]
        w = [x for x in self._w]

        ret = self.__class__.__name__ + "(v={}, w={})".format(v, w)
        return ret

    def __str__(self):
        """Output the Pauli label."""
        label = ''
        for v, w in zip(self._v[::-1], self._w[::-1]):
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
        """Multiply two Paulis.

        Raises:
            QISKitError: if the number of qubits of two paulis are different.
        """
        if len(self) != len(other):
            raise QISKitError("These Paulis cannot be multiplied - different "
                              "number of qubits. ({} vs {})".format(len(self), len(other)))
        v_new = np.logical_xor(self._v, other.v)
        w_new = np.logical_xor(self._w, other.w)
        return Pauli(v_new, w_new)

    def __hash__(self):
        """Make object is hashable, based on the pauli label to hash."""
        return hash(str(self))

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
        r"""
        Multiply two Paulis and track the sign.

        $P_3 = P_1 \otimes P_2$: X*Y

        Args:
            p1 (Pauli): pauli 1
            p2 (Pauli): pauli 2

        Returns:
            Pauli: the multiplied pauli
            complex: the sign of the multiplication, 1, -1, 1j or -1j
        """
        new_pauli = p1 * p2
        phase_changes = 0
        for v1, w1, v2, w2 in zip(p1.v, p1.w, p2.v, p2.w):
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
        """Present the pauli labels in I, X, Y, Z format.

        Order is $q_{n-1} .... q_0$

        Returns:
            str: pauli label
        """
        return str(self)

    def to_matrix(self):
        """
        Convert Pauli to a matrix representation.

        Order is q_{n-1} .... q_0

        Returns:
            numpy.array: a matrix that represents the pauli.
        """
        mat = self.to_spmatrix()
        return mat.toarray()

    def to_spmatrix(self):
        """
        Convert Pauli to a sparse matrix representation (CSR format).

        Order is q_{n-1} .... q_0

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

    def update_v(self, v, indices=None):
        """
        Update partial of entire v.

        Args:
            v (numpy.ndarray): to-be-updated v.
            indices (numpy.ndarray or list or optional): to-be-updated qubit indices

        Returns:
            Pauli: self

        Raises:
            QISKitError: when updating whole v, the number of qubits must be the same.
        """
        if indices is None:
            if len(self._v) != len(v):
                raise QISKitError("During updating whole v, you can not chagne the number of qubits.")
            self._v = v
        else:
            if not isinstance(indices, list) or not isinstance(indices, np.ndarray):
                indices = [indices]
            for p, idx in enumerate(indices):
                self._v[idx] = v[p]

        return self

    def update_w(self, w, indices=None):
        """
        Update partial of entire w.

        Args:
            w (numpy.ndarray): to-be-updated w.
            indices (numpy.ndarray or list or optional): to-be-updated qubit indices

        Returns:
            Pauli: self

        Raises:
            QISKitError: when updating whole w, the number of qubits must be the same.
        """
        if indices is None:
            if len(self._w) != len(w):
                raise QISKitError("During updating whole w, you can not chagne the number of qubits.")
            self._w = w
        else:
            if not isinstance(indices, list) or not isinstance(indices, np.ndarray):
                indices = [indices]
            for p, idx in enumerate(indices):
                self._w[idx] = w[p]

        return self

    def insert_qubits(self, indices, pauli_labels):
        """
        Insert pauli to the targeted indices.

        Args:
            indices ([int]): the qubit indices to be inserted.
            paulis_label([str]): to-be-inserted pauli

        Note:
            the indices refers to the localion of original paulis,
            e.g. if indices = [0, 2], pauli_labels = ['Z', 'I'] and original pauli = 'ZYXI'
            the pauli will be updated to ZY'I'XI'Z'
            'Z' and 'I' are inserted before the qubit at 0 and 2.
        """
        if not isinstance(indices, list):
            indices = [indices]

        if not isinstance(pauli_labels, list):
            pauli_labels = [pauli_labels]

        tmp = Pauli.from_label(pauli_labels)

        self._v = np.insert(self._v, indices, tmp.v)
        self._w = np.insert(self._w, indices, tmp.w)

        return self

    def append_qubits(self, pauli_labels):
        r"""Append pauli to the higher order of qubit.

        The resulted pauli is $P_{new} \otimes P_{old}$

        Args:
            paulis_labels(str): to-be-inserted pauli

        Returns:
            Pauli: self
        """
        tmp = Pauli.from_label(pauli_labels)
        self.kron(tmp)

        return self

    def delete_qubits(self, indices):
        """
        Delete pauli at the indices.

        Args:
            indices([int]): the indices of to-be-deleted paulis.

        Returns:
            Pauli: self
        """
        if not isinstance(indices, list):
            indices = [indices]

        self._v = np.delete(self._v, indices)
        self._w = np.delete(self._w, indices)

        return self

    @staticmethod
    def generate_random_pauli(num_qubits):
        """Return a random Pauli on number of qubits.

        Args:
            num_qubits (int): the number of qubits.

        Returns:
            Pauli: the random pauli
        """
        v = np.random.randint(2, size=num_qubits).astype(np.bool)
        w = np.random.randint(2, size=num_qubits).astype(np.bool)
        return Pauli(v, w)

    @staticmethod
    def generate_single_qubit_pauli(index, pauli_label, num_qubits):
        """
        Generate single qubit pauli at index with pauli_label with length num_qubits.

        Args:
            index (int): the qubit index to insert the single qubii
            pauli_label (str): pauli
            num_qubits (int): the length of pauli

        Returns:
            Pauli: single qubit pauli
        """
        tmp = Pauli.from_label(pauli_label)
        v = np.zeros(num_qubits, dtype=np.bool)
        w = np.zeros(num_qubits, dtype=np.bool)

        v[index] = tmp.v[0]
        w[index] = tmp.w[0]

        return Pauli(v, w)

    def kron(self, other):
        r"""Kron product of two paulis.

        Order is $P_2 (other) \otimes P_1 (self)$

        Args:
            other (Pauli): P2

        Returns:
            Pauli: self
        """
        self._v = np.concatenate((self._v, other.v))
        self._w = np.concatenate((self._w, other.w))
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
    p = Pauli.from_label('ZIII')
    p.insert_qubits([0, 2], ['Z', 'X'])
    print(p)
    # p.append_qubits(list('ZX'))
    p.append_qubits('ZXY')
    print(p)
    print(repr(p))
    p2 = eval(repr(p))
    a = np.zeros(4)
    b = np.ones(4)
    c = Pauli(a, b)
    print(c._v.dtype)

    p1 = Pauli(v=[0, 0, 0, 1], w=[0, 0, 0, 0])
    print(p1)

    dict_p = {}

    dict_p[p] = 1
    print(dict_p[p1])
