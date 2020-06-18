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
Symplectic Pauli Table Class
"""
# pylint: disable=invalid-name

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.pauli import Pauli
from qiskit.quantum_info.operators.custom_iterator import CustomIterator


class PauliTable(BaseOperator):
    r"""Symplectic representation of a list Pauli matrices.

    **Symplectic Representation**

    The symplectic representation of a single-qubit Pauli matrix
    is a pair of boolean values :math:`[x, z]` such that the Pauli matrix
    is given by :math:`P = (-i)^{z * x} \sigma_z^z.\sigma_x^x`.
    The correspondence between labels, symplectic representation,
    and matrices for single-qubit Paulis are shown in Table 1.

    .. list-table:: Pauli Representations
        :header-rows: 1

        * - Label
          - Symplectic
          - Matrix
        * - ``"I"``
          - :math:`[0, 0]`
          - :math:`\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`
        * - ``"X"``
          - :math:`[1, 0]`
          - :math:`\begin{bmatrix} 0 & 1 \\ 1 & 0  \end{bmatrix}`
        * - ``"Y"``
          - :math:`[1, 1]`
          - :math:`\begin{bmatrix} 0 & -i \\ i & 0  \end{bmatrix}`
        * - ``"Z"``
          - :math:`[0, 1]`
          - :math:`\begin{bmatrix} 1 & 0 \\ 0 & -1  \end{bmatrix}`

    The full Pauli table is a M x 2N boolean matrix:

    .. math::

        \left(\begin{array}{ccc|ccc}
            x_{0,0} & ... & x_{0,N-1} & z_{0,0} & ... & z_{0,N-1}  \\
            x_{1,0} & ... & x_{1,N-1} & z_{1,0} & ... & z_{1,N-1}  \\
            \vdots & \ddots & \vdots & \vdots & \ddots & \vdots  \\
            x_{M-1,0} & ... & x_{M-1,N-1} & z_{M-1,0} & ... & z_{M-1,N-1}
        \end{array}\right)

    where each row is a block vector :math:`[X_i, Z_i]` with
    :math:`X = [x_{i,0}, ..., x_{i,N-1}]`, :math:`Z = [z_{i,0}, ..., z_{i,N-1}]`
    is the symplectic representation of an `N`-qubit Pauli.
    This representation is based on reference [1].

    PauliTable's can be created from a list of labels using :meth:`from_labels`,
    and converted to a list of labels or a list of matrices using
    :meth:`to_labels` and :meth:`to_matrix` respectively.

    **Group Product**

    The Pauli's in the Pauli table do not represent the full Pauli as they are
    restricted to having `+1` phase. The dot-product for the Pauli's is defined
    to discard any phase obtained from matrix multiplication so that we have
    :math:`X.Z = Z.X = Y`, etc. This means that for the PauliTable class the
    operator methods :meth:`compose` and :meth:`dot` are equivalent.

    +-------+---+---+---+---+
    | A.B   | I | X | Y | Z |
    +=======+===+===+===+===+
    | **I** | I | X | Y | Z |
    +-------+---+---+---+---+
    | **X** | X | I | Z | Y |
    +-------+---+---+---+---+
    | **Y** | Y | Z | I | X |
    +-------+---+---+---+---+
    | **Z** | Z | Y | X | I |
    +-------+---+---+---+---+

    **Qubit Ordering**

    The qubits are ordered in the table such the least significant qubit
    `[x_{i, 0}, z_{i, 0}]` is the first element of each of the :math:`X_i, Z_i`
    vector blocks. This is the opposite order to position in string labels or
    matrix tensor products where the least significant qubit is the right-most
    string character. For example Pauli ``"ZX"`` has ``"X"`` on qubit-0
    and ``"Z"`` on qubit 1, and would have symplectic vectors :math:`x=[1, 0]`,
    :math:`z=[0, 1]`.

    **Data Access**

    Subsets of rows can be accessed using the list access ``[]`` operator and
    will return a table view of part of the PauliTable. The underlying Numpy
    array can be directly accessed using the :attr:`array` property, and the
    sub-arrays for only the `X` or `Z` blocks can be accessed using the
    :attr:`X` and :attr:`Z` properties respectively.

    **Iteration**

    Rows in the Pauli table can be iterated over like a list. Iteration can
    also be done using the label or matrix representation of each row using the
    :meth:`label_iter` and :meth:`matrix_iter` methods.

    References:
        1. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """

    def __init__(self, data):
        """Initialize the PauliTable.

        Args:
            data (array or str or ScalarOp or PauliTable): input data.

        Raises:
            QiskitError: if input array is invalid shape.

        Additional Information:
            The input array is not copied so multiple Pauli tables
            can share the same underlying array.
        """
        if isinstance(data, (np.ndarray, list)):
            self._array = np.asarray(data, dtype=np.bool)
        elif isinstance(data, str):
            # If input is a single Pauli string we convert to table
            self._array = PauliTable._from_label(data)
        elif isinstance(data, PauliTable):
            # Share underlying array
            self._array = data._array
        elif isinstance(data, Pauli):
            self._array = np.hstack([data.x, data.z])
        elif isinstance(data, ScalarOp):
            # Initialize an N-qubit identity
            if data.num_qubits is None:
                raise QiskitError(
                    '{} is not an N-qubit identity'.format(data))
            self._array = np.zeros((1, 2 * data.num_qubits), dtype=np.bool)
        else:
            raise QiskitError("Invalid input data for PauliTable.")

        # Input must be a (K, 2*N) shape matrix for M N-qubit Paulis.
        if self._array.ndim == 1:
            self._array = np.reshape(self._array, (1, self._array.size))
        if self._array.ndim != 2 or self._array.shape[1] % 2 != 0:
            raise QiskitError("Invalid shape for PauliTable.")

        # Set size properties
        self._num_paulis = self._array.shape[0]
        dims = (self._array.shape[1] // 2) * (2, )
        super().__init__(dims, dims)

    def __repr__(self):
        """Display representation."""
        prefix = 'PauliTable('
        return '{}{})'.format(prefix, np.array2string(
            self._array, separator=',', prefix=prefix))

    def __str__(self):
        """String representation."""
        return 'PauliTable: {}'.format(self.to_labels())

    def __eq__(self, other):
        """Test if two Pauli tables are equal."""
        if isinstance(other, PauliTable):
            return np.all(self._array == other._array)
        return False

    # ---------------------------------------------------------------------
    # Direct array access
    # ---------------------------------------------------------------------

    @property
    def array(self):
        """The underlying boolean array."""
        return self._array

    @array.setter
    def array(self, value):
        """Set the underlying boolean array."""
        # We use [:, :] array view so that setting the array cannot
        # change the arrays shape.
        self._array[:, :] = value

    @property
    def X(self):
        """The X block of the :attr:`array`."""
        return self._array[:, 0:self._num_qubits]

    @X.setter
    def X(self, val):
        self._array[:, 0:self._num_qubits] = val

    @property
    def Z(self):
        """The Z block of the :attr:`array`."""
        return self._array[:, self._num_qubits:2*self._num_qubits]

    @Z.setter
    def Z(self, val):
        self._array[:, self._num_qubits:2*self._num_qubits] = val

    # ---------------------------------------------------------------------
    # Size Properties
    # ---------------------------------------------------------------------

    @property
    def shape(self):
        """The full shape of the :meth:`array`"""
        return self._array.shape

    @property
    def size(self):
        """The number of Pauli rows in the table."""
        return self._num_paulis

    def __len__(self):
        """Return the number of Pauli rows in the table."""
        return self.size

    # ---------------------------------------------------------------------
    # Pauli Array methods
    # ---------------------------------------------------------------------

    def __getitem__(self, key):
        """Return a view of the PauliTable."""
        # Returns a view of specified rows of the PauliTable
        # This supports all slicing operations the underlying array supports.
        if isinstance(key, (int, np.int)):
            key = [key]
        return PauliTable(self._array[key])

    def __setitem__(self, key, value):
        """Update PauliTable."""
        # Modify specified rows of the PauliTable
        if not isinstance(value, PauliTable):
            value = PauliTable(value)
        self._array[key] = value.array

    def delete(self, ind, qubit=False):
        """Return a copy with Pauli rows deleted from table.

        When deleting qubits the qubit index is the same as the
        column index of the underlying :attr:`X` and :attr:`Z` arrays.

        Args:
            ind (int or list): index(es) to delete.
            qubit (bool): if True delete qubit columns, otherwise delete
                          Pauli rows (Default: False).

        Returns:
            PauliTable: the resulting table with the entries removed.

        Raises:
            QiskitError: if ind is out of bounds for the array size or
                         number of qubits.
        """
        if isinstance(ind, int):
            ind = [ind]

        # Row deletion
        if not qubit:
            if max(ind) >= self.size:
                raise QiskitError("Indices {} are not all less than the size"
                                  " of the PauliTable ({})".format(ind, self.size))
            return PauliTable(np.delete(self._array, ind, axis=0))

        # Column (qubit) deletion
        if max(ind) >= self.num_qubits:
            raise QiskitError("Indices {} are not all less than the number of"
                              " qubits in the PauliTable ({})".format(ind, self.num_qubits))
        cols = ind + [self._num_qubits + i for i in ind]
        return PauliTable(np.delete(self._array, cols, axis=1))

    def insert(self, ind, value, qubit=False):
        """Insert Pauli's into the table.

        When inserting qubits the qubit index is the same as the
        column index of the underlying :attr:`X` and :attr:`Z` arrays.

        Args:
            ind (int): index to insert at.
            value (PauliTable): values to insert.
            qubit (bool): if True delete qubit columns, otherwise delete
                          Pauli rows (Default: False).

        Returns:
            PauliTable: the resulting table with the entries inserted.

        Raises:
            QiskitError: if the insertion index is invalid.
        """
        if not isinstance(ind, int):
            raise QiskitError("Insert index must be an integer.")

        if not isinstance(value, PauliTable):
            value = PauliTable(value)

        # Row insertion
        if not qubit:
            if ind > self.size:
                raise QiskitError("Index {} is larger than the number of rows in the"
                                  " PauliTable ({}).".format(ind, self.num_qubits))
            return PauliTable(np.insert(self.array, ind, value.array, axis=0))

        # Column insertion
        if ind > self.num_qubits:
            raise QiskitError("Index {} is greater than number of qubits"
                              " in the PauliTable ({})".format(ind, self.num_qubits))
        if value.size == 1:
            # Pad blocks to correct size
            value_x = np.vstack(self.size * [value.X])
            value_z = np.vstack(self.size * [value.Z])
        elif value.size == self.size:
            #  Blocks are already correct size
            value_x = value.X
            value_z = value.Z
        else:
            # Blocks are incorrect size
            raise QiskitError("Input PauliTable must have a single row, or"
                              " the same number of rows as the Pauli Table"
                              " ({}).".format(self.size))
        # Build new array by blocks
        return PauliTable(np.hstack((self.X[:, :ind], value_x, self.X[:, ind:],
                                     self.Z[:, :ind], value_z, self.Z[:, ind:])))

    def argsort(self, weight=False):
        """Return indices for sorting the rows of the table.

        The default sort method is lexicographic sorting by qubit number.
        By using the `weight` kwarg the output can additionally be sorted
        by the number of non-identity terms in the Pauli, where the set of
        all Pauli's of a given weight are still ordered lexicographically.

        Args:
            weight (bool): optionally sort by weight if True (Default: False).

        Returns:
            array: the indices for sorting the table.
        """
        # Get order of each Pauli using
        # I => 0, X => 1, Y => 2, Z => 3
        x = self.X
        z = self.Z
        order = 1 * (x & ~z) + 2 * (x & z) + 3 * (~x & z)
        # Optionally get the weight of Pauli
        # This is the number of non identity terms
        if weight:
            weights = np.sum(x | z, axis=1)

        # Sort by order
        # To preserve ordering between successive sorts we
        # are use the 'stable' sort method
        indices = np.arange(self.size)
        for i in range(self.num_qubits):
            sort_inds = order[:, i].argsort(kind='stable')
            order = order[sort_inds]
            indices = indices[sort_inds]
            if weight:
                weights = weights[sort_inds]

        # If using weights we implement a final sort by total number
        # of non-identity Paulis
        if weight:
            indices = indices[weights.argsort(kind='stable')]
        return indices

    def sort(self, weight=False):
        """Sort the rows of the table.

        The default sort method is lexicographic sorting by qubit number.
        By using the `weight` kwarg the output can additionally be sorted
        by the number of non-identity terms in the Pauli, where the set of
        all Pauli's of a given weight are still ordered lexicographically.

        **Example**

        Consider sorting all a random ordering of all 2-qubit Paulis

        .. jupyter-execute::

            from numpy.random import shuffle
            from qiskit.quantum_info.operators import PauliTable

            # 2-qubit labels
            labels = ['II', 'IX', 'IY', 'IZ', 'XI', 'XX', 'XY', 'XZ',
                      'YI', 'YX', 'YY', 'YZ', 'ZI', 'ZX', 'ZY', 'ZZ']
            # Shuffle Labels
            shuffle(labels)
            pt = PauliTable.from_labels(labels)
            print('Initial Ordering')
            print(pt)

            # Lexicographic Ordering
            srt = pt.sort()
            print('Lexicographically sorted')
            print(srt)

            # Weight Ordering
            srt = pt.sort(weight=True)
            print('Weight sorted')
            print(srt)

        Args:
            weight (bool): optionally sort by weight if True (Default: False).

        Returns:
            PauliTable: a sorted copy of the original table.
        """
        return self[self.argsort(weight=weight)]

    def unique(self, return_index=False, return_counts=False):
        """Return unique Paulis from the table.

        **Example**

        .. jupyter-execute::

            from qiskit.quantum_info.operators import PauliTable

            pt = PauliTable.from_labels(['X', 'Y', 'X', 'I', 'I', 'Z', 'X', 'Z'])
            unique = pt.unique()
            print(unique)

        Args:
            return_index (bool): If True, also return the indices that
                                 result in the unique array.
                                 (Default: False)
            return_counts (bool): If True, also return the number of times
                                  each unique item appears in the table.

        Returns:
            PauliTable: unique
                the table of the unique rows.

            unique_indices: np.ndarray, optional
                The indices of the first occurrences of the unique values in
                the original array. Only provided if ``return_index`` is True.\

            unique_counts: np.array, optional
                The number of times each of the unique values comes up in the
                original array. Only provided if ``return_counts`` is True.
        """
        if return_counts:
            _, index, counts = np.unique(self.array, return_index=True,
                                         return_counts=True, axis=0)
        else:
            _, index = np.unique(self.array, return_index=True, axis=0)
        # Sort the index so we return unique rows in the original array order
        sort_inds = index.argsort()
        index = index[sort_inds]
        unique = self[index]
        # Concatinate return tuples
        ret = (unique, )
        if return_index:
            ret += (index, )
        if return_counts:
            ret += (counts[sort_inds], )
        if len(ret) == 1:
            return ret[0]
        return ret

    # ---------------------------------------------------------------------
    # BaseOperator methods
    # ---------------------------------------------------------------------

    def tensor(self, other):
        """Return the tensor output product of two tables.

        This returns the combination of the tensor product of all Paulis
        in the current table with all Pauli's in the other table, with the
        other tables qubits being the least-significant in the returned table.
        This is the opposite tensor order to :meth:`expand`.

        **Example**

        .. jupyter-execute::

            from qiskit.quantum_info.operators import PauliTable

            current = PauliTable.from_labels(['I', 'X'])
            other =  PauliTable.from_labels(['Y', 'Z'])
            print(current.tensor(other))

        Args:
            other (PauliTable): another PauliTable.

        Returns:
            PauliTable: the tensor outer product table.

        Raises:
            QiskitError: if other cannot be converted to a PauliTable.
        """
        if not isinstance(other, PauliTable):
            other = PauliTable(other)
        x1, x2 = self._block_stack(self.X, other.X)
        z1, z2 = self._block_stack(self.Z, other.Z)
        return PauliTable(np.hstack([x2, x1, z2, z1]))

    def expand(self, other):
        """Return the expand output product of two tables.

        This returns the combination of the tensor product of all Paulis
        in the other table with all Pauli's in the current table, with the
        current tables qubits being the least-significant in the returned table.
        This is the opposite tensor order to :meth:`tensor`.

        **Example**

        .. jupyter-execute::

            from qiskit.quantum_info.operators import PauliTable

            current = PauliTable.from_labels(['I', 'X'])
            other =  PauliTable.from_labels(['Y', 'Z'])
            print(current.expand(other))

        Args:
            other (PauliTable): another PauliTable.

        Returns:
            PauliTable: the expand outer product table.

        Raises:
            QiskitError: if other cannot be converted to a PauliTable.
        """
        if not isinstance(other, PauliTable):
            other = PauliTable(other)
        x1, x2 = self._block_stack(self.X, other.X)
        z1, z2 = self._block_stack(self.Z, other.Z)
        return PauliTable(np.hstack([x1, x2, z1, z2]))

    def compose(self, other, qargs=None, front=True):
        """Return the compose output product of two tables.

        This returns the combination of the dot product of all Paulis
        in the current table with all Pauli's in the other table and
        discards the complex phase from the product. Note that for
        PauliTables this method is equivalent to :meth:`dot` and hence
        the ``front`` kwarg does not change the output.

        **Example**

        .. jupyter-execute::

            from qiskit.quantum_info.operators import PauliTable

            current = PauliTable.from_labels(['I', 'X'])
            other =  PauliTable.from_labels(['Y', 'Z'])
            print(current.compose(other))

        Args:
            other (PauliTable): another PauliTable.
            qargs (None or list): qubits to apply dot product on (Default: None).
            front (bool): If True use `dot` composition method [default: False].

        Returns:
            PauliTable: the compose outer product table.

        Raises:
            QiskitError: if other cannot be converted to a PauliTable.
        """
        # pylint: disable=unused-argument
        if qargs is None:
            qargs = getattr(other, 'qargs', None)
        if not isinstance(other, PauliTable):
            other = PauliTable(other)
        if qargs is None and other.num_qubits != self.num_qubits:
            raise QiskitError("other PauliTable must be on the same number of qubits.")
        if qargs and other.num_qubits != len(qargs):
            raise QiskitError("Number of qubits in the other PauliTable does not match qargs.")

        # Stack X and Z blocks for output size
        x1, x2 = self._block_stack(self.X, other.X)
        z1, z2 = self._block_stack(self.Z, other.Z)

        if qargs is not None:
            ret_x, ret_z = x1.copy(), z1.copy()
            x1 = x1[:, qargs]
            z1 = z1[:, qargs]
            ret_x[:, qargs] = x1 ^ x2
            ret_z[:, qargs] = z1 ^ z2
            pauli = np.hstack([ret_x, ret_z])
        else:
            pauli = np.hstack((x1 ^ x2, z1 ^ z2))
        return PauliTable(pauli)

    def dot(self, other, qargs=None):
        """Return the dot output product of two tables.

        This returns the combination of the dot product of all Paulis
        in the current table with all Pauli's in the other table and
        discards the complex phase from the product. Note that for
        PauliTables this method is equivalent to :meth:`compose`.

        **Example**

        .. jupyter-execute::

            from qiskit.quantum_info.operators import PauliTable

            current = PauliTable.from_labels(['I', 'X'])
            other =  PauliTable.from_labels(['Y', 'Z'])
            print(current.dot(other))

        Args:
            other (PauliTable): another PauliTable.
            qargs (None or list): qubits to apply dot product on (Default: None).

        Returns:
            PauliTable: the dot outer product table.

        Raises:
            QiskitError: if other cannot be converted to a PauliTable.
        """
        return self.compose(other, qargs=qargs, front=True)

    def _add(self, other, qargs=None):
        """Append with another PauliTable.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (PauliTable): another table.
            qargs (None or list): optional subsystems to add on
                                  (Default: None)

        Returns:
            PauliTable: the concatinated table self + other.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        if not isinstance(other, PauliTable):
            other = PauliTable(other)

        self._validate_add_dims(other, qargs)

        if qargs is None or (sorted(qargs) == qargs
                             and len(qargs) == self.num_qubits):
            return PauliTable(np.vstack((self._array, other._array)))

        # Pad other with identity and then add
        padded = PauliTable(
            np.zeros((1, 2 * self.num_qubits), dtype=np.bool))
        padded = padded.compose(other, qargs=qargs)
        return PauliTable(np.vstack((self._array, padded._array)))

    def conjugate(self):
        """Not implemented."""
        raise NotImplementedError(
            "{} does not support conjugatge".format(type(self)))

    def transpose(self):
        """Not implemented."""
        raise NotImplementedError(
            "{} does not support transpose".format(type(self)))

    # ---------------------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------------------

    def commutes(self, pauli):
        """Return list of commutation properties for each row with a Pauli.

        The returned vector is the same length as the size of the table and
        contains `True` for rows that commute with the Pauli, and `False`
        for the rows that anti-commute.

        Args:
            pauli (PauliTable): a single Pauli row.

        Returns:
            array: The boolean vector of which rows commute or anti-commute.

        Raises:
            QiskitError: if input is not a single Pauli row.
        """
        if not isinstance(pauli, PauliTable):
            pauli = PauliTable(pauli)
        if pauli.size != 1:
            raise QiskitError("Input is not a single Pauli.")
        return self._commutes(self, pauli)

    def commutes_with_all(self, other):
        """Return indexes of rows that commute other.

        If other is a multi-row Pauli table the returned vector indexes rows
        of the current PauliTable that commute with *all* Pauli's in other.
        If no rows satisfy the condition the returned array will be empty.

        Args:
            other (PauliTable): a single Pauli or multi-row PauliTable.

        Returns:
            array: index array of the commuting rows.
        """
        return self._commutes_with_all(other)

    def anticommutes_with_all(self, other):
        """Return indexes of rows that commute other.

        If other is a multi-row Pauli table the returned vector indexes rows
        of the current PauliTable that anti-commute with *all* Pauli's in other.
        If no rows satisfy the condition the returned array will be empty.

        Args:
            other (PauliTable): a single Pauli or multi-row
                                                PauliTable.

        Returns:
            array: index array of the anti-commuting rows.
        """
        return self._commutes_with_all(other, anti=True)

    def _commutes_with_all(self, other, anti=False):
        """Return row indexes that commute with all rows in another PauliTable.

        Args:
            other (PauliTable): a PauliTable.
            anti (bool): if True return rows that anti-commute, otherwise
                         return rows taht commute (Default: False).

        Returns:
            array: index array of commuting or anti-commuting row.
        """
        if not isinstance(other, PauliTable):
            other = PauliTable(other)
        comms = PauliTable._commutes(self, other[0])
        inds, = np.where(comms == int(not anti))
        for pauli in other[1:]:
            comms = PauliTable._commutes(self[inds], pauli)
            new_inds, = np.where(comms == int(not anti))
            if new_inds.size == 0:
                # No commuting rows
                return new_inds
            inds = inds[new_inds]
        return inds

    @staticmethod
    def _commutes(pauli_table, pauli):
        """Return row indexes of pauli_table that commute with pauli

        Args:
            pauli_table (PauliTable): a multi-row PauliTable.
            pauli (PauliTable): a single-row PauliTable.

        Returns:
            array: boolean vector of which rows commute (True) or
                   anti-commute (False).
        """
        # Find positions where self and pauli are not identities
        non_iden = (pauli_table.X | pauli_table.Z) & (pauli.X | pauli.Z)
        # Multiply array by Pauli, and set entries where inputs
        # where I to I
        tmp = PauliTable(pauli_table.array ^ pauli.array)
        tmp.X = (tmp.X & non_iden)
        tmp.Z = (tmp.Z & non_iden)
        # Find total number of non I pauli's remaining in table
        # if there are an even number the row commutes with the
        # input Pauli, otherwise it anti-commutes
        return np.logical_not(np.sum((tmp.X | tmp.Z), axis=1) % 2)

    @staticmethod
    def _block_stack(array1, array2):
        """Stack two arrays along their first axis."""
        sz1 = len(array1)
        sz2 = len(array2)
        out_shape1 = (sz1 * sz2, ) + array1.shape[1:]
        out_shape2 = (sz1 * sz2, ) + array2.shape[1:]
        if sz2 > 1:
            # Stack blocks for output table
            ret1 = np.reshape(np.stack(sz2 * [array1], axis=1),
                              out_shape1)
        else:
            ret1 = array1
        if sz1 > 1:
            # Stack blocks for output table
            ret2 = np.reshape(np.vstack(sz1 * [array2]), out_shape2)
        else:
            ret2 = array2
        return ret1, ret2

    # ---------------------------------------------------------------------
    # Representation conversions
    # ---------------------------------------------------------------------

    @classmethod
    def from_labels(cls, labels):
        """Construct a PauliTable from a list of Pauli strings.

        Args:
            labels (list): Pauli string label(es).

        Returns:
            PauliTable: the constructed PauliTable.

        Raises:
            QiskitError: If the input list is empty or contains invalid
            Pauli strings.
        """
        n_paulis = len(labels)
        if n_paulis == 0:
            raise QiskitError("Input Pauli list is empty.")
        # Get size from first Pauli
        first = cls._from_label(labels[0])
        array = np.zeros((n_paulis, len(first)), dtype=np.bool)
        array[0] = first
        for i in range(1, n_paulis):
            array[i] = cls._from_label(labels[i])
        return cls(array)

    def to_labels(self, array=False):
        r"""Convert a PauliTable to a list Pauli string labels.

        For large PauliTables converting using the ``array=True``
        kwarg will be more efficient since it allocates memory for
        the full Numpy array of labels in advance.

        .. list-table:: Pauli Representations
            :header-rows: 1

            * - Label
              - Symplectic
              - Matrix
            * - ``"I"``
              - :math:`[0, 0]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`
            * - ``"X"``
              - :math:`[1, 0]`
              - :math:`\begin{bmatrix} 0 & 1 \\ 1 & 0  \end{bmatrix}`
            * - ``"Y"``
              - :math:`[1, 1]`
              - :math:`\begin{bmatrix} 0 & -i \\ i & 0  \end{bmatrix}`
            * - ``"Z"``
              - :math:`[0, 1]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & -1  \end{bmatrix}`

        Args:
            array (bool): return a Numpy array if True, otherwise
                          return a list (Default: False).

        Returns:
            list or array: The rows of the PauliTable in label form.
        """
        ret = np.zeros(self.size, dtype='<U{}'.format(self._num_qubits))
        for i in range(self.size):
            ret[i] = self._to_label(self._array[i])
        if array:
            return ret
        return ret.tolist()

    def to_matrix(self, sparse=False, array=False):
        r"""Convert to a list or array of Pauli matrices.

        For large PauliTables converting using the ``array=True``
        kwarg will be more efficient since it allocates memory a full
        rank-3 Numpy array of matrices in advance.

        .. list-table:: Pauli Representations
            :header-rows: 1

            * - Label
              - Symplectic
              - Matrix
            * - ``"I"``
              - :math:`[0, 0]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`
            * - ``"X"``
              - :math:`[1, 0]`
              - :math:`\begin{bmatrix} 0 & 1 \\ 1 & 0  \end{bmatrix}`
            * - ``"Y"``
              - :math:`[1, 1]`
              - :math:`\begin{bmatrix} 0 & -i \\ i & 0  \end{bmatrix}`
            * - ``"Z"``
              - :math:`[0, 1]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & -1  \end{bmatrix}`

        Args:
            sparse (bool): if True return sparse CSR matrices, otherwise
                           return dense Numpy arrays (Default: False).
            array (bool): return as rank-3 numpy array if True, otherwise
                          return a list of Numpy arrays (Default: False).

        Returns:
            list: A list of dense Pauli matrices if `array=False` and `sparse=False`.
            list: A list of sparse Pauli matrices if `array=False` and `sparse=True`.
            array: A dense rank-3 array of Pauli matrices if `array=True`.
        """
        if not array:
            # We return a list of Numpy array matrices
            return [self._to_matrix(pauli, sparse=sparse) for pauli in self._array]
        # For efficiency we also allow returning a single rank-3
        # array where first index is the Pauli row, and second two
        # indices are the matrix indices
        dim = 2 ** self.num_qubits
        ret = np.zeros((self.size, dim, dim), dtype=np.complex)
        for i in range(self.size):
            ret[i] = self._to_matrix(self._array[i])
        return ret

    @staticmethod
    def _from_label(label):
        """Return the symplectic representation of a Pauli string"""
        if label[0] == '+':
            # We allow +1 phase sign so we can convert back from positive
            # stabilizer strings
            label = label[1:]
        num_qubits = len(label)
        symp = np.zeros(2 * num_qubits, dtype=np.bool)
        xs = symp[0:num_qubits]
        zs = symp[num_qubits:2*num_qubits]
        for i, char in enumerate(label):
            if char not in ['I', 'X', 'Y', 'Z']:
                raise QiskitError("Pauli string contains invalid character:"
                                  " {} not in ['I', 'X', 'Y', 'Z'].".format(char))
            if char in ['X', 'Y']:
                xs[num_qubits - 1 - i] = True
            if char in ['Z', 'Y']:
                zs[num_qubits - 1 - i] = True
        return symp

    @staticmethod
    def _to_label(pauli):
        """Return the Pauli string from symplectic representation."""
        # Cast in symplectic representation
        # This should avoid a copy if the pauli is already a row
        # in the symplectic table
        symp = np.asarray(pauli, dtype=np.bool)
        num_qubits = symp.size // 2
        x = symp[0:num_qubits]
        z = symp[num_qubits:2*num_qubits]
        paulis = np.zeros(num_qubits, dtype='<U1')
        for i in range(num_qubits):
            if not z[i]:
                if not x[i]:
                    paulis[num_qubits - 1 - i] = 'I'
                else:
                    paulis[num_qubits - 1 - i] = 'X'
            elif not x[i]:
                paulis[num_qubits - 1 - i] = 'Z'
            else:
                paulis[num_qubits - 1 - i] = 'Y'
        return str().join(paulis)

    @staticmethod
    def _to_matrix(pauli, sparse=False, real_valued=False):
        """Return the Pauli matrix from symplectic representation.

        Args:
            pauli (array): symplectic Pauli vector.
            sparse (bool): if True return a sparse CSR matrix, otherwise
                           return a dense Numpy array (Default: False).
            real_valued (bool): if True return real Pauli matrices with
                                Y returned as iY (Default: False).
        Returns:
            array: if sparse=False.
            csr_matrix: if sparse=True.
        """

        def count1(i):
            """Count number of set bits in int or array"""
            i = i - ((i >> 1) & 0x55555555)
            i = (i & 0x33333333) + ((i >> 2) & 0x33333333)
            return (((i + (i >> 4) & 0xF0F0F0F) * 0x1010101) & 0xffffffff) >> 24

        symp = np.asarray(pauli, dtype=np.bool)
        num_qubits = symp.size // 2
        x = symp[0:num_qubits]
        z = symp[num_qubits:2*num_qubits]

        dim = 2 ** num_qubits
        twos_array = 1 << np.arange(num_qubits)
        x_indices = np.array(x).dot(twos_array)
        z_indices = np.array(z).dot(twos_array)

        indptr = np.arange(dim + 1, dtype=np.uint)
        indices = indptr ^ x_indices
        data = (-1) ** np.mod(count1(z_indices & indptr), 2)
        if real_valued:
            dtype = float
        else:
            dtype = complex
            data = (-1j) ** np.sum(x & z) * data

        if sparse:
            # Return sparse matrix
            from scipy.sparse import csr_matrix
            return csr_matrix((data, indices, indptr), shape=(dim, dim),
                              dtype=dtype)

        # Build dense matrix using csr format
        mat = np.zeros((dim, dim), dtype=dtype)
        for i in range(dim):
            mat[i][indices[indptr[i]:indptr[i+1]]] = data[indptr[i]:indptr[i+1]]
        return mat

    # ---------------------------------------------------------------------
    # Custom Iterators
    # ---------------------------------------------------------------------

    def label_iter(self):
        """Return a label representation iterator.

        This is a lazy iterator that converts each row into the string
        label only as it is used. To convert the entire table to labels use
        the :meth:`to_labels` method.

        Returns:
            LabelIterator: label iterator object for the PauliTable.
        """
        class LabelIterator(CustomIterator):
            """Label representation iteration and item access."""
            def __repr__(self):
                return "<PauliTable_label_iterator at {}>".format(hex(id(self)))

            def __getitem__(self, key):
                return self.obj._to_label(self.obj.array[key])
        return LabelIterator(self)

    def matrix_iter(self, sparse=False):
        """Return a matrix representation iterator.

        This is a lazy iterator that converts each row into the Pauli matrix
        representation only as it is used. To convert the entire table to
        matrices use the :meth:`to_matrix` method.

        Args:
            sparse (bool): optionally return sparse CSR matrices if True,
                           otherwise return Numpy array matrices
                           (Default: False)

        Returns:
            MatrixIterator: matrix iterator object for the PauliTable.
        """
        class MatrixIterator(CustomIterator):
            """Matrix representation iteration and item access."""
            def __repr__(self):
                return "<PauliTable_matrix_iterator at {}>".format(hex(id(self)))

            def __getitem__(self, key):
                return self.obj._to_matrix(self.obj.array[key], sparse=sparse)
        return MatrixIterator(self)
