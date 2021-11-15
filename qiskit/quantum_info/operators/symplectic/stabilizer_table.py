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
Symplectic Stabilizer Table Class
"""

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.custom_iterator import CustomIterator
from qiskit.quantum_info.operators.mixins import AdjointMixin, generate_apidocs
from qiskit.quantum_info.operators.scalar_op import ScalarOp
from qiskit.quantum_info.operators.symplectic.pauli import Pauli
from qiskit.quantum_info.operators.symplectic.pauli_table import PauliTable
from qiskit.utils.deprecation import deprecate_function


class StabilizerTable(BaseOperator, AdjointMixin):
    r"""Symplectic representation of a list Stabilizer matrices.

    **Symplectic Representation**

    The symplectic representation of a single-qubit Stabilizer matrix
    is a pair of boolean values :math:`[x, z]` and a boolean phase `p`
    such that the Stabilizer matrix is given by
    :math:`S = (-1)^p \sigma_z^z.\sigma_x^x`.
    The correspondence between labels, symplectic representation,
    stabilizer matrices, and Pauli matrices for the single-qubit case is
    shown in the following table.

    .. list-table:: Table 1: Stabilizer Representations
        :header-rows: 1

        * - Label
          - Phase
          - Symplectic
          - Matrix
          - Pauli
        * - ``"+I"``
          - 0
          - :math:`[0, 0]`
          - :math:`\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`
          - :math:`I`
        * - ``"-I"``
          - 1
          - :math:`[0, 0]`
          - :math:`\begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}`
          - :math:`-I`
        * - ``"X"``
          - 0
          - :math:`[1, 0]`
          - :math:`\begin{bmatrix} 0 & 1 \\ 1 & 0  \end{bmatrix}`
          - :math:`X`
        * - ``"-X"``
          - 1
          - :math:`[1, 0]`
          - :math:`\begin{bmatrix} 0 & -1 \\ -1 & 0  \end{bmatrix}`
          - :math:`-X`
        * - ``"Y"``
          - 0
          - :math:`[1, 1]`
          - :math:`\begin{bmatrix} 0 & 1 \\ -1 & 0  \end{bmatrix}`
          - :math:`iY`
        * - ``"-Y"``
          - 1
          - :math:`[1, 1]`
          - :math:`\begin{bmatrix} 0 & -1 \\ 1 & 0  \end{bmatrix}`
          - :math:`-iY`
        * - ``"Z"``
          - 0
          - :math:`[0, 1]`
          - :math:`\begin{bmatrix} 1 & 0 \\ 0 & -1  \end{bmatrix}`
          - :math:`Z`
        * - ``"-Z"``
          - 1
          - :math:`[0, 1]`
          - :math:`\begin{bmatrix} -1 & 0 \\ 0 & 1  \end{bmatrix}`
          - :math:`-Z`

    Internally this is stored as a length `N` boolean phase vector
    :math:`[p_{N-1}, ..., p_{0}]` and a size :math:`M \times 2N` boolean matrix:

    .. math::

        \left(\begin{array}{ccc|ccc}
            x_{0,0} & ... & x_{0,N-1} & z_{0,0} & ... & z_{0,N-1}  \\
            x_{1,0} & ... & x_{1,N-1} & z_{1,0} & ... & z_{1,N-1}  \\
            \vdots & \ddots & \vdots & \vdots & \ddots & \vdots  \\
            x_{M-1,0} & ... & x_{M-1,N-1} & z_{M-1,0} & ... & z_{M-1,N-1}
        \end{array}\right)

    where each row is a block vector :math:`[X_i, Z_i]` with
    :math:`X_i = [x_{i,0}, ..., x_{i,N-1}]`, :math:`Z_i = [z_{i,0}, ..., z_{i,N-1}]`
    is the symplectic representation of an `N`-qubit Pauli.
    This representation is based on reference [1].

    StabilizerTable's can be created from a list of labels using :meth:`from_labels`,
    and converted to a list of labels or a list of matrices using
    :meth:`to_labels` and :meth:`to_matrix` respectively.

    **Group Product**

    The product of the stabilizer elements is defined with respect to the
    matrix multiplication of the matrices in Table 1. In terms of
    stabilizes labels the dot product group structure is

    +-------+----+----+----+----+
    | A.B   |  I |  X |  Y |  Z |
    +=======+====+====+====+====+
    | **I** |  I |  X |  Y |  Z |
    +-------+----+----+----+----+
    | **X** |  X |  I | -Z |  Y |
    +-------+----+----+----+----+
    | **Y** |  Y |  Z | -I | -X |
    +-------+----+----+----+----+
    | **Z** |  Z | -Y |  X |  I |
    +-------+----+----+----+----+

    The :meth:`dot` method will return the output for
    :code:`row.dot(col) = row.col`, while the :meth:`compose` will return
    :code:`row.compose(col) = col.row` from the above table.

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
    will return a table view of part of the StabilizerTable. The underlying
    phase vector and Pauli array can be directly accessed using the :attr:`phase`
    and :attr:`array` properties respectively. The sub-arrays for only the
    `X` or `Z` blocks can be accessed using the :attr:`X` and :attr:`Z`
    properties respectively.

    **Iteration**

    Rows in the Stabilizer table can be iterated over like a list. Iteration can
    also be done using the label or matrix representation of each row using the
    :meth:`label_iter` and :meth:`matrix_iter` methods.

    References:
        1. S. Aaronson, D. Gottesman, *Improved Simulation of Stabilizer Circuits*,
           Phys. Rev. A 70, 052328 (2004).
           `arXiv:quant-ph/0406196 <https://arxiv.org/abs/quant-ph/0406196>`_
    """

    def __init__(self, data, phase=None):
        """Initialize the StabilizerTable.

        Args:
            data (array or str or PauliTable or StabilizerTable): input StabilizerTable data.
            phase (array or bool or None): optional phase vector for input data
                                           (Default: None).

        Raises:
            QiskitError: if input array or phase vector has an invalid shape.

        Additional Information:
            The input array is not copied so multiple Pauli and Stabilizer tables
            can share the same underlying array.
        """
        if isinstance(data, str) and phase is None:
            pauli, phase = StabilizerTable._from_label(data)
        elif isinstance(data, StabilizerTable):
            pauli = data._array
            if phase is None:
                phase = data._phase
        else:
            pauli = data
        # Initialize the Pauli table
        if isinstance(pauli, (np.ndarray, list)):
            self._array = np.asarray(pauli, dtype=bool)
        elif isinstance(pauli, PauliTable):
            # Share underlying array
            self._array = pauli._array
        elif isinstance(pauli, Pauli):
            self._array = np.hstack([pauli.x, pauli.z])
        elif isinstance(pauli, ScalarOp):
            # Initialize an N-qubit identity
            if pauli.num_qubits is None:
                raise QiskitError(f"{pauli} is not an N-qubit identity")
            self._array = np.zeros((1, 2 * pauli.num_qubits), dtype=bool)
        else:
            raise QiskitError("Invalid input pauli for StabilizerTable.")

        # Input must be a (K, 2*N) shape matrix for M N-qubit Paulis.
        if self._array.ndim == 1:
            self._array = np.reshape(self._array, (1, self._array.size))
        if self._array.ndim != 2 or self._array.shape[1] % 2 != 0:
            raise QiskitError("Invalid shape for StabilizerTable.")

        # Set size properties
        self._num_paulis = self._array.shape[0]
        num_qubits = self._array.shape[1] // 2
        super().__init__(num_qubits=num_qubits)

        # Initialize the phase vector
        if phase is None or phase is False:
            self._phase = np.zeros(self.size, dtype=bool)
        elif phase is True:
            self._phase = np.ones(self.size, dtype=bool)
        else:
            self._phase = np.asarray(phase, dtype=bool)
            if self._phase.shape != (self.size,):
                raise QiskitError("Phase vector is incorrect shape.")

    def __repr__(self):
        return f"StabilizerTable(\n{repr(self._array)},\nphase={repr(self._phase)})"

    def __str__(self):
        """String representation"""
        return f"StabilizerTable: {self.to_labels()}"

    def __eq__(self, other):
        """Test if two StabilizerTables are equal"""
        if isinstance(other, StabilizerTable):
            return np.all(self._phase == other._phase) and np.all(self._array == other._array)
        return False

    def copy(self):
        """Return a copy of the StabilizerTable."""
        return StabilizerTable(self._array.copy(), self._phase.copy())

    # ---------------------------------------------------------------------
    # Direct array access
    # ---------------------------------------------------------------------

    @property
    @deprecate_function(
        "The StabilizerTable.pauli method is deprecated as of Qiskit Terra 0.19.0 "
        "and will be removed no sooner than 3 months after the releasedate. "
        "Use StabilizerTable.array method instead.",
    )  # pylint: disable=bad-docstring-quotes
    def pauli(self):
        """DEPRECATED - Return PauliTable"""
        return PauliTable(self._array)

    @pauli.setter
    @deprecate_function(
        "The StabilizerTable.pauli method is deprecated as of Qiskit Terra 0.19.0 "
        "and will be removed no sooner than 3 months after the releasedate. "
        "Use StabilizerTable.array method instead.",
    )  # pylint: disable=bad-docstring-quotes
    def pauli(self, value):
        if not isinstance(value, PauliTable):
            value = PauliTable(value)
        self._array[:, :] = value._array

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
    def X(self):  # pylint: disable=invalid-name
        """The X block of the :attr:`array`."""
        return self._array[:, 0 : self.num_qubits]

    @X.setter
    def X(self, val):  # pylint: disable=invalid-name
        self._array[:, 0 : self.num_qubits] = val

    @property
    def Z(self):  # pylint: disable=invalid-name
        """The Z block of the :attr:`array`."""
        return self._array[:, self.num_qubits : 2 * self.num_qubits]

    @Z.setter
    def Z(self, val):  # pylint: disable=invalid-name
        self._array[:, self.num_qubits : 2 * self.num_qubits] = val

    @property
    def phase(self):
        """Return phase vector"""
        return self._phase

    @phase.setter
    def phase(self, value):
        self._phase[:] = value

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
    # Array methods
    # ---------------------------------------------------------------------

    def __getitem__(self, key):
        """Return a view of StabilizerTable"""
        if isinstance(key, (int, np.integer)):
            key = [key]
        return StabilizerTable(self._array[key], self._phase[key])

    def __setitem__(self, key, value):
        """Update StabilizerTable"""
        if not isinstance(value, StabilizerTable):
            value = StabilizerTable(value)
        self._array[key] = value.array
        self._phase[key] = value.phase

    def delete(self, ind, qubit=False):
        """Return a copy with Stabilizer rows deleted from table.

        When deleting qubit columns, qubit-0 is the right-most
        (largest index) column, and qubit-(N-1) is the left-most
        (0 index) column of the underlying :attr:`X` and :attr:`Z`
        arrays.

        Args:
            ind (int or list): index(es) to delete.
            qubit (bool): if True delete qubit columns, otherwise delete
                          Stabilizer rows (Default: False).

        Returns:
            StabilizerTable: the resulting table with the entries removed.

        Raises:
            QiskitError: if ind is out of bounds for the array size or
                         number of qubits.
        """
        if isinstance(ind, (int, np.integer)):
            ind = [ind]

        if qubit:
            # When deleting qubit columns we don't need to modify
            # the phase vector
            if max(ind) >= self.num_qubits:
                raise QiskitError(
                    "Indices {} are not all less than the number of"
                    " qubits in the StabilizerTable ({})".format(ind, self.num_qubits)
                )
            cols = ind + [self.num_qubits + i for i in ind]
            return StabilizerTable(np.delete(self._array, cols, axis=1), self._phase)

        if max(ind) >= self.size:
            raise QiskitError(
                "Indices {} are not all less than the size of the StabilizerTable ({})".format(
                    ind, self.size
                )
            )
        return StabilizerTable(
            np.delete(self._array, ind, axis=0), np.delete(self._phase, ind, axis=0)
        )

    def insert(self, ind, value, qubit=False):
        """Insert stabilizers's into the table.

        When inserting qubit columns, qubit-0 is the right-most
        (largest index) column, and qubit-(N-1) is the left-most
        (0 index) column of the underlying :attr:`X` and :attr:`Z`
        arrays.

        Args:
            ind (int): index to insert at.
            value (StabilizerTable): values to insert.
            qubit (bool): if True delete qubit columns, otherwise delete
                          Pauli rows (Default: False).

        Returns:
            StabilizerTable: the resulting table with the entries inserted.

        Raises:
            QiskitError: if the insertion index is invalid.
        """
        if not isinstance(ind, (int, np.integer)):
            raise QiskitError("Insert index must be an integer.")
        if not isinstance(value, StabilizerTable):
            value = StabilizerTable(value)

        # Update StabilizerTable component
        # Row insertion
        if not qubit:
            if ind > self.size:
                raise QiskitError(
                    f"Index {ind} is larger than the number of rows in the "
                    f"StabilizerTable ({self.num_qubits})."
                )
            return StabilizerTable(
                np.insert(self.array, ind, value.array, axis=0),
                np.insert(self.phase, ind, value.phase, axis=0),
            )

        # Column insertion
        if ind > self.num_qubits:
            raise QiskitError(
                f"Index {ind} is greater than number of qubits "
                f"in the StabilizerTable ({self.num_qubits})"
            )
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
            raise QiskitError(
                "Input StabilizerTable must have a single row, or "
                f"the same number of rows as the Pauli Table ({self.size})."
            )
        # Build new array by blocks
        array = np.hstack(
            (
                self.X[:, :ind],
                value_x,
                self.X[:, ind:],
                self.Z[:, :ind],
                value_z,
                self.Z[:, ind:],
            )
        )

        # Update phase vector
        if not qubit:
            phase = np.insert(self._phase, ind, value._phase, axis=0)
        else:
            phase = np.logical_xor(self._phase, value._phase)
        return StabilizerTable(array, phase)

    def argsort(self, weight=False):
        """Return indices for sorting the rows of the StabilizerTable.

        The default sort method is lexicographic sorting of Paulis by
        qubit number. By using the `weight` kwarg the output can additionally
        be sorted by the number of non-identity terms in the Stabilizer,
        where the set of all Pauli's of a given weight are still ordered
        lexicographically.

        This does not sort based on phase values. It will preserve the
        original order of rows with the same Pauli's but different phases.

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
            sort_inds = order[:, i].argsort(kind="stable")
            order = order[sort_inds]
            indices = indices[sort_inds]
            if weight:
                weights = weights[sort_inds]

        # If using weights we implement a final sort by total number
        # of non-identity Paulis
        if weight:
            indices = indices[weights.argsort(kind="stable")]
        return indices

    def sort(self, weight=False):
        """Sort the rows of the table.

        The default sort method is lexicographic sorting by qubit number.
        By using the `weight` kwarg the output can additionally be sorted
        by the number of non-identity terms in the Pauli, where the set of
        all Pauli's of a given weight are still ordered lexicographically.

        This does not sort based on phase values. It will preserve the
        original order of rows with the same Pauli's but different phases.

        Consider sorting all a random ordering of all 2-qubit Paulis

        .. jupyter-execute::

            from numpy.random import shuffle
            from qiskit.quantum_info.operators import StabilizerTable

            # 2-qubit labels
            labels = ['+II', '+IX', '+IY', '+IZ', '+XI', '+XX', '+XY', '+XZ',
                      '+YI', '+YX', '+YY', '+YZ', '+ZI', '+ZX', '+ZY', '+ZZ',
                      '-II', '-IX', '-IY', '-IZ', '-XI', '-XX', '-XY', '-XZ',
                      '-YI', '-YX', '-YY', '-YZ', '-ZI', '-ZX', '-ZY', '-ZZ']
            # Shuffle Labels
            shuffle(labels)
            st = StabilizerTable.from_labels(labels)
            print('Initial Ordering')
            print(st)

            # Lexicographic Ordering
            srt = st.sort()
            print('Lexicographically sorted')
            print(srt)

            # Weight Ordering
            srt = st.sort(weight=True)
            print('Weight sorted')
            print(srt)

        Args:
            weight (bool): optionally sort by weight if True (Default: False).

        Returns:
            StabilizerTable: a sorted copy of the original table.
        """
        return self[self.argsort(weight=weight)]

    def unique(self, return_index=False, return_counts=False):
        """Return unique stabilizers from the table.

        **Example**

        .. jupyter-execute::

            from qiskit.quantum_info.operators import StabilizerTable

            st = StabilizerTable.from_labels(['+X', '+I', '-I', '-X', '+X', '-X', '+I'])
            unique = st.unique()
            print(unique)

        Args:
            return_index (bool): If True, also return the indices that
                                 result in the unique array.
                                 (Default: False)
            return_counts (bool): If True, also return the number of times
                                  each unique item appears in the table.

        Returns:
            StabilizerTable: unique
                the table of the unique rows.

            unique_indices: np.ndarray, optional
                The indices of the first occurrences of the unique values in
                the original array. Only provided if ``return_index`` is True.\

            unique_counts: np.array, optional
                The number of times each of the unique values comes up in the
                original array. Only provided if ``return_counts`` is True.
        """
        # Combine array and phases into single array for sorting
        stack = np.hstack([self._array, self._phase.reshape((self.size, 1))])
        if return_counts:
            _, index, counts = np.unique(stack, return_index=True, return_counts=True, axis=0)
        else:
            _, index = np.unique(stack, return_index=True, axis=0)
        # Sort the index so we return unique rows in the original array order
        sort_inds = index.argsort()
        index = index[sort_inds]
        unique = self[index]
        # Concatenate return tuples
        ret = (unique,)
        if return_index:
            ret += (index,)
        if return_counts:
            ret += (counts[sort_inds],)
        if len(ret) == 1:
            return ret[0]
        return ret

    def conjugate(self):
        """Not implemented."""
        raise NotImplementedError(f"{type(self)} does not support conjugatge")

    def transpose(self):
        """Not implemented."""
        raise NotImplementedError(f"{type(self)} does not support transpose")

    # ---------------------------------------------------------------------
    # Utility methods
    # ---------------------------------------------------------------------

    def tensor(self, other):
        """Return the tensor output product of two tables.

        This returns the combination of the tensor product of all
        stabilizers in the `current` table with all stabilizers in the
        `other` table. The `other` tables qubits will be the
        least-significant in the returned table. This is the opposite
        tensor order to :meth:`tensor`.

        **Example**

        .. jupyter-execute::

            from qiskit.quantum_info.operators import StabilizerTable

            current = StabilizerTable.from_labels(['+I', '-X'])
            other =  StabilizerTable.from_labels(['-Y', '+Z'])
            print(current.tensor(other))

        Args:
            other (StabilizerTable): another StabilizerTable.

        Returns:
            StabilizerTable: the tensor outer product table.

        Raises:
            QiskitError: if other cannot be converted to a StabilizerTable.
        """
        if not isinstance(other, StabilizerTable):
            other = StabilizerTable(other)
        return self._tensor(self, other)

    def expand(self, other):
        """Return the expand output product of two tables.

        This returns the combination of the tensor product of all
        stabilizers in the `other` table with all stabilizers in the
        `current` table. The `current` tables qubits will be the
        least-significant in the returned table. This is the opposite
        tensor order to :meth:`tensor`.

        **Example**

        .. jupyter-execute::

            from qiskit.quantum_info.operators import StabilizerTable

            current = StabilizerTable.from_labels(['+I', '-X'])
            other =  StabilizerTable.from_labels(['-Y', '+Z'])
            print(current.expand(other))

        Args:
            other (StabilizerTable): another StabilizerTable.

        Returns:
            StabilizerTable: the expand outer product table.

        Raises:
            QiskitError: if other cannot be converted to a StabilizerTable.
        """
        if not isinstance(other, StabilizerTable):
            other = StabilizerTable(other)
        return self._tensor(other, self)

    def compose(self, other, qargs=None, front=False):
        """Return the compose output product of two tables.

        This returns the combination of the compose product of all
        stabilizers in the current table with all stabilizers in the
        other table.

        The individual stabilizer compose product is given by

        +----------------------+----+----+----+----+
        | :code:`A.compose(B)` |  I |  X |  Y |  Z |
        +======================+====+====+====+====+
        | **I**                |  I |  X |  Y |  Z |
        +----------------------+----+----+----+----+
        | **X**                |  X |  I |  Z | -Y |
        +----------------------+----+----+----+----+
        | **Y**                |  Y | -Z | -I |  X |
        +----------------------+----+----+----+----+
        | **Z**                |  Z |  Y | -X |  I |
        +----------------------+----+----+----+----+

        If `front=True` the composition will be given by the
        :meth:`dot` method.

        **Example**

        .. jupyter-execute::

            from qiskit.quantum_info.operators import StabilizerTable

            current = StabilizerTable.from_labels(['+I', '-X'])
            other =  StabilizerTable.from_labels(['+X', '-Z'])
            print(current.compose(other))

        Args:
            other (StabilizerTable): another StabilizerTable.
            qargs (None or list): qubits to apply compose product on
                                  (Default: None).
            front (bool): If True use `dot` composition method
                          (default: False).

        Returns:
            StabilizerTable: the compose outer product table.

        Raises:
            QiskitError: if other cannot be converted to a StabilizerTable.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)
        if not isinstance(other, StabilizerTable):
            other = StabilizerTable(other)
        if qargs is None and other.num_qubits != self.num_qubits:
            raise QiskitError("other StabilizerTable must be on the same number of qubits.")
        if qargs and other.num_qubits != len(qargs):
            raise QiskitError("Number of qubits in the other StabilizerTable does not match qargs.")

        # Stack X and Z blocks for output size
        x1, x2 = self._block_stack(self.X, other.X)
        z1, z2 = self._block_stack(self.Z, other.Z)
        phase1, phase2 = self._block_stack(self.phase, other.phase)

        if qargs is not None:
            ret_x, ret_z = x1.copy(), z1.copy()
            x1 = x1[:, qargs]
            z1 = z1[:, qargs]
            ret_x[:, qargs] = x1 ^ x2
            ret_z[:, qargs] = z1 ^ z2
            pauli = np.hstack([ret_x, ret_z])
        else:
            pauli = np.hstack((x1 ^ x2, z1 ^ z2))

        # We pick up a minus sign for products:
        # Y.Y = -I, X.Y = -Z, Y.Z = -X, Z.X = -Y
        if front:
            minus = (x1 & z2 & (x2 | z1)) | (~x1 & x2 & z1 & ~z2)
        else:
            minus = (x2 & z1 & (x1 | z2)) | (~x2 & x1 & z2 & ~z1)
        phase_shift = np.array(np.sum(minus, axis=1) % 2, dtype=bool)
        phase = phase_shift ^ phase1 ^ phase2
        return StabilizerTable(pauli, phase)

    def dot(self, other, qargs=None):
        """Return the dot output product of two tables.

        This returns the combination of the compose product of all
        stabilizers in the current table with all stabilizers in the
        other table.

        The individual stabilizer dot product is given by

        +------------------+----+----+----+----+
        | :code:`A.dot(B)` |  I |  X |  Y |  Z |
        +==================+====+====+====+====+
        | **I**            |  I |  X |  Y |  Z |
        +------------------+----+----+----+----+
        | **X**            |  X |  I | -Z |  Y |
        +------------------+----+----+----+----+
        | **Y**            |  Y |  Z | -I | -X |
        +------------------+----+----+----+----+
        | **Z**            |  Z | -Y |  X |  I |
        +------------------+----+----+----+----+

        **Example**

        .. jupyter-execute::

            from qiskit.quantum_info.operators import StabilizerTable

            current = StabilizerTable.from_labels(['+I', '-X'])
            other =  StabilizerTable.from_labels(['+X', '-Z'])
            print(current.dot(other))

        Args:
            other (StabilizerTable): another StabilizerTable.
            qargs (None or list): qubits to apply dot product on
                                  (Default: None).

        Returns:
            StabilizerTable: the dot outer product table.

        Raises:
            QiskitError: if other cannot be converted to a StabilizerTable.
        """
        return self.compose(other, qargs=qargs, front=True)

    @classmethod
    def _tensor(cls, a, b):
        x1, x2 = a._block_stack(a.X, b.X)
        z1, z2 = a._block_stack(a.Z, b.Z)
        array = np.hstack([x2, x1, z2, z1])
        phase1, phase2 = a._block_stack(a.phase, b.phase)
        phase = np.logical_xor(phase1, phase2)
        return StabilizerTable(array, phase)

    def _add(self, other, qargs=None):
        """Append with another StabilizerTable.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (StabilizerTable): another table.
            qargs (None or list): optional subsystems to add on
                                  (Default: None)

        Returns:
            StabilizerTable: the concatenated table self + other.
        """
        if qargs is None:
            qargs = getattr(other, "qargs", None)

        if not isinstance(other, StabilizerTable):
            other = StabilizerTable(other)

        self._op_shape._validate_add(other._op_shape, qargs)

        if qargs is None or (sorted(qargs) == qargs and len(qargs) == self.num_qubits):
            return StabilizerTable(
                np.vstack((self._array, other._array)), np.hstack((self._phase, other._phase))
            )

        # Pad other with identity and then add
        padded = StabilizerTable(np.zeros((1, 2 * self.num_qubits), dtype=bool))
        padded = padded.compose(other, qargs=qargs)

        return StabilizerTable(
            np.vstack((self._array, padded._array)), np.hstack((self._phase, padded._phase))
        )

    def __add__(self, other):
        qargs = getattr(other, "qargs", None)
        return self._add(other, qargs=qargs)

    def _multiply(self, other):
        """Multiply (XOR) phase vector of the StabilizerTable.

        This updates the phase vector of the table. Allowed values for
        multiplication are ``False``, ``True``, 1 or -1. Multiplying by
        -1 or ``False`` is equivalent. As is multiplying by 1 or ``True``.

        Args:
            other (bool or int): a Boolean value.

        Returns:
           StabilizerTable: the updated stabilizer table.

        Raises:
            QiskitError: if other is not in (False, True, 1, -1).
        """
        # Numeric (integer) value case
        if not isinstance(other, bool) and other not in [1, -1]:
            raise QiskitError("Can only multiply a Stabilizer value by +1 or -1 phase.")

        # We have to be careful we don't cast True <-> +1 when
        # we store -1 phase as boolen True value
        if (isinstance(other, bool) and other) or other == -1:
            ret = self.copy()
            ret._phase ^= True
            return ret
        return self

    @staticmethod
    def _block_stack(array1, array2):
        """Stack two arrays along their first axis."""
        sz1 = len(array1)
        sz2 = len(array2)
        out_shape1 = (sz1 * sz2,) + array1.shape[1:]
        out_shape2 = (sz1 * sz2,) + array2.shape[1:]
        if sz2 > 1:
            # Stack blocks for output table
            ret1 = np.reshape(np.stack(sz2 * [array1], axis=1), out_shape1)
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
        r"""Construct a StabilizerTable from a list of Pauli stabilizer strings.

        Pauli Stabilizer string labels are Pauli strings with an optional
        ``"+"`` or ``"-"`` character. If there is no +/-sign a + phase is
        used by default.

        .. list-table:: Stabilizer Representations
            :header-rows: 1

            * - Label
              - Phase
              - Symplectic
              - Matrix
              - Pauli
            * - ``"+I"``
              - 0
              - :math:`[0, 0]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`
              - :math:`I`
            * - ``"-I"``
              - 1
              - :math:`[0, 0]`
              - :math:`\begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}`
              - :math:`-I`
            * - ``"X"``
              - 0
              - :math:`[1, 0]`
              - :math:`\begin{bmatrix} 0 & 1 \\ 1 & 0  \end{bmatrix}`
              - :math:`X`
            * - ``"-X"``
              - 1
              - :math:`[1, 0]`
              - :math:`\begin{bmatrix} 0 & -1 \\ -1 & 0  \end{bmatrix}`
              - :math:`-X`
            * - ``"Y"``
              - 0
              - :math:`[1, 1]`
              - :math:`\begin{bmatrix} 0 & 1 \\ -1 & 0  \end{bmatrix}`
              - :math:`iY`
            * - ``"-Y"``
              - 1
              - :math:`[1, 1]`
              - :math:`\begin{bmatrix} 0 & -1 \\ 1 & 0  \end{bmatrix}`
              - :math:`-iY`
            * - ``"Z"``
              - 0
              - :math:`[0, 1]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & -1  \end{bmatrix}`
              - :math:`Z`
            * - ``"-Z"``
              - 1
              - :math:`[0, 1]`
              - :math:`\begin{bmatrix} -1 & 0 \\ 0 & 1  \end{bmatrix}`
              - :math:`-Z`

        Args:
            labels (list): Pauli stabilizer string label(es).

        Returns:
            StabilizerTable: the constructed StabilizerTable.

        Raises:
            QiskitError: If the input list is empty or contains invalid
                         Pauli stabilizer strings.
        """
        if isinstance(labels, str):
            labels = [labels]
        n_paulis = len(labels)
        if n_paulis == 0:
            raise QiskitError("Input Pauli list is empty.")
        # Get size from first Pauli
        pauli, phase = cls._from_label(labels[0])
        table = np.zeros((n_paulis, len(pauli)), dtype=bool)
        phases = np.zeros(n_paulis, dtype=bool)
        table[0], phases[0] = pauli, phase
        for i in range(1, n_paulis):
            table[i], phases[i] = cls._from_label(labels[i])
        return cls(table, phases)

    def to_labels(self, array=False):
        r"""Convert a StabilizerTable to a list Pauli stabilizer string labels.

        For large StabilizerTables converting using the ``array=True``
        kwarg will be more efficient since it allocates memory for
        the full Numpy array of labels in advance.

        .. list-table:: Stabilizer Representations
            :header-rows: 1

            * - Label
              - Phase
              - Symplectic
              - Matrix
              - Pauli
            * - ``"+I"``
              - 0
              - :math:`[0, 0]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`
              - :math:`I`
            * - ``"-I"``
              - 1
              - :math:`[0, 0]`
              - :math:`\begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}`
              - :math:`-I`
            * - ``"X"``
              - 0
              - :math:`[1, 0]`
              - :math:`\begin{bmatrix} 0 & 1 \\ 1 & 0  \end{bmatrix}`
              - :math:`X`
            * - ``"-X"``
              - 1
              - :math:`[1, 0]`
              - :math:`\begin{bmatrix} 0 & -1 \\ -1 & 0  \end{bmatrix}`
              - :math:`-X`
            * - ``"Y"``
              - 0
              - :math:`[1, 1]`
              - :math:`\begin{bmatrix} 0 & 1 \\ -1 & 0  \end{bmatrix}`
              - :math:`iY`
            * - ``"-Y"``
              - 1
              - :math:`[1, 1]`
              - :math:`\begin{bmatrix} 0 & -1 \\ 1 & 0  \end{bmatrix}`
              - :math:`-iY`
            * - ``"Z"``
              - 0
              - :math:`[0, 1]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & -1  \end{bmatrix}`
              - :math:`Z`
            * - ``"-Z"``
              - 1
              - :math:`[0, 1]`
              - :math:`\begin{bmatrix} -1 & 0 \\ 0 & 1  \end{bmatrix}`
              - :math:`-Z`

        Args:
            array (bool): return a Numpy array if True, otherwise
                          return a list (Default: False).

        Returns:
            list or array: The rows of the StabilizerTable in label form.
        """
        ret = np.zeros(self.size, dtype=f"<U{1 + self.num_qubits}")
        for i in range(self.size):
            ret[i] = self._to_label(self._array[i], self._phase[i])
        if array:
            return ret
        return ret.tolist()

    def to_matrix(self, sparse=False, array=False):
        r"""Convert to a list or array of Stabilizer matrices.

        For large StabilizerTables converting using the ``array=True``
        kwarg will be more efficient since it allocates memory for the full
        rank-3 Numpy array of matrices in advance.

        .. list-table:: Stabilizer Representations
            :header-rows: 1

            * - Label
              - Phase
              - Symplectic
              - Matrix
              - Pauli
            * - ``"+I"``
              - 0
              - :math:`[0, 0]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`
              - :math:`I`
            * - ``"-I"``
              - 1
              - :math:`[0, 0]`
              - :math:`\begin{bmatrix} -1 & 0 \\ 0 & -1 \end{bmatrix}`
              - :math:`-I`
            * - ``"X"``
              - 0
              - :math:`[1, 0]`
              - :math:`\begin{bmatrix} 0 & 1 \\ 1 & 0  \end{bmatrix}`
              - :math:`X`
            * - ``"-X"``
              - 1
              - :math:`[1, 0]`
              - :math:`\begin{bmatrix} 0 & -1 \\ -1 & 0  \end{bmatrix}`
              - :math:`-X`
            * - ``"Y"``
              - 0
              - :math:`[1, 1]`
              - :math:`\begin{bmatrix} 0 & 1 \\ -1 & 0  \end{bmatrix}`
              - :math:`iY`
            * - ``"-Y"``
              - 1
              - :math:`[1, 1]`
              - :math:`\begin{bmatrix} 0 & -1 \\ 1 & 0  \end{bmatrix}`
              - :math:`-iY`
            * - ``"Z"``
              - 0
              - :math:`[0, 1]`
              - :math:`\begin{bmatrix} 1 & 0 \\ 0 & -1  \end{bmatrix}`
              - :math:`Z`
            * - ``"-Z"``
              - 1
              - :math:`[0, 1]`
              - :math:`\begin{bmatrix} -1 & 0 \\ 0 & 1  \end{bmatrix}`
              - :math:`-Z`

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
            return [
                self._to_matrix(pauli, phase, sparse=sparse)
                for pauli, phase in zip(self._array, self._phase)
            ]
        # For efficiency we also allow returning a single rank-3
        # array where first index is the Pauli row, and second two
        # indices are the matrix indices
        dim = 2 ** self.num_qubits
        ret = np.zeros((self.size, dim, dim), dtype=float)
        for i in range(self.size):
            ret[i] = self._to_matrix(self._array[i], self._phase[i])
        return ret

    @staticmethod
    def _from_label(label):
        """Return the symplectic representation of a Pauli stabilizer string"""
        # Check if first character is '+' or '-'
        phase = False
        if label[0] in ["-", "+"]:
            phase = label[0] == "-"
            label = label[1:]

        num_qubits = len(label)
        symp = np.zeros(2 * num_qubits, dtype=bool)
        xs = symp[0:num_qubits]
        zs = symp[num_qubits : 2 * num_qubits]
        for i, char in enumerate(label):
            if char not in ["I", "X", "Y", "Z"]:
                raise QiskitError(
                    "Pauli string contains invalid character:"
                    " {} not in ['I', 'X', 'Y', 'Z'].".format(char)
                )
            if char in ["X", "Y"]:
                xs[num_qubits - 1 - i] = True
            if char in ["Z", "Y"]:
                zs[num_qubits - 1 - i] = True
        return symp, phase

    @staticmethod
    def _to_label(pauli, phase):
        """Return the Pauli stabilizer string from symplectic representation."""
        # pylint: disable=arguments-differ
        # Cast in symplectic representation
        # This should avoid a copy if the pauli is already a row
        # in the symplectic table
        symp = np.asarray(pauli, dtype=bool)
        num_qubits = symp.size // 2
        x = symp[0:num_qubits]
        z = symp[num_qubits : 2 * num_qubits]
        paulis = np.zeros(num_qubits, dtype="<U1")
        for i in range(num_qubits):
            if not z[i]:
                if not x[i]:
                    paulis[num_qubits - 1 - i] = "I"
                else:
                    paulis[num_qubits - 1 - i] = "X"
            elif not x[i]:
                paulis[num_qubits - 1 - i] = "Z"
            else:
                paulis[num_qubits - 1 - i] = "Y"
        label = "".join(paulis)
        if phase:
            return "-" + label
        return "+" + label

    @staticmethod
    def _to_matrix(pauli, phase, sparse=False):
        """ "Return the Pauli stabilizer matrix from symplectic representation.

        Args:
            pauli (array): symplectic Pauli vector.
            phase (bool): the phase value for the Pauli.
            sparse (bool): if True return a sparse CSR matrix, otherwise
                           return a dense Numpy array (Default: False).

        Returns:
            array: if sparse=False.
            csr_matrix: if sparse=True.
        """
        symp = np.asarray(pauli, dtype=bool)
        num_qubits = symp.size // 2
        x = symp[0:num_qubits]
        z = symp[num_qubits : 2 * num_qubits]

        dim = 2 ** num_qubits
        twos_array = 1 << np.arange(num_qubits)
        x_indices = np.array(x).dot(twos_array)
        z_indices = np.array(z).dot(twos_array)

        indptr = np.arange(dim + 1, dtype=np.uint)
        indices = indptr ^ x_indices
        data = np.array([(-1) ** (bin(i).count("1") % 2) for i in z_indices & indptr])

        if sparse:
            # Return sparse matrix
            from scipy.sparse import csr_matrix

            mat = csr_matrix((data, indices, indptr), shape=(dim, dim), dtype=float)
            if phase:
                mat *= -1
            return mat

        # Build dense matrix using csr format
        mat = np.zeros((dim, dim), dtype=float)
        for i in range(dim):
            mat[i][indices[indptr[i] : indptr[i + 1]]] = data[indptr[i] : indptr[i + 1]]

        if phase:
            mat *= -1
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
            LabelIterator: label iterator object for the StabilizerTable.
        """

        class LabelIterator(CustomIterator):
            """Label representation iteration and item access."""

            def __repr__(self):
                return f"<StabilizerTable_label_iterator at {hex(id(self))}>"

            def __getitem__(self, key):
                return self.obj._to_label(self.obj.array[key], self.obj.phase[key])

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
            MatrixIterator: matrix iterator object for the StabilizerTable.
        """

        class MatrixIterator(CustomIterator):
            """Matrix representation iteration and item access."""

            def __repr__(self):
                return f"<StabilizerTable_matrix_iterator at {hex(id(self))}>"

            def __getitem__(self, key):
                return self.obj._to_matrix(self.obj.array[key], self.obj.phase[key], sparse=sparse)

        return MatrixIterator(self)


# Update docstrings for API docs
generate_apidocs(StabilizerTable)
