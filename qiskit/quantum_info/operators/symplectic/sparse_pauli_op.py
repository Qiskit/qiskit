# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
N-Qubit Sparse Pauli Operator class.
"""

from numbers import Number
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic.pauli_table import PauliTable
from qiskit.quantum_info.operators.symplectic.pauli_utils import pauli_basis
from qiskit.quantum_info.operators.custom_iterator import CustomIterator


class SparsePauliOp(BaseOperator):
    """Sparse N-qubit operator in a Pauli basis representation.

    This is a sparse representation of an N-qubit matrix
    :class:`~qiskit.quantum_info.Operator` in terms of N-qubit
    :class:`~qiskit.quantum_info.PauliTable` and complex coefficients.

    It can be used for performing operator arithmetic for hundred of qubits
    if the number of non-zero Pauli basis terms is sufficiently small.

    The Pauli basis components are stored as a
    :class:`~qiskit.quantum_info.PauliTable` object and can be accessed
    using the :attr:`~SparsePauliOp.table` attribute. The coefficients
    are stored as a complex Numpy array vector and can be accessed using
    the :attr:`~SparsePauliOp.coeffs` attribute.
    """

    def __init__(self, data, coeffs=None):
        """Initialize an operator object.

        Args:
            data (PauliTable): Pauli table of terms.
            coeffs (np.ndarray): complex coefficients for Pauli terms.

        Raises:
            QiskitError: If the input data or coeffs are invalid.
        """
        if isinstance(data, SparsePauliOp):
            table = data._table
            coeffs = data._coeffs
        else:
            table = PauliTable(data)
            if coeffs is None:
                coeffs = np.ones(table.size, dtype=np.complex)
        # Initialize PauliTable
        self._table = table

        # Initialize Coeffs
        self._coeffs = np.asarray(coeffs, dtype=complex)
        if self._coeffs.shape != (self._table.size, ):
            raise QiskitError("coeff vector is incorrect shape for number"
                              " of Paulis {} != {}".format(self._coeffs.shape,
                                                           self._table.size))
        # Initialize BaseOperator
        super().__init__(self._table._input_dims, self._table._output_dims)

    def __repr__(self):
        prefix = 'SparsePauliOp('
        pad = len(prefix) * ' '
        return '{}{},\n{}coeffs={})'.format(
            prefix, np.array2string(
                self.table.array, separator=', ', prefix=prefix),
            pad, np.array2string(self.coeffs, separator=', '))

    def __eq__(self, other):
        """Check if two SparsePauliOp operators are equal"""
        return (super().__eq__(other)
                and np.allclose(self.coeffs, other.coeffs)
                and self.table == other.table)

    # ---------------------------------------------------------------------
    # Data accessors
    # ---------------------------------------------------------------------

    @property
    def size(self):
        """The number of Pauli of Pauli terms in the operator."""
        return self._table.size

    def __len__(self):
        """Return the size."""
        return self.size

    @property
    def table(self):
        """Return the the PauliTable."""
        return self._table

    @table.setter
    def table(self, value):
        if not isinstance(value, PauliTable):
            value = PauliTable(value)
        self._table.array = value.array

    @property
    def coeffs(self):
        """Return the Pauli coefficients."""
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        """Set Pauli coefficients."""
        self._coeffs[:] = value

    def __getitem__(self, key):
        """Return a view of the SparsePauliOp."""
        # Returns a view of specified rows of the PauliTable
        # This supports all slicing operations the underlying array supports.
        if isinstance(key, (int, np.int)):
            key = [key]
        return SparsePauliOp(self.table[key], self.coeffs[key])

    def __setitem__(self, key, value):
        """Update SparsePauliOp."""
        # Modify specified rows of the PauliTable
        if not isinstance(value, SparsePauliOp):
            value = SparsePauliOp(value)
        self.table[key] = value.table
        self.coeffs[key] = value.coeffs

    # ---------------------------------------------------------------------
    # BaseOperator Methods
    # ---------------------------------------------------------------------

    def is_unitary(self, atol=None, rtol=None):
        """Return True if operator is a unitary matrix.

        Args:
            atol (float): Optional. Absolute tolerance for checking if
                          coefficients are zero (Default: 1e-8).
            rtol (float): Optinoal. relative tolerance for checking if
                          coefficients are zero (Default: 1e-5).

        Returns:
            bool: True if the operator is unitary, False otherwise.
        """
        # Get default atol and rtol
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        # Compose with adjoint
        val = self.compose(self.adjoint()).simplify()
        # See if the result is an identity
        return (val.size == 1 and np.isclose(val.coeffs[0], 1.0, atol=atol, rtol=rtol)
                and not np.any(val.table.X) and not np.any(val.table.Z))

    def conjugate(self):
        """Return the conjugate of the operator."""
        # Conjugation conjugates phases and also Y.conj() = -Y
        # Hence we need to multiply conjugated coeffs by -1
        # for rows with an odd number of Y terms.
        # Find rows with Ys
        ret = self.transpose()
        ret._coeffs = ret._coeffs.conj()
        return ret

    def transpose(self):
        """Return the transpose of the operator."""
        # The only effect transposition has is Y.T = -Y
        # Hence we need to multiply coeffs by -1 for rows with an
        # odd number of Y terms.
        ret = self.copy()
        minus = (-1) ** np.mod(np.sum(ret.table.X & ret.table.Z, axis=1), 2)
        ret._coeffs *= minus
        return ret

    def adjoint(self):
        """Return the adjoint of the operator."""
        # Pauli's are self adjoint, so we only need to
        # conjugate the phases
        ret = self.copy()
        ret._coeffs = ret._coeffs.conj()
        return ret

    def compose(self, other, qargs=None, front=False):
        """Return the composition channel self∘other.

        Args:
            other (SparsePauliOp): an operator object.
            qargs (list or None): a list of subsystem positions to compose other on.
            front (bool or None): If False compose in standard order other(self(input))
                          otherwise compose in reverse order self(other(input))
                          [default: False]

        Returns:
            SparsePauliOp: The composed operator.

        Raises:
            QiskitError: if other cannot be converted to an Operator or has
            incompatible dimensions.
        """
        # pylint: disable=invalid-name
        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        if not isinstance(other, SparsePauliOp):
            other = SparsePauliOp(other)

        # Validate composition dimensions and qargs match
        self._get_compose_dims(other, qargs, front)

        # Implement composition of the Pauli table
        x1, x2 = PauliTable._block_stack(self.table.X, other.table.X)
        z1, z2 = PauliTable._block_stack(self.table.Z, other.table.Z)
        c1, c2 = PauliTable._block_stack(self.coeffs, other.coeffs)

        if qargs is not None:
            ret_x, ret_z = x1.copy(), z1.copy()
            x1 = x1[:, qargs]
            z1 = z1[:, qargs]
            ret_x[:, qargs] = x1 ^ x2
            ret_z[:, qargs] = z1 ^ z2
            table = np.hstack([ret_x, ret_z])
        else:
            table = np.hstack((x1 ^ x2, z1 ^ z2))

        # Take product of coefficients and add phase correction
        coeffs = c1 * c2
        # We pick additional phase terms for the products
        # X.Y = i * Z, Y.Z = i * X, Z.X = i * Y
        # Y.X = -i * Z, Z.Y = -i * X, X.Z = -i * Y
        if front:
            plus_i = (x1 & ~z1 & x2 & z2) | (x1 & z1 & ~x2 & z2) | (~x1 & z1 & x2 & ~z2)
            minus_i = (x2 & ~z2 & x1 & z1) | (x2 & z2 & ~x1 & z1) | (~x2 & z2 & x1 & ~z1)
        else:
            minus_i = (x1 & ~z1 & x2 & z2) | (x1 & z1 & ~x2 & z2) | (~x1 & z1 & x2 & ~z2)
            plus_i = (x2 & ~z2 & x1 & z1) | (x2 & z2 & ~x1 & z1) | (~x2 & z2 & x1 & ~z1)
        coeffs *= 1j ** np.array(np.sum(plus_i, axis=1), dtype=int)
        coeffs *= (-1j) ** np.array(np.sum(minus_i, axis=1), dtype=int)
        return SparsePauliOp(table, coeffs)

    def dot(self, other, qargs=None):
        """Return the composition channel self∘other.

        Args:
            other (SparsePauliOp): an operator object.
            qargs (list or None): a list of subsystem positions to compose other on.

        Returns:
            SparsePauliOp: The composed operator.

        Raises:
            QiskitError: if other cannot be converted to an Operator or has
            incompatible dimensions.
        """
        return self.compose(other, qargs=qargs, front=True)

    def tensor(self, other):
        """Return the tensor product operator self ⊗ other.

        Args:
            other (SparsePauliOp): a operator subclass object.

        Returns:
            SparsePauliOp: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other cannot be converted to a SparsePauliOp
                         operator.
        """
        if not isinstance(other, SparsePauliOp):
            other = SparsePauliOp(other)
        table = self.table.tensor(other.table)
        coeffs = np.kron(self.coeffs, other.coeffs)
        return SparsePauliOp(table, coeffs)

    def expand(self, other):
        """Return the tensor product operator other ⊗ self.

        Args:
            other (SparsePauliOp): an operator object.

        Returns:
            SparsePauliOp: the tensor product operator other ⊗ self.

        Raises:
            QiskitError: if other cannot be converted to a SparsePauliOp
                         operator.
        """
        if not isinstance(other, SparsePauliOp):
            other = SparsePauliOp(other)
        table = self.table.expand(other.table)
        coeffs = np.kron(self.coeffs, other.coeffs)
        return SparsePauliOp(table, coeffs)

    def _add(self, other, qargs=None):
        """Return the operator self + other.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (SparsePauliOp): an operator object.
            qargs (None or list): optional subsystems to add on
                                  (Default: None)

        Returns:
            SparsePauliOp: the operator self + other.

        Raises:
            QiskitError: if other cannot be converted to a SparsePauliOp
                         or has incompatible dimensions.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        if not isinstance(other, SparsePauliOp):
            other = SparsePauliOp(other)

        self._validate_add_dims(other, qargs)

        table = self.table._add(other.table, qargs=qargs)
        coeffs = np.hstack((self.coeffs, other.coeffs))
        ret = SparsePauliOp(table, coeffs)
        return ret

    def _multiply(self, other):
        """Return the operator other * self.

        Args:
            other (complex): a complex number.

        Returns:
            SparsePauliOp: the operator other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        if other == 0:
            # Check edge case that we deleted all Paulis
            # In this case we return an identity Pauli with a zero coefficient
            table = np.zeros((1, 2 * self.num_qubits), dtype=np.bool)
            coeffs = np.array([0j])
            return SparsePauliOp(table, coeffs)
        # Otherwise we just update the phases
        return SparsePauliOp(self.table, other * self.coeffs)

    # ---------------------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------------------

    def simplify(self, atol=None, rtol=None):
        """Simplify PauliTable by combining duplicaties and removing zeros.

        Args:
            atol (float): Optional. Absolute tolerance for checking if
                          coefficients are zero (Default: 1e-8).
            rtol (float): Optinoal. relative tolerance for checking if
                          coefficients are zero (Default: 1e-5).

        Returns:
            SparsePauliOp: the simplified SparsePauliOp operator.
        """
        # Get default atol and rtol
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        table, indexes = np.unique(self.table.array,
                                   return_inverse=True, axis=0)
        coeffs = np.zeros(len(table), dtype=np.complex)
        for i, val in zip(indexes, self.coeffs):
            coeffs[i] += val
        # Delete zero coefficient rows
        # TODO: Add atol/rtol for zero comparison
        non_zero = [i for i in range(coeffs.size)
                    if not np.isclose(coeffs[i], 0, atol=atol, rtol=rtol)]
        table = table[non_zero]
        coeffs = coeffs[non_zero]
        # Check edge case that we deleted all Paulis
        # In this case we return an identity Pauli with a zero coefficient
        if coeffs.size == 0:
            table = np.zeros((1, 2*self.num_qubits), dtype=np.bool)
            coeffs = np.array([0j])
        return SparsePauliOp(table, coeffs)

    # ---------------------------------------------------------------------
    # Additional conversions
    # ---------------------------------------------------------------------

    @staticmethod
    def from_operator(obj, atol=None, rtol=None):
        """Construct from an Operator objector.

        Note that the cost of this contruction is exponential as it involves
        taking inner products with every element of the N-qubit Pauli basis.

        Args:
            obj (Operator): an N-qubit operator.
            atol (float): Optional. Absolute tolerance for checking if
                          coefficients are zero (Default: 1e-8).
            rtol (float): Optinoal. relative tolerance for checking if
                          coefficients are zero (Default: 1e-5).

        Returns:
            SparsePauliOp: the SparsePauliOp representation of the operator.

        Raises:
            QiskitError: if the input operator is not an N-qubit operator.
        """
        # Get default atol and rtol
        if atol is None:
            atol = SparsePauliOp._ATOL_DEFAULT
        if rtol is None:
            rtol = SparsePauliOp._RTOL_DEFAULT

        if not isinstance(obj, Operator):
            obj = Operator(obj)

        # Check dimension is N-qubit operator
        num_qubits = obj.num_qubits
        if num_qubits is None:
            raise QiskitError("Input Operator is not an N-qubit operator.")
        data = obj.data

        # Index of non-zero basis elements
        inds = []
        # Non-zero coefficients
        coeffs = []
        # Non-normalized basis factor
        denom = 2 ** num_qubits
        # Compute coefficients from basis
        basis = pauli_basis(num_qubits)
        for i, mat in enumerate(basis.matrix_iter()):
            coeff = np.trace(mat.dot(data)) / denom
            if not np.isclose(coeff, 0, atol=atol, rtol=rtol):
                inds.append(i)
                coeffs.append(coeff)
        # Get Non-zero coeff PauliTable terms
        table = basis[inds]
        return SparsePauliOp(table, coeffs)

    @staticmethod
    def from_list(obj):
        """Construct from a list [(pauli_str, coeffs)]"""
        obj = list(obj)  # To convert zip or other iterable
        num_qubits = len(PauliTable._from_label(obj[0][0]))
        size = len(obj)
        coeffs = np.zeros(size, dtype=complex)
        labels = np.zeros(size, dtype='<U{}'.format(num_qubits))
        for i, item in enumerate(obj):
            labels[i] = item[0]
            coeffs[i] = item[1]
        table = PauliTable.from_labels(labels)
        return SparsePauliOp(table, coeffs)

    def to_list(self, array=False):
        """Convert to a list Pauli string labels and coefficients.

        For operators with a lot of terms converting using the ``array=True``
        kwarg will be more efficient since it allocates memory for
        the full Numpy array of labels in advance.

        Args:
            array (bool): return a Numpy array if True, otherwise
                          return a list (Default: False).

        Returns:
            list or array: List of pairs (label, coeff) for rows of the PauliTable.
        """
        # Dtype for a structured array with string labels and complex coeffs
        pauli_labels = self.table.to_labels(array=True)
        labels = np.zeros(self.size,
                          dtype=[('labels', pauli_labels.dtype),
                                 ('coeffs', 'c16')])
        labels['labels'] = pauli_labels
        labels['coeffs'] = self.coeffs
        if array:
            return labels
        return labels.tolist()

    def to_matrix(self, sparse=False):
        """Convert to a dense or sparse matrix.

        Args:
            sparse (bool): if True return a sparse CSR matrix, otherwise
                           return dense Numpy array (Default: False).

        Returns:
            array: A dense matrix if `sparse=False`.
            csr_matrix: A sparse matrix in CSR format if `sparse=True`.
        """
        mat = None
        for i in self.matrix_iter(sparse=sparse):
            if mat is None:
                mat = i
            else:
                mat += i
        return mat

    def to_operator(self):
        """Convert to a matrix Operator object"""
        return Operator(self.to_matrix())

    # ---------------------------------------------------------------------
    # Custom Iterators
    # ---------------------------------------------------------------------

    def label_iter(self):
        """Return a label representation iterator.

        This is a lazy iterator that converts each term in the SparsePauliOp
        into a tuple (label, coeff). To convert the entire table to labels
        use the :meth:`to_labels` method.

        Returns:
            LabelIterator: label iterator object for the PauliTable.
        """
        class LabelIterator(CustomIterator):
            """Label representation iteration and item access."""
            def __repr__(self):
                return "<SparsePauliOp_label_iterator at {}>".format(
                    hex(id(self)))

            def __getitem__(self, key):
                coeff = self.obj.coeffs[key]
                pauli = PauliTable._to_label(self.obj.table.array[key])
                return (pauli, coeff)

        return LabelIterator(self)

    def matrix_iter(self, sparse=False):
        """Return a matrix representation iterator.

        This is a lazy iterator that converts each term in the SparsePauliOp
        into a matrix as it is used. To convert to a single matrix use the
        :meth:`to_matrix` method.

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
                return "<SparsePauliOp_matrix_iterator at {}>".format(hex(id(self)))

            def __getitem__(self, key):
                coeff = self.obj.coeffs[key]
                mat = PauliTable._to_matrix(self.obj.table.array[key],
                                            sparse=sparse)
                return coeff * mat

        return MatrixIterator(self)
