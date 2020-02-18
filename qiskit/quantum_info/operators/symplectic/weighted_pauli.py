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


class WeightedPauli(BaseOperator):
    """Weighted Pauli operator class.

    This is a sparse representation of an N-qubit matrix
    :class:`~qiskit.quantum_info.Operator` in terms of N-qubit Pauli basis
    components and complex coefficients.

    It can be used for performing operator arithmatic on operators for
    hundred of qubits if the number of non-zero Pauli basis terms is small.

    The Pauli basis components are stored as a
    :class:`~qiskit.quantum_info.PauliTable` object and can be accessed
    using the :attr:`~WeightedPauli.pauli` attribute. The coefficients
    are stored as a complex Numpy array vector and can be accessed using
    the :attr:`~WeightedPauli.coeffs` attribute.
    """

    def __init__(self, data, coeffs=None):
        """Initialize an operator object."""
        if isinstance(data, WeightedPauli):
            pauli = data._pauli
            coeffs = data._coeffs
        else:
            pauli = PauliTable(data)
            if coeffs is None:
                coeffs = np.ones(pauli.size, dtype=np.complex)
        # Initialize PauliTable
        self._pauli = pauli

        # Initialize Coeffs
        self._coeffs = np.asarray(coeffs, dtype=complex)
        if self._coeffs.shape != (self._pauli.size, ):
            raise QiskitError("coeff vector is incorrect shape for number"
                              " of Paulis {} != {}".format(self._coeffs.shape,
                                                           self.pauli.size))
        # Initialize BaseOperator
        super().__init__(pauli._input_dims, pauli._output_dims)

    def __repr__(self):
        prefix = 'WeightedPauli('
        pad = len(prefix) * ' '
        return '{}{},\n{}coeffs={})'.format(
            prefix, np.array2string(
                self.pauli.array, separator=', ', prefix=prefix),
            pad, np.array2string(self.coeffs, separator=', '))

    def __eq__(self, other):
        """Check if two WeightedPauli operators are equal"""
        return (super().__eq__(other)
                and np.allclose(self.coeffs, other.coeffs)
                and self.pauli == other.pauli)

    # ---------------------------------------------------------------------
    # Data accessors
    # ---------------------------------------------------------------------

    @property
    def size(self):
        """The number of Pauli of Pauli terms in the operator."""
        return self._pauli.size

    def __len__(self):
        """Return the size."""
        return self.size

    @property
    def pauli(self):
        """Return the the PauliTable."""
        return self._pauli

    @pauli.setter
    def pauli(self, value):
        if not isinstance(value, PauliTable):
            value = PauliTable(value)
        self._pauli.array = value.array

    @property
    def coeffs(self):
        """Return the Pauli coefficients."""
        return self._coeffs

    @coeffs.setter
    def coeffs(self, value):
        """Set Pauli coefficients."""
        self._coeffs[:] = value

    def __getitem__(self, key):
        """Return a view of the WeightedPauli."""
        # Returns a view of specified rows of the PauliTable
        # This supports all slicing operations the underlying array supports.
        if isinstance(key, (int, np.int)):
            key = [key]
        return WeightedPauli(self.pauli[key], self.coeffs[key])

    def __setitem__(self, key, value):
        """Update WeightedPauli."""
        # Modify specified rows of the PauliTable
        if not isinstance(value, WeightedPauli):
            value = WeightedPauli(value)
        self.pauli[key] = value.pauli
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
        if (val.size == 1 and np.all(~val.pauli.X & ~val.pauli.Z)
                and np.isclose(val.coeffs[0], 1.0, atol=atol, rtol=rtol)):
            return True
        return False

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
        minus = (-1) ** np.sum(ret.X & ret.Z, axis=1) % 2
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
            other (WeightedPauli): an operator object.
            qargs (list or None): a list of subsystem positions to compose other on.
            front (bool or None): If False compose in standard order other(self(input))
                          otherwise compose in reverse order self(other(input))
                          [default: False]

        Returns:
            WeightedPauli: The composed operator.

        Raises:
            QiskitError: if other cannot be converted to an Operator or has
            incompatible dimensions.
        """
        # pylint: disable=invalid-name
        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        if not isinstance(other, WeightedPauli):
            other = WeightedPauli(other)

        # Validate composition dimensions and qargs match
        self._get_compose_dims(other, qargs, front)

        # Implement composition of the Pauli table
        x1, x2 = PauliTable._block_stack(self.pauli.X, other.pauli.X)
        z1, z2 = PauliTable._block_stack(self.pauli.Z, other.pauli.Z)
        c1, c2 = PauliTable._block_stack(self.coeffs, other.coeffs)

        if qargs is not None:
            ret_x, ret_z = x1.copy(), z1.copy()
            x1 = x1[:, qargs]
            z1 = z1[:, qargs]
            ret_x[:, qargs] = x1 ^ x2
            ret_z[:, qargs] = z1 ^ z2
            pauli = np.hstack([ret_x, ret_z])
        else:
            pauli = np.hstack((x1 ^ x2, z1 ^ z2))

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
        return WeightedPauli(pauli, coeffs)

    def dot(self, other, qargs=None):
        """Return the composition channel self∘other.

        Args:
            other (WeightedPauli): an operator object.
            qargs (list or None): a list of subsystem positions to compose other on.

        Returns:
            WeightedPauli: The composed operator.

        Raises:
            QiskitError: if other cannot be converted to an Operator or has
            incompatible dimensions.
        """
        return self.compose(other, qargs=qargs, front=True)

    def tensor(self, other):
        """Return the tensor product operator self ⊗ other.

        Args:
            other (WeightedPauli): a operator subclass object.

        Returns:
            WeightedPauli: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other cannot be converted to a WeightedPauli
                         operator.
        """
        if not isinstance(other, WeightedPauli):
            other = WeightedPauli(other)
        pauli = self.pauli.tensor(other.pauli)
        coeffs = np.concatenate([self.coeffs, other.coeffs])
        return WeightedPauli(pauli, coeffs)

    def expand(self, other):
        """Return the tensor product operator other ⊗ self.

        Args:
            other (WeightedPauli): an operator object.

        Returns:
            WeightedPauli: the tensor product operator other ⊗ self.

        Raises:
            QiskitError: if other cannot be converted to a WeightedPauli
                         operator.
        """
        if not isinstance(other, WeightedPauli):
            other = WeightedPauli(other)
        pauli = self.pauli.expand(other.pauli)
        coeffs = np.concatenate([self.coeffs, other.coeffs])
        return WeightedPauli(pauli, coeffs)

    def _add(self, other, qargs=None):
        """Return the operator self + other.

        If ``qargs`` are specified the other operator will be added
        assuming it is identity on all other subsystems.

        Args:
            other (WeightedPauli): an operator object.
            qargs (None or list): optional subsystems to add on
                                  (Default: None)

        Returns:
            WeightedPauli: the operator self + other.

        Raises:
            QiskitError: if other cannot be converted to a WeightedPauli
                         or has incompatible dimensions.
        """
        if qargs is None:
            qargs = getattr(other, 'qargs', None)

        if not isinstance(other, WeightedPauli):
            other = WeightedPauli(other)

        self._validate_add_dims(other, qargs)

        table = self.pauli._add(other.pauli, qargs=qargs)
        coeffs = np.hstack((self.coeffs, other.coeffs))
        ret = WeightedPauli(table, coeffs)
        return ret

    def _multiply(self, other):
        """Return the operator other * self.

        Args:
            other (complex): a complex number.

        Returns:
            WeightedPauli: the operator other * self.

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
            return WeightedPauli(table, coeffs)
        # Otherwise we just update the phases
        return WeightedPauli(self.pauli, other * self.coeffs)

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
            WeightedPauli: the simplified WeightedPauli operator.
        """
        # Get default atol and rtol
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        table, indexes = np.unique(self.pauli.array,
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
        return WeightedPauli(table, coeffs)

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
            WeightedPauli: the WeightedPauli representation of the operator.

        Raises:
            QiskitError: if the input operator is not an N-qubit operator.
        """
        # Get default atol and rtol
        if atol is None:
            atol = WeightedPauli.atol
        if rtol is None:
            rtol = WeightedPauli.rtol

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
        pauli = basis[inds]
        return WeightedPauli(pauli, coeffs)

    @staticmethod
    def from_labels(obj):
        """Construct from a list [(coeff, Pauli_Str)]"""
        num_qubits = len(PauliTable._from_label(obj[0][0]))
        size = len(obj)
        coeffs = np.zeros(size, dtype=complex)
        labels = np.zeros(size, dtype='<U{}'.format(num_qubits))
        for i, item in enumerate(obj):
            labels[i] = item[0]
            coeffs[i] = item[1]
        table = PauliTable.from_labels(labels)
        return WeightedPauli(table, coeffs)

    def to_labels(self, array=False):
        """Convert to a list Pauli string labels and coefficients.

        For operators with a lot of terms converting using the ``array=True``
        kwarg will be more efficient since it allocates memory for
        the full Numpy array of labels in advance.

        Args:
            array (bool): return a Numpy array if True, otherwise
                          return a list (Default: False).

        Returns:
            list or array: The rows of the PauliTable in label form.
        """
        # Dtype for a structured array with string labels and complex coeffs
        pauli_labels = self.pauli.to_labels(array=True)
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

        This is a lazy iterator that converts each term in the WeightedPauli
        into a tuple (label, coeff). To convert the entire table to labels
        use the :meth:`to_labels` method.

        Returns:
            LabelIterator: label iterator object for the PauliTable.
        """
        class LabelIterator(CustomIterator):
            """Label representation iteration and item access."""
            def __repr__(self):
                return "<WeightedPauli_label_iterator at {}>".format(
                    hex(id(self)))

            def __getitem__(self, key):
                coeff = self.obj.coeffs[key]
                pauli = PauliTable._to_label(self.obj.pauli.array[key])
                return (pauli, coeff)

        return LabelIterator(self)

    def matrix_iter(self, sparse=False):
        """Return a matrix representation iterator.

        This is a lazy iterator that converts each term in the WeightedPauli
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
                return "<WeightedPauli_matrix_iterator at {}>".format(hex(id(self)))

            def __getitem__(self, key):
                coeff = self.obj.coeffs[key]
                mat = PauliTable._to_matrix(self.obj.pauli.array[key],
                                            sparse=sparse)
                return coeff * mat

        return MatrixIterator(self)
