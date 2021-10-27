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
from typing import Dict

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.custom_iterator import CustomIterator
from qiskit.quantum_info.operators.linear_op import LinearOp
from qiskit.quantum_info.operators.mixins import generate_apidocs
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic.pauli import BasePauli
from qiskit.quantum_info.operators.symplectic.pauli_list import PauliList
from qiskit.quantum_info.operators.symplectic.pauli_table import PauliTable
from qiskit.quantum_info.operators.symplectic.pauli_utils import pauli_basis
from qiskit.utils.deprecation import deprecate_function


class SparsePauliOp(LinearOp):
    """Sparse N-qubit operator in a Pauli basis representation.

    This is a sparse representation of an N-qubit matrix
    :class:`~qiskit.quantum_info.Operator` in terms of N-qubit
    :class:`~qiskit.quantum_info.PauliList` and complex coefficients.

    It can be used for performing operator arithmetic for hundred of qubits
    if the number of non-zero Pauli basis terms is sufficiently small.

    The Pauli basis components are stored as a
    :class:`~qiskit.quantum_info.PauliList` object and can be accessed
    using the :attr:`~SparsePauliOp.paulis` attribute. The coefficients
    are stored as a complex Numpy array vector and can be accessed using
    the :attr:`~SparsePauliOp.coeffs` attribute.
    """

    def __init__(self, data, coeffs=None):
        """Initialize an operator object.

        Args:
            data (Paulilist, SparsePauliOp, PauliTable): Pauli list of terms.
            coeffs (np.ndarray): complex coefficients for Pauli terms.

        Raises:
            QiskitError: If the input data or coeffs are invalid.
        """
        if isinstance(data, SparsePauliOp):
            pauli_list = data._pauli_list
            coeffs = data._coeffs
        else:
            pauli_list = PauliList(data)
            if coeffs is None:
                coeffs = np.ones(pauli_list.size, dtype=complex)
        # Initialize PauliList
        self._pauli_list = PauliList.from_symplectic(pauli_list.z, pauli_list.x)

        # Initialize Coeffs
        self._coeffs = np.asarray((-1j) ** pauli_list.phase * coeffs, dtype=complex)
        if self._coeffs.shape != (self._pauli_list.size,):
            raise QiskitError(
                "coeff vector is incorrect shape for number"
                " of Paulis {} != {}".format(self._coeffs.shape, self._pauli_list.size)
            )
        # Initialize LinearOp
        super().__init__(num_qubits=self._pauli_list.num_qubits)

    def __array__(self, dtype=None):
        if dtype:
            return np.asarray(self.to_matrix(), dtype=dtype)
        return self.to_matrix()

    def __repr__(self):
        prefix = "SparsePauliOp("
        pad = len(prefix) * " "
        return "{}{},\n{}coeffs={})".format(
            prefix,
            self.paulis.to_labels(),
            pad,
            np.array2string(self.coeffs, separator=", "),
        )

    def __eq__(self, other):
        """Check if two SparsePauliOp operators are equal"""
        return (
            super().__eq__(other)
            and np.allclose(self.coeffs, other.coeffs)
            and self.paulis == other.paulis
        )

    @property
    def settings(self) -> Dict:
        """Return settings."""
        return {"data": self._pauli_list, "coeffs": self._coeffs}

    # ---------------------------------------------------------------------
    # Data accessors
    # ---------------------------------------------------------------------

    @property
    def size(self):
        """The number of Pauli of Pauli terms in the operator."""
        return self._pauli_list.size

    def __len__(self):
        """Return the size."""
        return self.size

    # pylint: disable=bad-docstring-quotes

    @property
    @deprecate_function(
        "The SparsePauliOp.table method is deprecated as of Qiskit Terra 0.19.0 "
        "and will be removed no sooner than 3 months after the releasedate. "
        "Use SparsePauliOp.paulis method instead.",
    )
    def table(self):
        """DEPRECATED - Return the the PauliTable."""
        return PauliTable(np.column_stack((self.paulis.x, self.paulis.z)))

    @table.setter
    @deprecate_function(
        "The SparsePauliOp.table method is deprecated as of Qiskit Terra 0.19.0 "
        "and will be removed no sooner than 3 months after the releasedate. "
        "Use SparsePauliOp.paulis method instead.",
    )
    def table(self, value):
        if not isinstance(value, PauliTable):
            value = PauliTable(value)
        self._pauli_list = PauliList(value)

    @property
    def paulis(self):
        """Return the the PauliList."""
        return self._pauli_list

    @paulis.setter
    def paulis(self, value):
        if not isinstance(value, PauliList):
            value = PauliList(value)
        self._pauli_list = value

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
        # Returns a view of specified rows of the PauliList
        # This supports all slicing operations the underlying array supports.
        if isinstance(key, (int, np.integer)):
            key = [key]
        return SparsePauliOp(self.paulis[key], self.coeffs[key])

    def __setitem__(self, key, value):
        """Update SparsePauliOp."""
        # Modify specified rows of the PauliList
        if not isinstance(value, SparsePauliOp):
            value = SparsePauliOp(value)
        self.paulis[key] = value.paulis
        self.coeffs[key] = value.coeffs

    # ---------------------------------------------------------------------
    # LinearOp Methods
    # ---------------------------------------------------------------------

    def conjugate(self):
        # Conjugation conjugates phases and also Y.conj() = -Y
        # Hence we need to multiply conjugated coeffs by -1
        # for rows with an odd number of Y terms.
        # Find rows with Ys
        ret = self.transpose()
        ret._coeffs = ret._coeffs.conj()
        return ret

    def transpose(self):
        # The only effect transposition has is Y.T = -Y
        # Hence we need to multiply coeffs by -1 for rows with an
        # odd number of Y terms.
        ret = self.copy()
        minus = (-1) ** np.mod(np.sum(ret.paulis.x & ret.paulis.z, axis=1), 2)
        ret._coeffs *= minus
        return ret

    def adjoint(self):
        # Pauli's are self adjoint, so we only need to
        # conjugate the phases
        ret = self.copy()
        ret._coeffs = ret._coeffs.conj()
        return ret

    def compose(self, other, qargs=None, front=False):
        if qargs is None:
            qargs = getattr(other, "qargs", None)

        if not isinstance(other, SparsePauliOp):
            other = SparsePauliOp(other)

        # Validate composition dimensions and qargs match
        self._op_shape.compose(other._op_shape, qargs, front)

        if qargs is not None:
            x1, z1 = self.paulis.x[:, qargs], self.paulis.z[:, qargs]
        else:
            x1, z1 = self.paulis.x, self.paulis.z
        x2, z2 = other.paulis.x, other.paulis.z
        num_qubits = other.num_qubits

        # This method is the outer version of `BasePauli.compose`.
        # `x1` and `z1` have shape `(self.size, num_qubits)`.
        # `x2` and `z2` have shape `(other.size, num_qubits)`.
        # `x1[:, no.newaxis]` results in shape `(self.size, 1, num_qubits)`.
        # `ar = ufunc(x1[:, np.newaxis], x2)` will be in shape `(self.size, other.size, num_qubits)`.
        # So, `ar.reshape((-1, num_qubits))` will be in shape `(self.size * other.size, num_qubits)`.
        # Ref: https://numpy.org/doc/stable/user/theory.broadcasting.html

        phase = np.add.outer(self.paulis._phase, other.paulis._phase).reshape(-1)
        if front:
            q = np.logical_and(x1[:, np.newaxis], z2).reshape((-1, num_qubits))
        else:
            q = np.logical_and(z1[:, np.newaxis], x2).reshape((-1, num_qubits))
        phase = np.mod(phase + 2 * np.sum(q, axis=1), 4)

        x3 = np.logical_xor(x1[:, np.newaxis], x2).reshape((-1, num_qubits))
        z3 = np.logical_xor(z1[:, np.newaxis], z2).reshape((-1, num_qubits))

        if qargs is None:
            pauli_list = PauliList(BasePauli(z3, x3, phase))
        else:
            x4 = np.repeat(self.paulis.x, other.size, axis=0)
            z4 = np.repeat(self.paulis.z, other.size, axis=0)
            x4[:, qargs] = x3
            z4[:, qargs] = z3
            pauli_list = PauliList(BasePauli(z4, x4, phase))

        coeffs = np.kron(self.coeffs, other.coeffs)
        return SparsePauliOp(pauli_list, coeffs)

    def tensor(self, other):
        if not isinstance(other, SparsePauliOp):
            other = SparsePauliOp(other)
        return self._tensor(self, other)

    def expand(self, other):
        if not isinstance(other, SparsePauliOp):
            other = SparsePauliOp(other)
        return self._tensor(other, self)

    @classmethod
    def _tensor(cls, a, b):
        paulis = a.paulis.tensor(b.paulis)
        coeffs = np.kron(a.coeffs, b.coeffs)
        return SparsePauliOp(paulis, coeffs)

    def _add(self, other, qargs=None):
        if qargs is None:
            qargs = getattr(other, "qargs", None)

        if not isinstance(other, SparsePauliOp):
            other = SparsePauliOp(other)

        self._op_shape._validate_add(other._op_shape, qargs)

        paulis = self.paulis._add(other.paulis, qargs=qargs)
        coeffs = np.hstack((self.coeffs, other.coeffs))
        ret = SparsePauliOp(paulis, coeffs)
        return ret

    def _multiply(self, other):
        if not isinstance(other, Number):
            raise QiskitError("other is not a number")
        if other == 0:
            # Check edge case that we deleted all Paulis
            # In this case we return an identity Pauli with a zero coefficient
            paulis = PauliList.from_symplectic(
                np.zeros((1, self.num_qubits), dtype=bool),
                np.zeros((1, self.num_qubits), dtype=bool),
            )
            coeffs = np.array([0j])
            return SparsePauliOp(paulis, coeffs)
        # Otherwise we just update the phases
        return SparsePauliOp(self.paulis, other * self.coeffs)

    # ---------------------------------------------------------------------
    # Utility Methods
    # ---------------------------------------------------------------------

    def is_unitary(self, atol=None, rtol=None):
        """Return True if operator is a unitary matrix.

        Args:
            atol (float): Optional. Absolute tolerance for checking if
                          coefficients are zero (Default: 1e-8).
            rtol (float): Optional. relative tolerance for checking if
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
        return (
            val.size == 1
            and np.isclose(val.coeffs[0], 1.0, atol=atol, rtol=rtol)
            and not np.any(val.paulis.x)
            and not np.any(val.paulis.z)
        )

    def simplify(self, atol=None, rtol=None):
        """Simplify PauliList by combining duplicates and removing zeros.

        Args:
            atol (float): Optional. Absolute tolerance for checking if
                          coefficients are zero (Default: 1e-8).
            rtol (float): Optional. relative tolerance for checking if
                          coefficients are zero (Default: 1e-5).

        Returns:
            SparsePauliOp: the simplified SparsePauliOp operator.
        """
        # Get default atol and rtol
        if atol is None:
            atol = self.atol
        if rtol is None:
            rtol = self.rtol

        # Pack bool vectors into np.uint8 vectors by np.packbits
        array = np.packbits(self.paulis.x, axis=1) * 256 + np.packbits(self.paulis.z, axis=1)
        _, indexes, inverses = np.unique(array, return_index=True, return_inverse=True, axis=0)
        coeffs = np.zeros(indexes.shape[0], dtype=complex)
        np.add.at(coeffs, inverses, self.coeffs)
        # Delete zero coefficient rows
        is_zero = np.isclose(coeffs, 0, atol=atol, rtol=rtol)
        # Check edge case that we deleted all Paulis
        # In this case we return an identity Pauli with a zero coefficient
        if np.all(is_zero):
            x = np.zeros((1, self.num_qubits), dtype=bool)
            z = np.zeros((1, self.num_qubits), dtype=bool)
            coeffs = np.array([0j], dtype=complex)
        else:
            non_zero = np.logical_not(is_zero)
            non_zero_indexes = indexes[non_zero]
            x = self.paulis.x[non_zero_indexes]
            z = self.paulis.z[non_zero_indexes]
            coeffs = coeffs[non_zero]
        return SparsePauliOp(PauliList.from_symplectic(z, x), coeffs)

    # ---------------------------------------------------------------------
    # Additional conversions
    # ---------------------------------------------------------------------

    @staticmethod
    def from_operator(obj, atol=None, rtol=None):
        """Construct from an Operator objector.

        Note that the cost of this construction is exponential as it involves
        taking inner products with every element of the N-qubit Pauli basis.

        Args:
            obj (Operator): an N-qubit operator.
            atol (float): Optional. Absolute tolerance for checking if
                          coefficients are zero (Default: 1e-8).
            rtol (float): Optional. relative tolerance for checking if
                          coefficients are zero (Default: 1e-5).

        Returns:
            SparsePauliOp: the SparsePauliOp representation of the operator.

        Raises:
            QiskitError: if the input operator is not an N-qubit operator.
        """
        # Get default atol and rtol
        if atol is None:
            atol = SparsePauliOp.atol
        if rtol is None:
            rtol = SparsePauliOp.rtol

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
        basis = pauli_basis(num_qubits, pauli_list=True)
        for i, mat in enumerate(basis.matrix_iter()):
            coeff = np.trace(mat.dot(data)) / denom
            if not np.isclose(coeff, 0, atol=atol, rtol=rtol):
                inds.append(i)
                coeffs.append(coeff)
        # Get Non-zero coeff PauliList terms
        paulis = basis[inds]
        return SparsePauliOp(paulis, coeffs)

    @staticmethod
    def from_list(obj):
        """Construct from a list [(pauli_str, coeffs)]"""
        obj = list(obj)  # To convert zip or other iterable
        num_qubits = len(obj[0][0])
        size = len(obj)
        coeffs = np.zeros(size, dtype=complex)
        labels = np.zeros(size, dtype=f"<U{num_qubits}")
        for i, item in enumerate(obj):
            labels[i] = item[0]
            coeffs[i] = item[1]
        paulis = PauliList(labels)
        return SparsePauliOp(paulis, coeffs)

    def to_list(self, array=False):
        """Convert to a list Pauli string labels and coefficients.

        For operators with a lot of terms converting using the ``array=True``
        kwarg will be more efficient since it allocates memory for
        the full Numpy array of labels in advance.

        Args:
            array (bool): return a Numpy array if True, otherwise
                          return a list (Default: False).

        Returns:
            list or array: List of pairs (label, coeff) for rows of the PauliList.
        """
        # Dtype for a structured array with string labels and complex coeffs
        pauli_labels = self.paulis.to_labels(array=True)
        labels = np.zeros(self.size, dtype=[("labels", pauli_labels.dtype), ("coeffs", "c16")])
        labels["labels"] = pauli_labels
        labels["coeffs"] = self.coeffs
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
                return f"<SparsePauliOp_label_iterator at {hex(id(self))}>"

            def __getitem__(self, key):
                coeff = self.obj.coeffs[key]
                pauli = self.obj.paulis.label_iter()[key]
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
            MatrixIterator: matrix iterator object for the PauliList.
        """

        class MatrixIterator(CustomIterator):
            """Matrix representation iteration and item access."""

            def __repr__(self):
                return f"<SparsePauliOp_matrix_iterator at {hex(id(self))}>"

            def __getitem__(self, key):
                coeff = self.obj.coeffs[key]
                mat = self.obj.paulis[key].to_matrix(sparse)
                return coeff * mat

        return MatrixIterator(self)


# Update docstrings for API docs
generate_apidocs(SparsePauliOp)
