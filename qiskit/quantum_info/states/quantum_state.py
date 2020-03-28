# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Abstract QuantumState class.
"""

from abc import ABC, abstractmethod

import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT


class QuantumState(ABC):
    """Abstract quantum state base class"""

    ATOL = ATOL_DEFAULT
    RTOL = RTOL_DEFAULT
    MAX_TOL = 1e-4

    def __init__(self, rep, data, dims):
        """Initialize a state object."""
        if not isinstance(rep, str):
            raise QiskitError("rep must be a string not a {}".format(
                rep.__class__))
        self._rep = rep
        self._data = data

        # Dimension attributes
        # Note that the tuples of input and output dims are ordered
        # from least-significant to most-significant subsystems
        self._dims = None        # tuple of dimensions of each subsystem
        self._dim = None         # combined dimension of all subsystems
        self._num_qubits = None  # number of qubit subsystems if N-qubit state
        self._set_dims(dims)

    def __eq__(self, other):
        if (isinstance(other, self.__class__)
                and self.dims() == other.dims()):
            return np.allclose(
                self.data, other.data, rtol=self._rtol, atol=self._atol)
        return False

    def __repr__(self):
        prefix = '{}('.format(self.rep)
        pad = len(prefix) * ' '
        return '{}{},\n{}dims={})'.format(
            prefix, np.array2string(
                self.data, separator=', ', prefix=prefix),
            pad, self._dims)

    @property
    def rep(self):
        """Return state representation string."""
        return self._rep

    @property
    def dim(self):
        """Return total state dimension."""
        return self._dim

    @property
    def num_qubits(self):
        """Return the number of qubits if a N-qubit state or None otherwise."""
        return self._num_qubits

    @property
    def data(self):
        """Return data."""
        return self._data

    @property
    def _atol(self):
        """The absolute tolerance parameter for float comparisons."""
        return self.__class__.ATOL

    @_atol.setter
    def _atol(self, atol):
        """Set the absolute tolerance parameter for float comparisons."""
        # NOTE: that this overrides the class value so applies to all
        # instances of the class.
        max_tol = self.__class__.MAX_TOL
        if atol < 0:
            raise QiskitError("Invalid atol: must be non-negative.")
        if atol > max_tol:
            raise QiskitError(
                "Invalid atol: must be less than {}.".format(max_tol))
        self.__class__.ATOL = atol

    @property
    def _rtol(self):
        """The relative tolerance parameter for float comparisons."""
        return self.__class__.RTOL

    @_rtol.setter
    def _rtol(self, rtol):
        """Set the relative tolerance parameter for float comparisons."""
        # NOTE: that this overrides the class value so applies to all
        # instances of the class.
        max_tol = self.__class__.MAX_TOL
        if rtol < 0:
            raise QiskitError("Invalid rtol: must be non-negative.")
        if rtol > max_tol:
            raise QiskitError(
                "Invalid rtol: must be less than {}.".format(max_tol))
        self.__class__.RTOL = rtol

    def _reshape(self, dims=None):
        """Reshape dimensions of the state.

        Arg:
            dims (tuple): new subsystem dimensions.

        Returns:
            self: returns self with reshaped dimensions.

        Raises:
            QiskitError: if combined size of all subsystem dimensions are not constant.
        """
        if dims is not None:
            if np.product(dims) != self._dim:
                raise QiskitError(
                    "Reshaped dims are incompatible with combined dimension."
                )
            self._dims = tuple(dims)
        return self

    def dims(self, qargs=None):
        """Return tuple of input dimension for specified subsystems."""
        if qargs is None:
            return self._dims
        return tuple(self._dims[i] for i in qargs)

    def copy(self):
        """Make a copy of current operator."""
        # pylint: disable=no-value-for-parameter
        # The constructor of subclasses from raw data should be a copy
        return self.__class__(self.data, self.dims())

    @abstractmethod
    def is_valid(self, atol=None, rtol=None):
        """Return True if a valid quantum state."""
        pass

    @abstractmethod
    def to_operator(self):
        """Convert state to matrix operator class"""
        pass

    @abstractmethod
    def conjugate(self):
        """Return the conjugate of the operator."""
        pass

    @abstractmethod
    def trace(self):
        """Return the trace of the quantum state as a density matrix."""
        pass

    @abstractmethod
    def purity(self):
        """Return the purity of the quantum state."""
        pass

    @abstractmethod
    def tensor(self, other):
        """Return the tensor product state self ⊗ other.

        Args:
            other (QuantumState): a quantum state object.

        Returns:
            QuantumState: the tensor product operator self ⊗ other.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        pass

    @abstractmethod
    def expand(self, other):
        """Return the tensor product state other ⊗ self.

        Args:
            other (QuantumState): a quantum state object.

        Returns:
            QuantumState: the tensor product state other ⊗ self.

        Raises:
            QiskitError: if other is not a quantum state.
        """
        pass

    @abstractmethod
    def add(self, other):
        """Return the linear combination self + other.

        Args:
            other (QuantumState): a quantum state object.

        Returns:
            LinearOperator: the linear combination self + other.

        Raises:
            QiskitError: if other is not a quantum state, or has
                         incompatible dimensions.
        """
        pass

    @abstractmethod
    def subtract(self, other):
        """Return the linear operator self - other.

        Args:
            other (QuantumState): a quantum state object.

        Returns:
            LinearOperator: the linear combination self - other.

        Raises:
            QiskitError: if other is not a quantum state, or has
                         incompatible dimensions.
        """
        pass

    @abstractmethod
    def multiply(self, other):
        """Return the linear operator self * other.

        Args:
            other (complex): a complex number.

        Returns:
            Operator: the linear combination other * self.

        Raises:
            QiskitError: if other is not a valid complex number.
        """
        pass

    @abstractmethod
    def evolve(self, other, qargs=None):
        """Evolve a quantum state by the operator.

        Args:
            other (Operator or QuantumChannel): The operator to evolve by.
            qargs (list): a list of QuantumState subsystem positions to apply
                           the operator on.

        Returns:
            QuantumState: the output quantum state.

        Raises:
            QiskitError: if the operator dimension does not match the
                         specified QuantumState subsystem dimensions.
        """
        pass

    @abstractmethod
    def probabilities(self, qargs=None, decimals=None):
        """Return the subsystem measurement probability vector.

        Measurement probabilities are with respect to measurement in the
        computation (diagonal) basis.

        Args:
            qargs (None or list): subsystems to return probabilities for,
                if None return for all subsystems (Default: None).
            decimals (None or int): the number of decimal places to round
                values. If None no rounding is done (Default: None).

        Returns:
            np.array: The Numpy vector array of probabilities.
        """
        pass

    def probabilities_dict(self, qargs=None, decimals=None):
        """Return the subsystem measurement probability dictionary.

        Measurement probabilities are with respect to measurement in the
        computation (diagonal) basis.

        This dictionary representation uses a Ket-like notation where the
        dictionary keys are qudit strings for the subsystem basis vectors.
        If any subsystem has a dimension greater than 10 comma delimiters are
        inserted between integers so that subsystems can be distinguished.

        Args:
            qargs (None or list): subsystems to return probabilities for,
                if None return for all subsystems (Default: None).
            decimals (None or int): the number of decimal places to round
                values. If None no rounding is done (Default: None).

        Returns:
            dict: The measurement probabilities in dict (ket) form.
        """
        return self._vector_to_dict(
            self.probabilities(qargs=qargs, decimals=decimals),
            self.dims(qargs),
            string_labels=True)

    @classmethod
    def _automatic_dims(cls, dims, size):
        """Check if input dimension corresponds to qubit subsystems."""
        return BaseOperator._automatic_dims(dims, size)

    def _set_dims(self, dims):
        """Set dimension attribute"""
        # Shape lists the dimension of each subsystem starting from
        # least significant through to most significant.
        self._dims = tuple(dims)
        # The total input and output dimensions are given by the product
        # of all subsystem dimensions
        self._dim = np.product(dims)
        # Check if an N-qubit operator
        if set(self._dims) == set([2]):
            # If so set the number of qubits
            self._num_qubits = len(self._dims)
        else:
            # Otherwise set the number of qubits to None
            self._num_qubits = None

    @staticmethod
    def _index_to_ket_array(inds, dims, string_labels=False):
        """Convert an index array into a ket array.

        Args:
            inds (np.array): an integer index array.
            dims (tuple): a list of subsystem dimensions.
            string_labels (bool): return ket as string if True, otherwise
                                  return as index array (Default: False).

        Returns:
            np.array: an array of ket strings if string_label=True, otherwise
                      an array of ket lists.
        """
        shifts = [1]
        for dim in dims[:-1]:
            shifts.append(shifts[-1] * dim)
        kets = np.array([(inds // shift) % dim for dim, shift in zip(dims, shifts)])

        if string_labels:
            max_dim = max(dims)
            char_kets = np.asarray(kets, dtype=np.unicode_)
            str_kets = char_kets[0]
            for row in char_kets[1:]:
                if max_dim > 10:
                    str_kets = np.char.add(',', str_kets)
                str_kets = np.char.add(row, str_kets)
            return str_kets.T

        return kets.T

    @staticmethod
    def _vector_to_dict(vec, dims, decimals=None, string_labels=False):
        """Convert a vector to a ket dictionary.

        This representation will not show zero values in the output dict.

        Args:
            vec (array): a Numpy vector array.
            dims (tuple): subsystem dimensions.
            decimals (None or int): number of decimal places to round to.
                                    (See Numpy.round), if None no rounding
                                    is done (Default: None).
            string_labels (bool): return ket as string if True, otherwise
                                  return as index array (Default: False).

        Returns:
            dict: the vector in dictionary `ket` form.
        """
        # Get indices of non-zero elements
        vals = vec if decimals is None else vec.round(decimals=decimals)
        inds, = vals.nonzero()

        # Convert to ket tuple based on subsystem dimensions
        kets = QuantumState._index_to_ket_array(
            inds, dims, string_labels=string_labels)

        # Make dict of tuples
        if string_labels:
            return dict(zip(kets, vec[inds]))

        return {tuple(ket): val for ket, val in zip(kets, vals[inds])}

    @staticmethod
    def _matrix_to_dict(mat, dims, decimals=None, string_labels=False):
        """Convert a matrix to a ket dictionary.

        This representation will not show zero values in the output dict.

        Args:
            mat (array): a Numpy matrix array.
            dims (tuple): subsystem dimensions.
            decimals (None or int): number of decimal places to round to.
                                    (See Numpy.round), if None no rounding
                                    is done (Default: None).
            string_labels (bool): return ket as string if True, otherwise
                                  return as index array (Default: False).

        Returns:
            dict: the matrix in dictionary `ket` form.
        """
        # Get indices of non-zero elements
        vals = mat if decimals is None else mat.round(decimals=decimals)
        inds_row, inds_col, = vals.nonzero()

        # Convert to ket tuple based on subsystem dimensions
        bras = QuantumState._index_to_ket_array(
            inds_row, dims, string_labels=string_labels)
        kets = QuantumState._index_to_ket_array(
            inds_col, dims, string_labels=string_labels)

        # Make dict of tuples
        if string_labels:
            return {'{}|{}'.format(ket, bra): val for ket, bra, val in zip(
                kets, bras, vals[inds_row, inds_col])}

        return {(tuple(ket), tuple(bra)): val for ket, bra, val in zip(
            kets, bras, vals[inds_row, inds_col])}

    @staticmethod
    def _accumulate_dims(dims, qargs):
        """Flatten subsystem dimensions for unspecified qargs.

        This has the potential to reduce the number of subsystems
        by combining consecutive subsystems between the specified
        qargs. For example, if we had a 5-qubit system with
        ``dims = (2, 2, 2, 2, 2)``, and ``qargs=[0, 4]``, then the
        flattened system will have dimensions ``new_dims = (2, 8, 2)``
        and qargs ``new_qargs = [0, 2]``.

        Args:
            dims (tuple): subsystem dimensions.
            qargs (list): qargs list.

        Returns:
            tuple: the pair (new_dims, new_qargs).
        """

        qargs_map = {}
        new_dims = []

        # Accumulate subsystems that can be combined
        accum = []
        for i, dim in enumerate(dims):
            if i in qargs:
                if accum:
                    new_dims.append(np.product(accum))
                    accum = []
                new_dims.append(dim)
                qargs_map[i] = len(new_dims) - 1
            else:
                accum.append(dim)
        if accum:
            new_dims.append(np.product(accum))
        return tuple(new_dims), [qargs_map[i] for i in qargs]

    @staticmethod
    def _subsystem_probabilities(probs, dims, qargs=None):
        """Marginalize a probability vector according to subsystems.

        Args:
            probs (np.array): a probability vector Numpy array.
            dims (tuple): subsystem dimensions.
            qargs (None or list): a list of subsystems to return
                marginalized probabilities for. If None return all
                probabilities (Default: None).

        Returns:
            np.array: the marginalized probability vector flattened
                      for the specified qargs.
        """

        if qargs is None:
            return probs

        # Accumulate dimensions to trace over
        accum_dims, accum_qargs = QuantumState._accumulate_dims(
            dims, qargs)

        # Get sum axis for maginalized subsystems
        n_qargs = len(accum_dims)
        axis = list(range(n_qargs))
        for i in accum_qargs:
            axis.remove(n_qargs - 1 - i)

        # Reshape the probability to a tensor and sum over maginalized axes
        new_probs = np.sum(np.reshape(probs, list(reversed(accum_dims))),
                           axis=tuple(axis))

        # Transpose output probs based on order of qargs
        if sorted(accum_qargs) != accum_qargs:
            axes = np.argsort(accum_qargs)
            return np.ravel(np.transpose(new_probs, axes=axes))

        return np.ravel(new_probs)

    # Overloads
    def __matmul__(self, other):
        # Check for subsystem case return by __call__ method
        if isinstance(other, tuple) and len(other) == 2:
            return self.evolve(other[0], qargs=other[1])
        return self.evolve(other)

    def __xor__(self, other):
        return self.tensor(other)

    def __mul__(self, other):
        return self.multiply(other)

    def __truediv__(self, other):
        return self.multiply(1 / other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return self.add(other)

    def __sub__(self, other):
        return self.subtract(other)

    def __neg__(self):
        return self.multiply(-1)
