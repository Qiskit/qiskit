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
# pylint: disable=invalid-name

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
        # Shape lists the dimension of each subsystem starting from
        # least significant through to most significant.
        self._dims = tuple(dims)
        self._dim = np.product(dims)

    def __eq__(self, other):
        if (isinstance(other, self.__class__)
                and self.dims() == other.dims()):
            return np.allclose(
                self.data, other.data, rtol=self._rtol, atol=self._atol)
        return False

    def __repr__(self):
        return '{}({}, dims={})'.format(
            self.rep, self.data, self._dims)

    def _repr_latex_(self):
        """
        Generate a LaTeX representation of the Quantum State.

        Copyright (c) 2011 and later, Paul D. Nation and Robert J. Johansson.
        """
        shape = self._data.shape
        atol = 1e-15
        s = r''

        s += ("{} ".format(self.rep) +
              "dims = {}".format(str(self._dims)) +
              ", shape = {}".format(str(shape))
             )
        if len(shape) == 2:
            M, N = shape
            flat = False
        else:
            M = shape[0]
            N = 1
            flat = True

        s += r'\begin{equation*}\left(\begin{array}{*{11}c}'

        def _format_float(value):
            if value == 0.0:
                return "0.0"
            elif abs(value) > 1000.0 or abs(value) < 0.001:
                return ("%.3e" % value).replace("e", r"\times10^{") + "}"
            elif abs(value - int(value)) < 0.001:
                return "%.1f" % value
            else:
                return "%.3f" % value

        def _format_element(_, n, d):
            s = " & " if n > 0 else ""
            if isinstance(d, str):
                return s + d
            else:
                if abs(np.imag(d)) < atol:
                    return s + _format_float(np.real(d))
                elif abs(np.real(d)) < atol:
                    return s + _format_float(np.imag(d)) + "j"
                else:
                    s_re = _format_float(np.real(d))
                    s_im = _format_float(np.imag(d))
                    if np.imag(d) > 0.0:
                        return s + "(" + s_re + "+" + s_im + "j)"
                    else:
                        return s + "(" + s_re + s_im + "j)"

        if M > 10 and N > 10:
            # truncated matrix output
            for m in range(5):
                for n in range(5):
                    s += _format_element(m, n, self.data[m, n])
                s += r' & \cdots'
                for n in range(N - 5, N):
                    s += _format_element(m, n, self.data[m, n])
                s += r'\\'

            for n in range(5):
                s += _format_element(m, n, r'\vdots')
            s += r' & \ddots'
            for n in range(N - 5, N):
                s += _format_element(m, n, r'\vdots')
            s += r'\\'

            for m in range(M - 5, M):
                for n in range(5):
                    s += _format_element(m, n, self.data[m, n])
                s += r' & \cdots'
                for n in range(N - 5, N):
                    s += _format_element(m, n, self.data[m, n])
                s += r'\\'

        elif M > 10 and N <= 10:
            # truncated vertically elongated matrix output
            for m in range(5):
                for n in range(N):
                    if flat:
                        _data = self.data[m]
                    else:
                        _data = self.data[m, n]
                    s += _format_element(m, n, _data)
                s += r'\\'

            for n in range(N):
                s += _format_element(m, n, r'\vdots')
            s += r'\\'

            for m in range(M - 5, M):
                for n in range(N):
                    s += _format_element(m, n, self.data[m, n])
                s += r'\\'

        elif M <= 10 and N > 10:
            # truncated horizontally elongated matrix output
            for m in range(M):
                for n in range(5):
                    s += _format_element(m, n, self.data[m, n])
                s += r' & \cdots'
                for n in range(N - 5, N):
                    s += _format_element(m, n, self.data[m, n])
                s += r'\\'

        else:
            # full output
            for m in range(M):
                for n in range(N):
                    if flat:
                        _data = self.data[m]
                    else:
                        _data = self.data[m, n]
                    s += _format_element(m, n, _data)
                s += r'\\'

        s += r'\end{array}\right)\end{equation*}'
        return s

    @property
    def rep(self):
        """Return state representation string."""
        return self._rep

    @property
    def dim(self):
        """Return total state dimension."""
        return self._dim

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

    @classmethod
    def _automatic_dims(cls, dims, size):
        """Check if input dimension corresponds to qubit subsystems."""
        return BaseOperator._automatic_dims(dims, size)

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
