# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

# pylint: disable=invalid-name,assignment-from-no-return
"""
Tools for working with Unitary Operators.

A simple unitary class and some tools.
"""
import copy
import numpy
import sympy

from qiskit.circuit.gate import Gate
from qiskit.quantum_info.synthesis.two_qubit_kak import two_qubit_kak
from qiskit.exceptions import QiskitError


class Unitary(Gate):
    """Class for representing unitary operators"""

    def __init__(self,
                 representation,
                 label=None,
                 validate=True,
                 rtol=1e-5,
                 atol=1e-8):
        """
        Create unitary.

        Args:
            representation (numpy.ndarray or list(list) or Unitary): unitary representation
            label (str): identifier hint for backend
            validate (bool): whether to validate unitarity of matrix
            rtol (float): relative tolerance (see numpy.allclose)
            atol (float): absolute tolerance (see numpy.allclose)
        """
        self.__validate = validate
        self.__rtol = rtol
        self.__atol = atol
        # set attributes for use by other methods
        self.__representation = None
        self.__n_qubits = None
        self.__label = None
        self.__dimension = None
        # set representation (depends on previous attributes)
        if isinstance(representation, (numpy.ndarray, sympy.Matrix, list)):
            self._representation = representation
            super().__init__('unitary', self.__n_qubits,
                             [sympy.Matrix(representation)])
        elif isinstance(representation, Unitary):
            for attrib, value in vars(representation).items():
                setattr(self, attrib, value)
        if isinstance(label, str):
            self._label = label
        self._decompostion = []  # storage (needed?)

    def __eq__(self, other):
        if not isinstance(other, Unitary):
            return NotImplemented
        for attrib, value in vars(other).items():
            if isinstance(value, numpy.ndarray):
                if not numpy.array_equal(value, getattr(self, attrib)):
                    return False
            elif isinstance(value, sympy.Matrix):
                if not numpy.array_equal(value, getattr(self, attrib)):
                    return False
            elif getattr(self, attrib) != getattr(other, attrib):
                return False
        return True

    def __str__(self):
        return str(self.representation)

    def __repr__(self):
        return '{}\n{}'.format(super().__repr__(), self.__representation.__repr__())

    def _define(self):
        """Calculate a subcircuit that implements this unitary.
        """
        if self.__dimension == 4:
            self.definition = two_qubit_kak(self)

    @property
    def dimension(self):
        """dimension of matrix

        Returns:
            int: dimension of matrix
        """
        return self.__dimension

    def tensor(self, other):
        """
        tensor product

        A.tensor(B) = A⊗B

        Args:
            other (Unitary): unitary to tensor

        Returns:
            Unitary: unitary object
        """
        dim = self.dimension + other.dimension
        output = Unitary(
            numpy.empty((dim, dim), dtype='complex'), validate=False)
        output._representation = numpy.kron(self.representation,
                                            other.representation)
        return output

    def expand(self, other):
        """
        tensor product with reverse order

        A.expand(B) = B⊗A

        Args:
            other (Unitary): unitary to tensor

        Returns:
            Unitary: unitary object
        """
        dim = self.dimension + other.dimension
        output = Unitary(
            numpy.empty((dim, dim), dtype='complex'), validate=False)
        output._representation = numpy.kron(other.representation,
                                            self.representation)
        return output

    def compose(self, other, front=False):
        """
        Compose unitary with other.

        A.compose(B) = A(B)
        A.compose(B, front=True) = B(A)

        Args:
            other (Unitary): unitary to compose with
            front (bool): Whether the other matrix is in front.

        Returns:
            Unitary: unitary object
        """
        output = copy.deepcopy(self)
        if front:
            numpy.matmul(
                other.representation,
                output.representation,
                out=output.representation)
        else:
            numpy.matmul(
                output.representation,
                other.representation,
                out=output.representation)
        return output

    def conjugate(self):
        """Return the conjugate of the Unitary."""
        output = copy.deepcopy(self)
        numpy.conj(output.representation, out=output.representation)
        return output

    def adjoint(self):
        """Return the adjoint of the unitary."""
        return self.transpose().conjugate()

    def transpose(self):
        """Return the transpose of the unitary."""
        output = copy.deepcopy(self)
        output._representation = self.representation.transpose()
        return output

    @property
    def representation(self):
        """
        Get representation. Currently this is just a unitary matrix.

        Returns:
            numpy.ndarray: matrix

        Raises:
            QiskitError: if representation not defined
        """
        if self.__representation is not None:
            return self.__representation
        else:
            raise QiskitError("representation not defined")

    @representation.setter
    def _representation(self, representation):
        """set matrix representation

        Args:
            representation (ndarray | list(list) | Unitary): representation to set

        Raises:
            QiskitError: if representation is not list, ndarray, or Unitary
        """
        if isinstance(representation, (numpy.ndarray, list, sympy.Matrix)):
            mat = numpy.asarray(representation, dtype='complex')
            if self.__validate:
                if not numpy.allclose(
                        mat.T.conj() @ mat,
                        numpy.identity(mat.shape[0]),
                        rtol=self.__rtol,
                        atol=self.__atol):
                    raise QiskitError("matrix is not unitary")
            self.__representation = mat
            self.__n_qubits = int(numpy.log2(mat.shape[0]))
            self.__dimension = mat.shape[0]
        elif representation is None:
            self.__representation = None  # for creating empty Unitary
        else:
            raise QiskitError('unrecognized unitary representation: {}'.format(
                type(representation)))

    def power(self, n):
        """Return the compose of a Unitary with itself n times.

        Args:
            n (int): the number of times to compose with self (n>0).

        Returns:
            Unitary: the n-times composition channel.
        """
        dim = self.dimension
        uni = Unitary(numpy.empty((dim, dim), dtype='complex'), validate=False)
        if n >= 0:
            uni._representation = numpy.linalg.matrix_power(
                self.representation, n)
        else:
            uni._representation = numpy.linalg.matrix_power(
                self.representation.T.conj(), n)
        return uni

    @property
    def _label(self):
        """get unitary label

        Returns:
            str: object's label
        """
        return self.__label

    @_label.setter
    def _label(self, name):
        """set unitary label to name

        Args:
            name (str or None): label to assign unitary

        Raises:
            TypeError: name is not string or None.
        """
        if isinstance(name, (str, type(None))):
            self.__label = name
        else:
            raise TypeError('label expects a string or None')
