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
from qiskit.exceptions import QiskitError
from qiskit.circuit.gate import Gate
from qiskit.dagcircuit import DAGCircuit


class Unitary(Gate):
    """Class for representing unitary operators"""

    def __init__(self, representation, *qargs, label=None, validate=True, rtol=1e-5, atol=1e-8):
        """
        Create unitary.

        Args:
            representation (numpy.ndarray or list(list) or Unitary): unitary representation.
            qargs (list(tuple(QuantumCircuit, int))): qubits to apply operation to.
            label (str): identifier hint for backend
            validate (bool): whether to validate unitarity of matrix when supplied
                either here or with Unitary.matrix.
            rtol (float): relative tolerance (see numpy.allclose).
            atol (float): absolute tolerance (see numpy.allclose).
        """
        self.__validate = validate
        self.__rtol = rtol
        self.__atol = atol
        # set attributes for use by other methods
        self.__representation = None
        self.__n_qubits = None
        self.__label = None
        self.__dimension = None
        # set matrix (depends on previous attributes)
        if isinstance(representation, numpy.ndarray):
            self._representation = representation
        elif isinstance(representation, list):
            self._representation = numpy.asarray(representation)
        elif isinstance(represenation, Unitary):
            self = copy.deepcopy(representation)
        if isinstance(label, str):
            self._label = label
        self._decompostion = []  # storage (needed?)
        super().__init__('unitary', [sympy.Matrix(matrix)], list(qargs))

    @property
    def dimension(self):
        """dimension of matrix

        Returns:
            int: dimension of matrix
        """
        return self._dimension

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
        output = Unitary(numpy.empty((dim, dim), dtype='complex'), validate=False)
        output.matrix = numpy.kron(self.matrix, other.matrix)
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
        output = Unitary(numpy.empty((dim, dim), dtype='complex'), validate=False)
        output.matrix = numpy.kron(other.matrix, self.matrix)
        return output

    def compose(self, other, inplace=False, front=False):
        """
        Compose unitary with other.

        A.compose(B) = A(B)
        A.compose(B, front=True) = B(A)

        Args:
            other (Unitary): unitary to compose with
            inplace (bool): If true, the operation modifies the matrix of this
                unitary. If false, a new Unitary object is created with the result.
            front (bool): Whether the other matrix is in front.

        Returns:
            Unitary: unitary object
        """
        if inplace:
            if front:
                numpy.matmul(other.matrix, self.matrix, out=self.matrix)
            else:
                numpy.matmul(self.matrix, other.matrix, out=self.matrix)
            return self
        else:
            output = copy.deepcopy(self)
            if front:
                numpy.matmul(other.matrix, output.matrix, out=output.matrix)
            else:
                numpy.matmul(output.matrix, other.matrix, out=output.matrix)
            return output

    def conjugate(self, inplace=False):
        """conjugate of unitary

        Args:
            inplace (bool): whether to do conjugation of this object

        Returns:
            Unitary: unitary object
        """
        if inplace:
            numpy.conj(self.matrix, out=self.matrix)
            return self
        else:
            output = copy.deepcopy(self)
            numpy.conj(output.matrix, out=output.matrix)
            return output

    def adjoint(self, inplace=False):
        """
        Adjoint

        Args:
            inplace (bool): don't create new structure

        Returns:
            Unitary: transposed unitary
        """
        return self.transpose(inplace=inplace).conjugate(inplace=inplace)

    def transpose(self, inplace=False):
        """
        tranpose unitary

        Args:
            inplace (bool): attempt inplace (relies on numpy.transpose)

        Returns:
            Unitary: transposed unitary
        """
        if inplace:
            self.matrix = self.matrix.transpose()
            return self
        else:
            output = copy.deepcopy(self)
            output.matrix = self.matrix.transpose()
            return output

    @property
    def _representation(self):
        """
        Get representation

        Returns:
            numpy.ndarray: matrix

        Raises:
            QiskitError: if representation not defined
        """
        if self._representation is not None:
            return self._representation
        else:
            raise QiskitError("representation not defined")

    @_representation.setter
    def _representation(self, representation):
        """set matrix representation

        Args:
            representation (ndarray | list(list) | Unitary): representation to set

        Raises:
            QiskitError: if representation is not list, ndarray, or Unitary
        """
        if isinstance(representation, (numpy.ndarray, list)):
            mat = numpy.asarray(representation, dtype='complex')
            if self.__validate:
                if not numpy.allclose(mat.T.conj() @ mat, numpy.identity(mat.shape[0]),
                                      rtol=self.__rtol, atol=self.__atol):
                    raise QiskitError("matrix is not unitary")
            self._representation = mat
            self.__n_qubits = int(numpy.log2(mat.shape[0]))
            self.__dimension = mat.shape[0]
        elif representation is None:
            self._representation = None  # for creating empty Unitary
        else:
            raise QiskitError('unrecognized unitary representation: {}'.format(
                type(representation)))
        
    def power(self, n, inplace=False):
        """Return n-th matrix power.

        Args:
            n (int): integer power
            inplace (bool): whether to do operation in place.

        Returns:
            Unitary: Unitary power
        """
        if inplace:
            if n >= 0:
                self.matrix = numpy.linalg.matrix_power(self.matrix, n)
            else:
                self.matrix = numpy.linalg.matrix_power(
                    self.matrix.T.conj(), n)
            return self
        else:
            dim = self.dimension
            uni = Unitary(numpy.empty((dim, dim), dtype='complex'), validate=False)
            if n >= 0:
                uni.matrix = numpy.linalg.matrix_power(self.matrix, n)
            else:
                uni.matrix = numpy.linalg.matrix_power(
                    self.matrix.T.conj(), n)
            return uni

    def reapply(self, circ):
        """reapply this gate to circ; TODO: remove when monkey patching removed"""
        self._modifiers(circ._attach(self))

    def _define_decompositions(self):
        decomposition = DAGCircuit()
        self._decompositions = [decomposition]

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
