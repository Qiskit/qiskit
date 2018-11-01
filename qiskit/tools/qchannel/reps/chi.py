# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from numbers import Number
import numpy as np
from .baserep import QChannelRep
from .utils import reravel


class Chi(QChannelRep):
    """Pauli basis Chi-matrix representation of a quantum channel

    The Chi-matrix is the Pauli-basis representation of the Chi-Matrix.
    """

    def __init__(self, data, input_dim=None, output_dim=None):
        chi_mat = np.array(data, dtype=complex)
        # Determine input and output dimensions
        dl, dr = chi_mat.shape
        if dl != dr:
            raise ValueError('Invalid Choi-matrix input.')
        if output_dim is None and input_dim is None:
            output_dim = int(np.sqrt(dl))
            input_dim = dl // output_dim
        elif input_dim is None:
            input_dim = dl // output_dim
        elif output_dim is None:
            output_dim = dl // input_dim
        # Check dimensions
        if input_dim * output_dim != dl:
            raise ValueError("Invalid input and output dimension for Chi-matrix input.")
        nqubits = int(np.log2(self.input_dim))
        if 2 ** nqubits != input_dim:
            raise ValueError("Input is not an n-qubit Chi matrix.")
        super().__init__('Chi', chi_mat, input_dim, output_dim)

    def kron(self, b):
        """Return Chi matrix for channel A \otimes B

        Args:
            b (Chi): Chi for channel B

        Returns:
            Choi: for composite channel A \otimes B

        Raises:
            TypeError: if b is not a Chi objects
        """
        if not isinstance(b, Chi):
            raise TypeError('Input channels must Chi')
        # Reshuffle indicies
        da_out, da_in = self.shape
        db_out, db_in = b.shape
        # Bipartite matrix shapes
        shape_a = (da_in, da_out, da_in, da_out)
        shape_b = (db_in, db_out, db_in, db_out)
        return Chi(reravel(self.data, b.data, shape_a, shape_b),
                   input_dim=da_in * db_in, output_dim=da_out * db_out)

    # Overloads
    def __mul__(self, other):
        if not isinstance(other, Number):
            raise TypeError("Not a number")
        return Chi(other * self.data, self.input_dim, self.output_dim)

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if not isinstance(other, Chi):
            raise TypeError('Other channel must be a {}'.format(Chi))
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        return Chi(self.data + other.data, self.input_dim, self.output_dim)

    def __sub__(self, other):
        if not isinstance(other, Chi):
            raise TypeError('Other channel must be a {}'.format(Chi))
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        return Chi(self.data - other.data, self.input_dim, self.output_dim)

    def __neg__(self):
        return Chi(-self.data, self.input_dim, self.output_dim)

    # Assignment overloads
    def __imul__(self, other):
        if not isinstance(other, Number):
            raise TypeError("Not a number")
        self._data *= other

    def __itruediv__(self, other):
        self.__imul__(1 / other)

    def __iadd__(self, other):
        if not isinstance(other, Chi):
            raise TypeError('Other channel must be a {}'.format(Chi))
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        self._data += other.data

    def __isub__(self, other):
        if not isinstance(other, Chi):
            raise TypeError('Other channel must be a {}'.format(Chi))
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        self._data -= other.data
