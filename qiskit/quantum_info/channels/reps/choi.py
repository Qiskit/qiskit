# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from numbers import Number
import numpy as np
from .baserep import QChannelRep
from .utils import reravel


class Choi(QChannelRep):
    """Choi-matrix representation of a quantum channel"""

    def __init__(self, data, input_dim=None, output_dim=None):
        choi_mat = np.array(data, dtype=complex)
        # Determine input and output dimensions
        dl, dr = choi_mat.shape
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
            raise ValueError("Invalid input and output dimension for Choi-matrix input.")
        super().__init__('Choi', choi_mat, input_dim, output_dim)

    def conjugate_channel(self):
        """Return the conjugate channel"""
        return Choi(np.conj(self._data), self.input_dim, self.output_dim)

    def transpose_channel(self):
        """Return the transpose channel"""
        # Swap input output indicies (need bipartite function)
        di = self.input_dim
        do = self.output_dim
        # Make bipartite matrix
        tensor = np.reshape(self._data, (di, do, di, do))
        # Swap input and output indicies
        tensor = np.transpose(tensor, (1, 0, 3, 2))
        # Transpose channel has input and output dimensions swapped
        data = np.reshape(tensor, (di * di, di * di))
        return Choi(data, do, di)

    def adjoint_channel(self):
        """Return the adjoint channel"""
        return self.conjugate_channel().transpose_channel()

    def kron(self, b):
        """Return Choi matrix for channel kron(A, B)

        Args:
            b (Choi): Choi for channel B

        Returns:
            Choi: for composite channel kron(A, B)

        Raises:
            TypeError: if b is not a Choi objects
        """
        if not isinstance(b, Choi):
            raise TypeError('Input channels must Choi')
        # Reshuffle indicies
        da_out, da_in = self.shape
        db_out, db_in = b.shape
        # Bipartite matrix shapes
        shape_a = (da_in, da_out, da_in, da_out)
        shape_b = (db_in, db_out, db_in, db_out)
        return Choi(reravel(self.data, b.data, shape_a, shape_b),
                    input_dim=da_in * db_in, output_dim=da_out * db_out)

    # Overloads
    def __mul__(self, other):
        if not isinstance(other, Number):
            raise TypeError("Not a number")
        return Choi(other * self.data, self.input_dim, self.output_dim)

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        if not isinstance(other, Choi):
            raise TypeError('Other channel must be a {}'.format(Choi))
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        return Choi(self.data + other.data, self.input_dim, self.output_dim)

    def __sub__(self, other):
        if not isinstance(other, Choi):
            raise TypeError('Other channel must be a {}'.format(Choi))
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        return Choi(self.data - other.data, self.input_dim, self.output_dim)

    def __neg__(self):
        return Choi(-self.data, self.input_dim, self.output_dim)

    # Assignment overloads
    def __imul__(self, other):
        if not isinstance(other, Number):
            raise TypeError("Not a number")
        self._data *= other

    def __itruediv__(self, other):
        self.__imul__(1 / other)

    def __iadd__(self, other):
        if not isinstance(other, Choi):
            raise TypeError('Other channel must be a {}'.format(Choi))
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        self._data += other.data

    def __isub__(self, other):
        if not isinstance(other, Choi):
            raise TypeError('Other channel must be a {}'.format(Choi))
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        self._data -= other.data
