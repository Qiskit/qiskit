# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from numbers import Number
import numpy as np
from .baserep import QChannelRep
from .utils import reravel


class PauliTM(QChannelRep):
    """Pauli transfer matrix representation of a quantum channel.

    The PTM is the Pauli-basis representation of the PauliTM.
    """

    def __init__(self, data, input_dim=None, output_dim=None):
        # Should we force this to be real?
        ptm = np.array(data, dtype=complex)
        # Determine input and output dimensions
        dout, din = ptm.shape
        if output_dim is None:
            output_dim = int(np.sqrt(dout))
        if input_dim is None:
            input_dim = int(np.sqrt(din))
        # Check dimensions
        if output_dim ** 2 != dout or input_dim ** 2 != din or input_dim != output_dim:
            raise ValueError("Invalid input and output dimension for Pauli transfer matrix input.")
        nqubits = int(np.log2(input_dim))
        if 2 ** nqubits != input_dim:
            raise ValueError("Input is not an n-qubit Pauli transfer matrix.")
        super().__init__('PauliTM', ptm, input_dim, output_dim)

    def compose(self, b):
        """Return PauliTM for composition channel A.B

        Args:
            b (PauliTM): PauliTM for channel B

        Returns:
            PauliTM: The PauliTM for composition channel A(B(rho))

        Raises:
            TypeError: if b is not a PauliTM object
        """
        if not isinstance(b, PauliTM):
            raise TypeError('Input channels must PauliTMs')
        # check dimensions are compatible
        if self.input_dim != b.output_dim:
            raise ValueError('input_dim of `a` must match output_dim of `b`')
        return PauliTM(np.dot(self.data, b.data), nput_dim=b.input_dim, output_dim=self.output_dim)

    def kron(self, b):
        """Return PauliTM for channel A \otimes B

        Args:
            b (PauliTM): channel B

        Returns:
            PauliTM: for the composite channel A \otimes B

        Raises:
            TypeError: if b is not a PauliTM object
        """
        if not isinstance(b, PauliTM):
            raise TypeError('Input channels must Pauli transfer matrices')
        # Reshuffle indicies
        da_out, da_in = self.shape
        db_out, db_in = b.shape
        # Bipartite matrix shapes
        shape_a = (da_out, da_out, db_out, db_out)
        shape_b = (da_in, da_in, db_in, db_in)
        return PauliTM(reravel(self.data, b.data, shape_a, shape_b),
                       input_dim=da_in * db_in, output_dim=da_out * db_out)

    # Overloads
    def __matmul__(self, other):
        return self.compose(other)

    def __mul__(self, other):
        if isinstance(other, Number):
            return PauliTM(other * self.data, self.input_dim, self.output_dim)
        else:
            raise TypeError("Not a number")

    def __truediv__(self, other):
        return self.__mul__(1 / other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __pow__(self, power):
        if not isinstance(power, int) or power < 1:
            raise ValueError("Can only exponentiate with positive integer powers.")
        if self.input_dim != self.output_dim:
            raise ValueError("Can only exponentiate with input_dim = output_dim.")
        data = self.data
        for j in range(power - 1):
            data = np.dot(data, self.data)
        return PauliTM(data, self.input_dim, self.output_dim)

    def __add__(self, other):
        if not isinstance(other, PauliTM):
            raise TypeError('Other channel must be a PauliTM')
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        return PauliTM(self.data + other.data, self.input_dim, self.output_dim)

    def __sub__(self, other):
        if not isinstance(other, PauliTM):
            raise TypeError('Other channel must be a PauliTM')
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        return PauliTM(self.data - other.data, self.input_dim, self.output_dim)

    def __neg__(self):
        return PauliTM(-self.data, self.input_dim, self.output_dim)

    # Assignment overloads
    def __imatmul__(self, other):
        if not isinstance(other, PauliTM):
            raise TypeError('Other channel must be a PauliTM')
        # check dimensions are compatible
        if other.output_dim != self.input_dim:
            raise ValueError('output_dim of other channel does not match input_dim.')
        self._data = np.dot(self.data, other.data)
        self._input_dim = other.input_dim

    def __imul__(self, other):
        if isinstance(other, Number):
            self._data *= other
        else:
            raise TypeError("Not a number")

    def __itruediv__(self, other):
        self.__imul__(1 / other)

    def __ipow__(self, power):
        if not isinstance(power, int) or power < 1:
            raise ValueError("Can only exponentiate with positive integer powers.")
        if self.input_dim != self.output_dim:
            raise ValueError("Can only exponentiate with input_dim = output_dim.")
        data = self.data
        for j in range(power - 1):
            self._data = np.dot(self.data, data)

    def __iadd__(self, other):
        if not isinstance(other, PauliTM):
            raise TypeError('Other channel must be a PauliTM')
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        self._data += other.data

    def __isub__(self, other):
        if not isinstance(other, PauliTM):
            raise TypeError('Other channel must be a PauliTM')
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        self._data -= other.data
