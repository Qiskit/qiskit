# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

from numbers import Number
import numpy as np

from .baserep import QChannelRep
from .utils import reravel


class SuperOp(QChannelRep):
    """Superoperator representation of a quantum channel"""

    def __init__(self, data, input_dim=None, output_dim=None):
        super_mat = np.array(data, dtype=complex)
        # Determine input and output dimensions
        dout, din = super_mat.shape
        if output_dim is None:
            output_dim = int(np.sqrt(dout))
        if input_dim is None:
            input_dim = int(np.sqrt(din))
        # Check dimensions
        if output_dim ** 2 != dout or input_dim ** 2 != din:
            raise ValueError("Invalid input and output dimension for superoperator input.")
        super().__init__('SuperOp', super_mat, input_dim, output_dim)

    def conjugate_channel(self):
        """Return the conjugate channel"""
        return SuperOp(np.conj(self._data), self.input_dim, self.output_dim)

    def transpose_channel(self):
        """Return the transpose channel"""
        # Swaps input and output dimensions
        dout = self.input_dim
        din = self.output_dim
        data = np.transpose(self._data)
        return SuperOp(data, din, dout)

    def adjoint_channel(self):
        """Return the adjoint channel"""
        return self.conjugate_channel().transpose_channel()

    def compose(self, b):
        """Return SuperOp for the composition channel A.B

        Args:
            b (SuperOp): channel B

        Returns:
            SuperOp: The SuperOp for the composition channel A(B(rho))

        Raises:
            TypeError: if b is not a SuperOp object
        """
        if not isinstance(b, SuperOp):
            raise TypeError('Input channels must SuperOps')
        # check dimensions are compatible
        if self.input_dim != b.output_dim:
            raise ValueError('input_dim of `a` must match output_dim of `b`')
        return SuperOp(np.dot(self.data, b.data),
                       input_dim=b.input_dim,
                       output_dim=self.output_dim)

    def kron(self, b):
        """Return SuperOp for the channel A \otimes B

        Args:
            b (SuperOp): channel B

        Returns:
            SuperOp: for the composite channel A \otimes B

        Raises:
            TypeError: if b is not a SuperOp object
        """
        if not isinstance(b, SuperOp):
            raise TypeError('Input channels must SuperOps')
        # Reshuffle indicies
        da_out, da_in = self.shape
        db_out, db_in = b.shape
        # Bipartite matrix shapes
        shape_a = (da_out, da_out, db_out, db_out)
        shape_b = (da_in, da_in, db_in, db_in)
        return SuperOp(reravel(self.data, b.data, shape_a, shape_b),
                       input_dim=da_in * db_in, output_dim=da_out * db_out)

    # Overloads
    def __matmul__(self, other):
        return self.compose(other)

    def __mul__(self, other):
        if not isinstance(other, Number):
            raise TypeError("Not a number")
        return SuperOp(other * self.data, self.input_dim, self.output_dim)

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
        return SuperOp(data, self.input_dim, self.output_dim)

    def __add__(self, other):
        if not isinstance(other, SuperOp):
            raise TypeError('Other channel must be a SuperOp')
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        return SuperOp(self.data + other.data, self.input_dim, self.output_dim)

    def __sub__(self, other):
        if not isinstance(other, SuperOp):
            raise TypeError('Other channel must be a SuperOp')
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        return SuperOp(self.data - other.data, self.input_dim, self.output_dim)

    def __neg__(self):
        return SuperOp(-self.data, self.input_dim, self.output_dim)

    # Assignment overloads
    def __imatmul__(self, other):
        if not isinstance(other, SuperOp):
            raise TypeError('Other channel must be a SuperOp')
        # check dimensions are compatible
        if other.output_dim != self.input_dim:
            raise ValueError('output_dim of other channel does not match input_dim.')
        self._data = np.dot(self.data, other.data)
        self._input_dim = other.input_dim

    def __imul__(self, other):
        if not isinstance(other, Number):
            raise TypeError("Not a number")
        self._data *= other

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
        if not isinstance(other, SuperOp):
            raise TypeError('Other channel must be a SuperOp')
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        self._data += other.data

    def __isub__(self, other):
        if not isinstance(other, SuperOp):
            raise TypeError('Other channel must be a SuperOp')
        if self.shape != other.shape:
            raise ValueError("shapes are not equal")
        self._data -= other.data
