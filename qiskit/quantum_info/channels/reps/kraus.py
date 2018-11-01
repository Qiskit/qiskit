# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import numpy as np
from .baserep import QChannelRep


class Kraus(QChannelRep):
    """Kraus representation of a quantum channel."""

    def __init__(self, data, input_dim=None, output_dim=None):

        # Check if it is a single unitary matrix A for channel:
        # E(rho) = A * rho * A^\dagger
        if isinstance(data, np.ndarray) or np.array(data).ndim == 2:
            # Convert single Kraus op to general Kraus pair
            kraus = ([np.array(data, dtype=complex)], None)
            shape = kraus[0][0].shape

        # Check if single Kraus set [A_i] for channel:
        # E(rho) = sum_i A_i * rho * A_i^dagger
        elif isinstance(data, list) and len(data) > 0:
            # Get dimensions from first Kraus op
            kraus = [np.array(data[0], dtype=complex)]
            shape = kraus[0].shape
            # Iterate over remaining ops and check they are same shape
            for a in data[1:]:
                op = np.array(a, dtype=complex)
                if op.shape != shape:
                    raise ValueError("Kraus operators are different dimensions.")
                kraus.append(op)
            # Convert single Kraus set to general Kraus pair
            kraus = (kraus, None)

        # Check if generalized Kraus set ([A_i], [B_i]) for channel:
        # E(rho) = sum_i A_i * rho * B_i^dagger
        elif isinstance(data, tuple) and len(data) == 2 and len(data[0]) > 0:
            kraus_left = [np.array(data[0][0], dtype=complex)]
            shape = kraus_left[0].shape
            for a in data[0][1:]:
                op = np.array(a, dtype=complex)
                if op.shape != shape:
                    raise ValueError("Kraus operators are different dimensions.")
                kraus_left.append(op)
            if data[1] is None:
                kraus = (kraus_left, None)
            else:
                kraus_right = []
                for b in data[1]:
                    op = np.array(b, dtype=complex)
                    if op.shape != shape:
                        raise ValueError("Kraus operators are different dimensions.")
                    kraus_right.append(op)
                kraus = (kraus_left, kraus_right)
        else:
            raise ValueError("Invalid input for Kraus channel.")

        dout, din = shape
        if (input_dim and input_dim != din) or (output_dim and output_dim != dout):
            raise ValueError("Invalid dimensions for Kraus input.")
        if kraus[1] is None or kraus[1] == kraus[0]:
            # Standard Kraus map
            super().__init__('Kraus', (kraus[0], None), input_dim=din, output_dim=dout)
        else:
            # General (non-CPTP) Kraus map
            super().__init__('Kraus', kraus, input_dim=din, output_dim=dout)

    @property
    def data(self):
        """Return list of Kraus matrices for channel."""
        if self._data[1] is None:
            # If only a single Kraus set, don't return the tuple
            # Just the fist set
            return self._data[0]
        else:
            # Otherwise return the tuple of both kraus sets
            return self._data

    def conjugate_channel(self):
        """Return the conjugate channel"""
        kraus_l, kraus_r = self._data
        kraus_l = [k.conj() for k in kraus_l]
        if kraus_r is not None:
            kraus_r = [k.conj() for k in kraus_r]
        return Kraus((kraus_l, kraus_r))

    def transpose_channel(self):
        """Return the transpose channel"""
        kraus_l, kraus_r = self._data
        kraus_l = [k.T for k in kraus_l]
        if kraus_r is not None:
            kraus_r = [k.T for k in kraus_r]
        din = self.output_dim
        dout = self.input_dim
        return Kraus((kraus_l, kraus_r), din, dout)

    def adjoint_channel(self):
        """Return the adjoint channel"""
        return self.conjugate_channel().transpose_channel()

    def kron(self, b):
        """Return Kraus for the channel A \otimes B

        Args:
            b (Kraus): channel B

        Returns:
            Kraus: for the composite channel A \otimes B

        Raises:
            TypeError: if b is not a Kraus object
        """
        if not isinstance(b, Kraus):
            raise TypeError('Input channels must Kraus')
        # Get tensor matrix
        ka_l, ka_r = self._data
        kb_l, kb_r = b._data
        kab_l = [np.kron(a, b) for a in ka_l for b in kb_l]
        if ka_r is None and kb_r is None:
            return Kraus((kab_l, None))
        else:
            if ka_r is None:
                ka_r = ka_l
            if kb_r is None:
                kb_r = kb_l
            kab_r = [np.kron(a, b) for a in ka_r for b in kb_r]
        return Kraus((kab_l, kab_r))
