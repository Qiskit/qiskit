# -*- coding: utf-8 -*-

# Copyright 2018, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import numpy as np
from .baserep import QChannelRep


class Stinespring(QChannelRep):
    """Stinespring representation of a quantum channel"""

    def __init__(self, data, input_dim=None, output_dim=None):
        if not isinstance(data, tuple):
            # Convert single Stinespring set to length 1 tuple
            stine = (np.array(data, dtype=complex), None)
        if isinstance(data, tuple) and len(data) == 2:
            if data[1] is None:
                stine = (np.array(data[0], dtype=complex), None)
            else:
                stine = (np.array(data[0], dtype=complex),
                         np.array(data[1], dtype=complex))

        dim_left, dim_right = stine[0].shape
        # If two stinespring matrices check they are same shape
        if stine[1] is not None:
            if stine[1].shape != (dim_left, dim_right):
                raise ValueError("Invalid Stinespring input.")
        if input_dim is None:
            input_dim = dim_right
        if output_dim is None:
            output_dim = input_dim
        if dim_right % output_dim != 0:
            raise ValueError("Invalid output dimension.")
        if stine[1] is None or (stine[1] == stine[0]).all():
            # Standard Stinespring map
            super().__init__('Stinespring', (stine[0], None),
                             input_dim=input_dim,
                             output_dim=output_dim)
        else:
            # General (non-CPTP) Stinespring map
            super().__init__('Stinespring', stine,
                             input_dim=input_dim,
                             output_dim=output_dim)

    @property
    def data(self):
        # Override to deal with data being either tuple or not
        if self._data[1] is None:
            return self._data[0]
        else:
            return self._data

    def kron(self, b):
        """Return Stinespring for the channel kron(A, B)

        Args:
            b (Stinespring): channel B

        Returns:
            Stinespring: for the composite channel kron(A, B)

        Raises:
            TypeError: if b is not a Stinespring object
        """
        if not isinstance(b, Stinespring):
            raise TypeError('Input channels must Stinespring')
        # Tensor stinespring ops
        sa_l, sa_r = self._data
        sb_l, sb_r = b._data
        sab_l = np.kron(sa_l, sb_l)
        if sa_r is None and sb_r is None:
            sab_r = None
        elif sa_r is None:
            sab_r = np.kron(sa_l, sb_r)
        elif sb_r is None:
            sab_r = np.kron(sa_r, sb_l)
        else:
            sab_r = np.kron(sa_r, sb_r)

        # Reshuffle tensor dimensions
        din_a = self.input_dim
        dout_a = self.output_dim
        din_b = b.input_dim
        dout_b = b.output_dim
        dtr_a = sa_l.shape[0] // dout_a
        dtr_b = sb_l.shape[0] // dout_b

        # Left stinespring
        shape_in = (dout_a, dtr_a, dout_b, dtr_b, din_a * din_b)
        shape_out = (dout_a * dtr_a * dout_b * dtr_b, din_a * din_b)
        sab_l = np.reshape(np.transpose(np.reshape(sab_l, shape_in),
                                        (0, 2, 1, 3, 4)), shape_out)

        # Right stinespring
        if sab_r is not None:
            shape_in = (dout_a, dtr_a, dout_b, dtr_b, din_a * din_b)
            shape_out = (dout_a * dtr_a * dout_b * dtr_b, din_a * din_b)
            sab_r = np.reshape(np.transpose(np.reshape(sab_r, shape_in),
                                            (0, 2, 1, 3, 4)), shape_out)

        return Stinespring((sab_l, sab_r), input_dim=din_a * din_b,
                           output_dim=dout_a * dout_b)
