#!python
#cython: language_level = 3
#distutils: language = c++

# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

cimport cython
import numpy as np

cpdef unsigned long long popcount(unsigned long long count):
  count = (count & 0x5555555555555555) + ((count >> 1) & 0x5555555555555555);
  count = (count & 0x3333333333333333) + ((count >> 2) & 0x3333333333333333);
  count = (count & 0x0f0f0f0f0f0f0f0f) + ((count >> 4) & 0x0f0f0f0f0f0f0f0f);
  count = (count & 0x00ff00ff00ff00ff) + ((count >> 8) & 0x00ff00ff00ff00ff);
  count = (count & 0x0000ffff0000ffff) + ((count >> 16) & 0x0000ffff0000ffff);
  count = (count & 0x00000000ffffffff) + ((count >> 32) & 0x00000000ffffffff);
  return count

cpdef double expval_pauli_no_x(complex[:] data, unsigned long long z_mask, complex phase):
    cpdef double val = 0
    for i in range(len(data)):
        current_val = (phase * data[i] * np.conj(data[i])).real
        if popcount(i & z_mask) & 1 != 0:
            current_val *= -1
        val += current_val
    return val

cpdef expval_pauli_with_x(complex[:] data, unsigned long long z_mask,
                          unsigned long long x_mask, complex phase,
                          unsigned int x_max):
        cpdef unsigned long long mask_u = ~(2 ** (x_max + 1) - 1) & 0xffffffffffffffff
        cpdef unsigned long long mask_l = 2**(x_max) - 1
        cpdef double val = 0
        for i in range(len(data) // 2):
            index = ((i << 1) & mask_u) | (i & mask_l)
            indices = [index, index ^ x_mask]
            data_pair = [data[indices[k]] for k in range(2)]
            current_val = [(phase * data_pair[1] * np.conj(data_pair[0])).real,
                           (phase * data_pair[0] * np.conj(data_pair[1])).real]
            for k in range(2):
                if popcount(indices[k] & z_mask) & 1 != 0:
                    val -= current_val[k]
                else:
                    val += current_val[k]
        return val