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

cdef unsigned long long popcount(unsigned long long count):
  count = (count & 0x5555555555555555) + ((count >> 1) & 0x5555555555555555);
  count = (count & 0x3333333333333333) + ((count >> 2) & 0x3333333333333333);
  count = (count & 0x0f0f0f0f0f0f0f0f) + ((count >> 4) & 0x0f0f0f0f0f0f0f0f);
  count = (count & 0x00ff00ff00ff00ff) + ((count >> 8) & 0x00ff00ff00ff00ff);
  count = (count & 0x0000ffff0000ffff) + ((count >> 16) & 0x0000ffff0000ffff);
  count = (count & 0x00000000ffffffff) + ((count >> 32) & 0x00000000ffffffff);
  return count

def expval_pauli_no_x(complex[::1] data, unsigned long long z_mask, complex phase):
    cdef double val = 0
    cdef int i
    cdef current_val
    for i in range(data.shape[0]):
        current_val = (phase * (data[i].real*data[i].real+data[i].imag*data[i].imag)).real
        if popcount(i & z_mask) & 1 != 0:
            current_val *= -1
        val += current_val
    return val

def expval_pauli_with_x(complex[::1] data, unsigned long long z_mask,
                          unsigned long long x_mask, complex phase,
                          unsigned int x_max):
        cdef unsigned long long mask_u = ~(2 ** (x_max + 1) - 1) & 0xffffffffffffffff
        cdef unsigned long long mask_l = 2**(x_max) - 1
        cdef double val = 0
        cdef unsigned int i
        cdef unsigned long long index_0
        cdef unsigned long long index_1
        cdef double current_val_0
        cdef double current_val_1
        for i in range(data.shape[0] // 2):
            index_0 = ((i << 1) & mask_u) | (i & mask_l)
            index_1 = index_0 ^ x_mask

            current_val_0 = (phase *
                    (data[index_1].real*data[index_0].real +
                     data[index_1].imag*data[index_0].imag +
                 1j*(data[index_1].imag*data[index_0].real -
                     data[index_1].real*data[index_0].imag))
                 ).real

            current_val_1 = (phase *
                    (data[index_0].real*data[index_1].real +
                     data[index_0].imag*data[index_1].imag +
                 1j*(data[index_0].imag*data[index_1].real -
                     data[index_0].real*data[index_1].imag))
                 ).real

            if popcount(index_0 & z_mask) & 1 != 0:
                val -= current_val_0
            else:
                val += current_val_0

            if popcount(index_1 & z_mask) & 1 != 0:
                val -= current_val_1
            else:
                val += current_val_1
        return val

def density_expval_pauli_no_x(complex[:, ::1] data, unsigned long long z_mask):
    cdef double val = 0
    cdef int i
    cdef current_val
    for i in range(data.shape[0]):
        current_val = (data[i][i]).real
        if popcount(i & z_mask) & 1 != 0:
            current_val *= -1
        val += current_val
    return val

def density_expval_pauli_with_x(complex[:, ::1] data, unsigned long long z_mask,
                                unsigned long long x_mask, complex phase,
                                unsigned int x_max):
        cdef unsigned long long mask_u = ~(2 ** (x_max + 1) - 1) & 0xffffffffffffffff
        cdef unsigned long long mask_l = 2**(x_max) - 1
        cdef double val = 0
        cdef unsigned int i
        cdef unsigned long long index_0
        cdef unsigned long long index_1
        cdef double current_val
        for i in range(data.shape[0] // 2):
            index_0 = ((i << 1) & mask_u) | (i & mask_l)
            index_1 = index_0 ^ x_mask
            current_val = 2 * (phase * data[index_1][index_0]).real
            if popcount(index_0 & z_mask) & 1 != 0:
                current_val *= -1
            val += current_val
        return val