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

cdef unsigned long long m1 = 0x5555555555555555
cdef unsigned long long m2 = 0x3333333333333333
cdef unsigned long long m4 = 0x0f0f0f0f0f0f0f0f
cdef unsigned long long m8 = 0x00ff00ff00ff00ff
cdef unsigned long long m16 = 0x0000ffff0000ffff
cdef unsigned long long m32 = 0x00000000ffffffff

cdef unsigned long long popcount(unsigned long long count):
  count = (count & m1) + ((count >> 1) & m1);
  count = (count & m2) + ((count >> 2) & m2);
  count = (count & m4) + ((count >> 4) & m4);
  count = (count & m8) + ((count >> 8) & m8);
  count = (count & m16) + ((count >> 16) & m16);
  count = (count & m32) + ((count >> 32) & m32);
  return count


def expval_pauli_no_x(complex[::1] data,
                      unsigned long long num_qubits,
                      unsigned long long z_mask):
    cdef double val = 0
    cdef int i
    cdef current_val
    cdef unsigned long long size = 1 << num_qubits
    for i in range(size):
        current_val = (data[i].real*data[i].real+data[i].imag*data[i].imag).real
        if popcount(i & z_mask) & 1 != 0:
            current_val *= -1
        val += current_val
    return val


def expval_pauli_with_x(complex[::1] data,
                        unsigned long long num_qubits,
                        unsigned long long z_mask,
                        unsigned long long x_mask,
                        complex phase,
                        unsigned int x_max):
        cdef unsigned long long mask_u = ~(2 ** (x_max + 1) - 1)
        cdef unsigned long long mask_l = 2**(x_max) - 1
        cdef double val = 0
        cdef unsigned int i
        cdef unsigned long long index_0
        cdef unsigned long long index_1
        cdef double current_val_0
        cdef double current_val_1
        cdef unsigned long long size = 1 << (num_qubits - 1)
        for i in range(size):
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


def density_expval_pauli_no_x(complex[::1] data,
                              unsigned long long num_qubits,
                              unsigned long long z_mask):
    cdef double val = 0
    cdef int i
    cdef unsigned long long nrows = 1 << num_qubits
    cdef unsigned long long stride = 1 + nrows
    cdef unsigned long long index
    for i in range(nrows):
        index = i * stride
        current_val = (data[index]).real
        if popcount(i & z_mask) & 1 != 0:
            current_val *= -1
        val += current_val
    return val


def density_expval_pauli_with_x(complex[::1] data,
                                unsigned long long num_qubits,
                                unsigned long long z_mask,
                                unsigned long long x_mask,
                                complex phase,
                                unsigned int x_max):
        cdef unsigned long long mask_u = ~(2 ** (x_max + 1) - 1)
        cdef unsigned long long mask_l = 2**(x_max) - 1
        cdef double val = 0
        cdef unsigned int i
        cdef double current_val
        cdef unsigned long long nrows = 1 << num_qubits
        cdef unsigned long long index_vec
        cdef unsigned long long index_mat
        for i in range(nrows >> 1):
            index_vec = ((i << 1) & mask_u) | (i & mask_l)
            index_mat = index_vec ^ x_mask + nrows * index_vec
            current_val = 2 * (phase * data[index_mat]).real
            if popcount(index_vec & z_mask) & 1 != 0:
                current_val *= -1
            val += current_val
        return val
