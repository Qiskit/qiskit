# This code is part of Mthree.
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
# cython: c_string_type=unicode, c_string_encoding=UTF-8
cimport cython
from cython.parallel cimport prange
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.string cimport string
from cython.operator cimport dereference, postincrement

from .compute cimport within_distance, compute_element, compute_col_norms


cdef class M3MatVec():
    cdef unsigned char * bitstrings
    cdef double * col_norms
    cdef bool MAX_DIST
    cdef unsigned int distance
    cdef public unsigned int num_elems
    cdef public unsigned int num_bits
    cdef double * cals
    cdef public dict sorted_counts
    
    def __cinit__(self, object counts, double[::1] cals, int distance=-1):
        
        cdef double shots = sum(counts.values())
        cdef map[string, double] counts_map = counts
        self.num_elems = counts_map.size()
        self.num_bits = len(next(iter(counts)))
        self.cals = &cals[0]
        self.sorted_counts = counts_map
        
        if distance == -1:
            distance = self.num_bits

        self.distance = distance
        self.MAX_DIST = self.distance == self.num_bits
        
        self.bitstrings = <unsigned char *>malloc(self.num_bits*self.num_elems*sizeof(unsigned char))
        self.col_norms = <double *>malloc(self.num_elems*sizeof(double))
        
        counts_to_bitstrings(&counts_map, self.bitstrings, self.num_bits)
        
        compute_col_norms(self.col_norms, self.bitstrings, self.cals,
                          self.num_bits, self.num_elems, distance)
        
    @cython.boundscheck(False)
    def get_col_norms(self):
        """
        Get the internally used column norms.

        Returns:
            ndarray: Column norms.
        """
        cdef size_t kk
        cdef double[::1] out = np.empty(self.num_elems, dtype=float)
        for kk in range(self.num_elems):
            out[kk] = self.col_norms[kk]
        return np.asarray(out, dtype=float)

    @cython.boundscheck(False)
    @cython.cdivision(True)
    def get_diagonal(self):
        cdef size_t kk
        cdef double temp_elem
        cdef double[::1] out = np.empty(self.num_elems, dtype=float)
        for kk in range(self.num_elems):
            temp_elem = compute_element(kk, kk, self.bitstrings,
                                        self.cals, self.num_bits)
            temp_elem /= self.col_norms[kk]
            out[kk] = temp_elem
        return np.asarray(out, dtype=float)

    @cython.boundscheck(False)
    def matvec(self, const double[::1] x):
        cdef size_t row
        if x.shape[0] != self.num_elems:
            raise Exception('Incorrect length of input vector.')
        cdef double[::1] out = np.empty(self.num_elems, dtype=float)
        with nogil:
            for row in prange(self.num_elems, schedule='static'):
                omp_matvec(row, &x[0], &out[0],
                           self.bitstrings, self.col_norms, self.cals,
                           self.num_elems, self.num_bits, self.distance,
                           self.MAX_DIST)
        return np.asarray(out, dtype=float)

    @cython.boundscheck(False)
    def rmatvec(self, const double[::1] x):
        cdef size_t col
        if x.shape[0] != self.num_elems:
            raise Exception('Incorrect length of input vector.')
        cdef double[::1] out = np.empty(self.num_elems, dtype=float)
        with nogil:
            for col in prange(self.num_elems, schedule='static'):
                omp_rmatvec(col, &x[0], &out[0],
                            self.bitstrings, self.col_norms, self.cals,
                            self.num_elems, self.num_bits, self.distance,
                            self.MAX_DIST)
        return np.asarray(out, dtype=float)

    def __dealloc__(self):
        if self.bitstrings is not NULL:
            free(self.bitstrings)
        if self.col_norms is not NULL:
            free(self.col_norms)

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void omp_matvec(size_t row,
                     const double * x,
                     double * out,
                     const unsigned char * bitstrings,
                     const double * col_norms,
                     const double * cals,
                     unsigned int num_elems,
                     unsigned int num_bits,
                     unsigned int distance,
                     bool MAX_DIST) nogil:
    cdef double temp_elem, row_sum = 0
    cdef size_t col
    for col in range(num_elems):
        if MAX_DIST or within_distance(row, col, bitstrings,
                                       num_bits, distance):
            temp_elem = compute_element(row, col, bitstrings,
                                        cals, num_bits)
            temp_elem /= col_norms[col]
            row_sum += temp_elem * x[col]
    out[row] = row_sum

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void omp_rmatvec(size_t col,
                      const double * x,
                      double * out,
                      const unsigned char * bitstrings,
                      const double * col_norms,
                      const double * cals,
                      unsigned int num_elems,
                      unsigned int num_bits,
                      unsigned int distance,
                      bool MAX_DIST) nogil:
    cdef double temp_elem, row_sum = 0
    cdef size_t row
    for row in range(num_elems):
        if MAX_DIST or within_distance(row, col, bitstrings,
                                       num_bits, distance):
            temp_elem = compute_element(row, col, bitstrings,
                                        cals, num_bits)
            temp_elem /= col_norms[col]
            row_sum += temp_elem * x[row]
    out[col] = row_sum


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void counts_to_bitstrings(map[string, double] * counts_map,
                               unsigned char * bitstrings,
                               unsigned int num_bits):
   
    cdef unsigned int idx, letter, start
    cdef map[string, double].iterator end = counts_map.end()
    cdef map[string, double].iterator it = counts_map.begin()
    cdef string temp
    idx = 0
    while it != end:
        start = num_bits*idx
        temp = dereference(it).first
        for letter in range(num_bits):
            bitstrings[start+letter] = <unsigned char>temp[letter]-48
        idx += 1
        postincrement(it)
