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
from libcpp.map cimport map
from libcpp.string cimport string
from libcpp cimport bool

from .converters cimport counts_to_internal
from .compute cimport compute_element, compute_col_norms, within_distance

@cython.boundscheck(False)
cdef double matrix_element(unsigned int row,
                           unsigned int col,
                           const unsigned char * bitstrings,
                           const double * cals,
                           unsigned int num_bits,
                           unsigned int distance,
                           unsigned int MAX_DIST) nogil:
    
    cdef size_t kk
    cdef double out = 0
    
    if MAX_DIST or within_distance(row, col, bitstrings, num_bits, distance):
        out = compute_element(row, col, bitstrings, cals, num_bits)
    return out

def bitstring_int(str string):
    return int(string, 2)

@cython.boundscheck(False)
@cython.cdivision(True)
def _reduced_cal_matrix(object counts, double[::1] cals,
                        unsigned int num_bits, unsigned int distance):
    
    cdef double shots = sum(counts.values())
    cdef map[string, double] counts_map = counts
    cdef unsigned int num_elems = counts_map.size()
    cdef unsigned int MAX_DIST

    MAX_DIST = distance == num_bits
    cdef double[::1,:] W = np.zeros((num_elems, num_elems), order='F', dtype=float)

    cdef double[::1] col_norms = np.zeros(num_elems, dtype=float)

    cdef unsigned char * bitstrings = <unsigned char *>malloc(num_bits*num_elems*sizeof(unsigned char))
    cdef double * input_probs = <double *>malloc(num_elems*sizeof(double))
    counts_to_internal(&counts_map, bitstrings, input_probs, num_bits, shots)

    cdef size_t ii, jj
    cdef double col_norm, _temp
    cdef dict out_dict = counts_map
    
    with nogil:
        for jj in prange(num_elems, schedule='static'):
            omp_element_compute(jj, bitstrings, &cals[0], num_elems, num_bits, distance,
                                &W[0,0], &col_norms[0], MAX_DIST)

    free(bitstrings)
    free(input_probs)
    return np.asarray(W), out_dict, np.asarray(col_norms)

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void omp_element_compute(size_t jj, const unsigned char * bitstrings,
                              const double * cals_ptr,
                              unsigned int num_elems,
                              unsigned int num_bits,
                              unsigned int distance,
                              double * W_ptr,
                              double * col_norms_ptr,
                              unsigned int MAX_DIST) nogil:
    """Computes the matrix elements for a single column
    """
    cdef double _temp, col_norm = 0
    cdef size_t ii, col_idx
    col_idx = jj*num_elems
    for ii in range(num_elems):
        _temp = matrix_element(ii, jj, bitstrings, cals_ptr,
                               num_bits, distance, MAX_DIST)
        W_ptr[col_idx+ii] = _temp
        col_norm += _temp
    col_norms_ptr[jj] = col_norm
    for ii in range(num_elems):
        W_ptr[col_idx+ii] /= col_norm

@cython.boundscheck(False)
def sdd_check(dict counts, double[::1] cals,
              unsigned int num_bits, unsigned int distance):
    """Determines if the sub-space A matrix is strictly
    diagonally dominant or not.

    Parameters:
        counts (dict): Input dict of counts.
        cals (double ndarray): Calibrations.
        num_bits (unsigned int): Number of bits in bit-strings.
        distance (unsigned int): Distance to go out to.

    Returns:
        int: Is matrix SDD or not.
    """
    cdef double shots = sum(counts.values())
    cdef map[string, double] counts_map = counts
    cdef unsigned int num_elems = counts_map.size()
    cdef unsigned int MAX_DIST

    cdef size_t row
    cdef unsigned int is_sdd = 1

    if distance > num_bits:
        raise ValueError('Distance ({}) cannot be larger than'
                         'the number of bits ({}).'.format(distance, num_bits))
    
    MAX_DIST = distance == num_bits

    # Assign memeory for bitstrings and input probabilities
    cdef unsigned char * bitstrings = <unsigned char *>malloc(num_bits*num_elems*sizeof(unsigned char))
    cdef double * input_probs = <double *>malloc(num_elems*sizeof(double))  
    # Assign memeory for column norms
    cdef double * col_norms = <double *>malloc(num_elems*sizeof(double))

    # Assign memeory sdd checks
    cdef bool * row_sdd = <bool *>malloc(num_elems*sizeof(bool))
    # Convert sorted counts dict into bistrings and input probability arrays
    counts_to_internal(&counts_map, bitstrings, input_probs, num_bits, shots)
    # Compute column norms
    compute_col_norms(col_norms, bitstrings, &cals[0], num_bits, num_elems, distance)

    with nogil:
        for row in prange(num_elems, schedule='static'):
            omp_sdd_compute(row, bitstrings, &cals[0], col_norms,
                            row_sdd, num_bits, num_elems, distance, MAX_DIST)
        
        for row in range(num_elems):
            if not row_sdd[row]:
                is_sdd = 0
                break

    # Free stuff here
    # ---------------
    free(bitstrings)
    free(input_probs)
    free(col_norms)
    free(row_sdd)

    return is_sdd

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void omp_sdd_compute(size_t row,
                          const unsigned char * bitstrings,
                          const double * cals,
                          const double * col_norms,
                          bool * row_sdd,
                          unsigned int num_bits,
                          unsigned int num_elems,
                          unsigned int distance,
                          unsigned int MAX_DIST) nogil:

    cdef size_t col
    cdef double diag_elem, mat_elem, row_sum = 0

    diag_elem = matrix_element(row, row, bitstrings, cals, num_bits, distance, MAX_DIST)
    diag_elem /= col_norms[row]
    for col in range(num_elems):
        if col != row:
            mat_elem = matrix_element(row, col, bitstrings, cals, num_bits, distance, MAX_DIST)
            mat_elem /= col_norms[col]
            row_sum += mat_elem
        row_sdd[row] = row_sum < diag_elem
