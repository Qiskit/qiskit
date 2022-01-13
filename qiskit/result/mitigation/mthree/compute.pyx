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
cimport cython
from cython.parallel cimport prange

@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline unsigned int within_distance(unsigned int row,
                                         unsigned int col,
                                         const unsigned char * bitstrings,
                                         unsigned int num_bits,
                                         unsigned int distance) nogil:
    """Computes the Hamming distance between two bitstrings.

    Parameters:
        row (unsigned int): The row index.
        col (unsigned int): The col index.
        bitstrings (unsigned char *): Pointer to array of all bitstrings.
        num_bits (unsigned int): The number of bits in a bitstring.
        distance (unsigned int): The distance to calculate out to.

    Returns:
        unsigned int: Are the bitstrings within given distance.
    """

    cdef size_t kk
    cdef unsigned int temp_dist = 0
    cdef unsigned int row_start = row*num_bits
    cdef unsigned int col_start = col*num_bits

    for kk in range(num_bits):
        temp_dist += bitstrings[row_start+kk] != bitstrings[col_start+kk]
    return temp_dist <= distance

@cython.boundscheck(False)
@cython.cdivision(True)
cdef inline double compute_element(unsigned int row,
                                   unsigned int col,
                                   const unsigned char * bitstrings,
                                   const double * cals,
                                   unsigned int num_bits) nogil:
    """Computes the matrix element specified by the input
    bit strings from the supplied tensored cals data.

    Parameters:
        row_arr (unsigned char *): Basis element giving row index.
        col_arr (unsigned char *): Basis element giving col index.
        cals (const double *): Tensored calibration data.
        num_qubits (unsigned int): Number of qubits in arrays

    Returns:
        double: Matrix element value.
    """
    cdef double res = 1
    cdef size_t kk
    cdef unsigned int offset
    cdef unsigned int row_start = num_bits*row
    cdef unsigned int col_start = num_bits*col
    
    for kk in range(num_bits):
        offset = 2*bitstrings[row_start+kk]+bitstrings[col_start+kk]
        res *= cals[4*kk+offset]
    return res

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void compute_col_norms(double * col_norms,
                            const unsigned char * bitstrings,
                            const double * cals,
                            unsigned int num_bits,
                            unsigned int num_elems,
                            unsigned int distance) nogil:
    """Compute the matrix column norms for each bitstring.

    Parameters:
        col_norms (double *): Pointer to double array in which to poulate norms.
        bitstrings (unsigned char *): Pointer to array of all bitstrings.
        num_bits (unsigned int): The number of bits in a bitstring.
        num_elems (unsigned int): The number of bitstring elements.
        distance (unsigned int): The distance to calculate out to.
    """
    cdef size_t col, row
    cdef double col_norm
    cdef unsigned int MAX_DIST = 0

    if distance == num_bits:
        MAX_DIST = 1
    # Compute the column norm for each element
    with nogil:
        for col in prange(num_elems, schedule='static'):
            col_norm = _inner_col_norm_loop(col,
                                            bitstrings,
                                            cals,
                                            num_bits,
                                            num_elems,
                                            distance,
                                            MAX_DIST)
            col_norms[col] = col_norm


cdef double _inner_col_norm_loop(unsigned int col,
                                 const unsigned char * bitstrings,
                                 const double * cals,
                                 unsigned int num_bits,
                                 unsigned int num_elems,
                                 unsigned int distance,
                                 unsigned int MAX_DIST) nogil:
    """An inner-loop function for computing col_norms in parallel.

    This is needed to get around an issue with how Cython tries to do
    OMP reductions.

    Parameters:
        col (int): The column of interest.
        col_norms (double *): Pointer to double array in which to poulate norms.
        bitstrings (unsigned char *): Pointer to array of all bitstrings.
        num_bits (unsigned int): The number of bits in a bitstring.
        num_elems (unsigned int): The number of bitstring elements.
        distance (unsigned int): The distance to calculate out to.
        MAX_DIST (unsigned int): Are we doing max distance.
    """
    cdef size_t row
    cdef double col_norm = 0

    for row in range(num_elems):
        if MAX_DIST or within_distance(row, col, bitstrings, num_bits, distance):
            col_norm += compute_element(row, col, bitstrings, cals, num_bits)
    return col_norm
