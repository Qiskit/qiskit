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
# cython: c_string_type=unicode, c_string_encoding=UTF-8
cimport cython
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free
from libcpp.map cimport map
from libcpp.string cimport string

from qiskit.mitigation.mthree.compute cimport compute_col_norms
from qiskit.mitigation.mthree.converters cimport counts_to_internal


def _test_vector_column_norm(object counts,
                             double[::1] cals,
                             int distance):
    """Test computing the column norm on a full vector

    Parameters:
        col (unsigned char memoryview): Bitstring for column
        cals (double memoryview): Input calibration data.
        distance (int): Distance (weight) of errors to consider.
    """
    cdef unsigned int num_bits = len(next(iter(counts)))
    cdef double shots = sum(counts.values())
    cdef map[string, double] counts_map = counts
    cdef unsigned int num_elems = counts_map.size()

    # Assign memeory for bitstrings and input probabilities
    cdef unsigned char * bitstrings = <unsigned char *>malloc(num_bits*num_elems*sizeof(unsigned char))
    cdef double * input_probs = <double *>malloc(num_elems*sizeof(double))
    # Assign memeory for column norms
    cdef double[::1] col_norms = np.zeros(num_elems, dtype=float)

    # Convert sorted counts dict into bistrings and input probability arrays
    counts_to_internal(&counts_map, bitstrings, input_probs, num_bits, shots)
    # Compute column norms
    compute_col_norms(&col_norms[0], bitstrings, &cals[0], num_bits, num_elems, distance)
    
    free(bitstrings)
    free(input_probs)
    return np.asarray(col_norms)