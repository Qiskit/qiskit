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
from libc.stdlib cimport malloc, free
from libcpp.map cimport map
from libcpp.string cimport string

from qiskit.mitigation.mthree.converters cimport counts_to_internal, internal_to_probs


def _test_counts_to_array(object counts):

    cdef double shots = sum(counts.values())
    cdef unsigned int num_bits = len(next(iter(counts)))
    cdef map[string, double] counts_map = counts
    cdef unsigned int num_elems = counts_map.size()
    cdef size_t kk, ll
    cdef list out
    
    # Assign memeory for bitstrings and input probabilities
    cdef unsigned char * bitstrings = <unsigned char *>malloc(num_bits*num_elems*sizeof(unsigned char))
    cdef double * input_probs = <double *>malloc(num_elems*sizeof(double))

    # Convert sorted counts dict into bistrings and input probability arrays
    counts_to_internal(&counts_map, bitstrings, input_probs, num_bits, shots)

    out = ['']*num_elems

    for kk in range(num_elems):
        for ll in range(num_bits):
            out[kk] += str(bitstrings[kk*num_bits+ll])

    #free data
    free(bitstrings)
    free(input_probs)
    return out


def _test_counts_roundtrip(object counts):
    
    cdef double shots = sum(counts.values())
    cdef unsigned int num_bits = len(next(iter(counts)))
    cdef map[string, double] counts_map = counts
    cdef unsigned int num_elems = counts_map.size()
    cdef size_t kk, ll
    
    # Assign memeory for bitstrings and input probabilities
    cdef unsigned char * bitstrings = <unsigned char *>malloc(num_bits*num_elems*sizeof(unsigned char))
    cdef double * input_probs = <double *>malloc(num_elems*sizeof(double))

    # Convert sorted counts dict into bistrings and input probability arrays
    counts_to_internal(&counts_map, bitstrings, input_probs, num_bits, shots)
    internal_to_probs(&counts_map, input_probs)
    cdef dict out = counts_map

    #free data
    free(bitstrings)
    free(input_probs)
    
    return out
