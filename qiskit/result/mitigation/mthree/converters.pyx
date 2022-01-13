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
from libcpp.map cimport map
from libcpp.string cimport string
from cython.operator cimport dereference, postincrement

@cython.boundscheck(False)
@cython.cdivision(True)
cdef void counts_to_internal(map[string, double] * counts_map,
                             unsigned char * vec,
                             double * probs,
                             unsigned int num_bits,
                             double shots):
    """Converts a Qiskit counts object (or Python dict) into an array
    of bitstrings and probabilities.
    
    Parameters:
        counts (object): A Qiskit counts object or Python dict.
        vec (unsigned char *): Pointer to array of bitstrings to populate.
        probs (double *): Pointer to array of probabilities to populate.
        num_bits (unsigned int): Number of bits in the bitstrings.
    """
    cdef unsigned int idx, letter, start
    cdef map[string, double].iterator end = counts_map.end()
    cdef map[string, double].iterator it = counts_map.begin()
    cdef string temp
    idx = 0
    while it != end:
        start = num_bits*idx
        probs[idx] = dereference(it).second / shots
        temp = dereference(it).first
        for letter in range(num_bits):
            vec[start+letter] = <unsigned char>temp[letter]-48
        idx += 1
        postincrement(it)


@cython.boundscheck(False)
@cython.cdivision(True)
cdef void internal_to_probs(map[string, double] * counts_map,
                            double * probs):
    """Converts internal arrays back into a Python dict.
    
    Parameters:
        vec (unsigned char *): Pointer to array of bitstrings.
        vec (dobule *): Pointer to array of probabilities.
        num_elems (unsigned int): Number of elements.
        num_bits (unsigned int): Number of bits in the bitstrings.
    """
    cdef size_t idx = 0
    cdef map[string, double].iterator end = counts_map.end()
    cdef map[string, double].iterator it = counts_map.begin()

    while it != end:
        dereference(it).second = probs[idx]
        idx += 1
        postincrement(it)
