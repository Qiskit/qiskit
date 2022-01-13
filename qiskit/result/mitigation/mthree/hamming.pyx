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
import numpy as np


def hamming_ball(unsigned char[::1] arr, unsigned int pos, int distance):
    """Compute all bitstrings up to a given Hamming distance
    away from a target bitstring.
    
    Parameters:
        arr (char array): Input bitstring.
        pos (int): Index position.
        distance (int): Maximum Hamming distance to consider
    
    Returns:
        list: List of all bitstrings as arrays.
    """
    cdef list out = []
    _hamming_core(arr, pos, distance, out)
    return out

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _hamming_core(unsigned char[::1] arr, unsigned int pos, int distance, list out):
    """Compute all bitstrings up to a given Hamming distance
    away from a target bitstring.
    
    Parameters:
        arr (char array): Input bitstring.
        pos (int): Index position.
        distance (int): Maximum Hamming distance to consider
        out (list): A empty list to store bitstrings to.
    """
    cdef unsigned int length = arr.shape[0]
    cdef int dist_offset
    cdef unsigned char temp
    cdef size_t kk
    
    if pos == length:
        # Here is where you do things for each bitstring
        out.append(np.asarray(arr.copy(), dtype=np.uint8))
        return
    
    if distance > 0:
        temp = arr[pos]
        for kk in range(2):
            arr[pos] = kk
            dist_offset = 0
            if temp != arr[pos]:
                dist_offset = -1
            _hamming_core(arr, pos+1, distance + dist_offset, out)
        arr[pos] = temp
    else:
        _hamming_core(arr, pos+1, distance, out)
