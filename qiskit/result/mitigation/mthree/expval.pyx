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
# pylint: disable=no-name-in-module
# cython: c_string_type=unicode, c_string_encoding=UTF-8
"""mthree expectation value"""
import numpy as np
cimport numpy as cnp
from mthree.exceptions import M3Error

cimport cython
from libcpp cimport bool
from libc.math cimport sqrt
from libcpp.string cimport string

OP_CONVERT = {'Z' : 0, 'I': 1, '0': 2, '1': 3}
cdef int[8] OPER_MAP = [1, -1, 1, 1, 1, 0, 0, 1]


@cython.boundscheck(False)
@cython.cdivision(True)
def exp_val(object dist, str exp_ops='', dict dict_ops={}):
    """Computes expectation values in computational basis for a supplied
    list of operators (Default is all Z).

    Parameters:
        quasi (dict): Input quasi-probability distribution.
        exp_ops (str): String representation of qubit operator to compute.
        dict_ops (dict): Dict representation of qubit operator to compute.

    Returns:
        float: Expectation value.
    """

    cdef unsigned int bits_len = len(next(iter(dist)))
    cdef unsigned char[::1] ops
    cdef bool dict_oper = 0
    if dict_ops:
        dict_oper = 1
    if exp_ops and dict_oper:
        raise M3Error('Both string and dict operators passed.')
    if not exp_ops and not dict_oper:
        exp_ops = 'Z'*bits_len
    if exp_ops:
        ops = np.array([OP_CONVERT[item] for item in exp_ops.upper()], dtype=np.uint8)
        if ops.shape[0] != bits_len:
            raise M3Error('exp_ops length does not equal number of bits.')
    
    # Find normalization to probs
    cdef double exp_val = 0
    cdef string key
    cdef double val
    cdef int oper_prod = 1
    cdef size_t kk
    cdef unsigned int shots
    for key, val in dist.items():
        if dict_oper:
            oper_prod = dict_ops.get(key, 0)
        else:
            oper_prod = 1
            for kk in range(bits_len):
                oper_prod *= OPER_MAP[2*ops[kk] + <int>(key[bits_len-kk-1])-48]

        exp_val += val * oper_prod
    
    return exp_val
