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
# pylint: disable=no-name-in-module
"""mthree expectation value"""
cimport cython
from libc.math cimport sqrt


@cython.boundscheck(False)
def exp_val(object quasi):
    """Computes expectation value in computational basis.

    Parameters:
        quasi (dict): Input quasi-probability distribution.

    Returns:
        float: Expectation value.
    """
    # Find normalization to probs
    cdef double exp_val = 0
    cdef str key
    cdef double val
    cdef unsigned int one_count
    for key, val in quasi.items():
        one_count = key.count('1') % 2
        exp_val += val * (-1 if one_count else 1)
    return exp_val

@cython.boundscheck(False)
def exp_val_and_stddev(object probs):
    """Computes expectation value and standard deviation in computational basis
    for a given probability distribution (not quasi-probs).

    Parameters:
        probs (dict): Input probability distribution.

    Returns:
        float: Expectation value.
        float: Standard deviation.
    """
    # Find normalization to probs
    cdef double exp_val = 0
    cdef double stddev, exp2 = 0
    cdef str key
    cdef double val
    cdef unsigned int one_count
    for key, val in probs.items():
        one_count = key.count('1') % 2
        exp_val += val * (-1 if one_count else 1)
        exp2 += val
    
    stddev = sqrt((exp2 - exp_val*exp_val) / probs.shots)
    
    return exp_val, stddev
