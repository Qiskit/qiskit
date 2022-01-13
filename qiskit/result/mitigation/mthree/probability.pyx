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
from libc.math cimport sqrt

@cython.boundscheck(False)
@cython.cdivision(True)
def quasi_to_probs(object quasiprobs):
    """Takes a quasiprobability distribution and maps
    it to the closest probability distribution as defined by
    the L2-norm.
    Parameters:
        quasiprobs (QuasiDistribution): Input quasiprobabilities.
    Returns:
        dict: Nearest probability distribution
        float: Distance between distributions
    Notes:
        Method from Smolin et al., Phys. Rev. Lett. 108, 070502 (2012).
    """
    cdef dict sorted_probs = dict(sorted(quasiprobs.items(), key=lambda item: item[1]))
    cdef unsigned int num_elems = len(sorted_probs)
    cdef dict new_probs = {}
    cdef double beta = 0
    cdef object key
    cdef double val, temp, diff = 0
    for key, val in sorted_probs.items():
        temp = val+beta/num_elems
        if temp < 0:
            beta += val
            num_elems -= 1
            diff += val*val
        else:
            diff += (beta/num_elems)*(beta/num_elems)
            new_probs[key] = sorted_probs[key] + beta/num_elems
    return new_probs, sqrt(diff)
