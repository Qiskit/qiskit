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
from libcpp.map cimport map
from libcpp.string cimport string

cdef void counts_to_internal(map[string, double] * counts,
                             unsigned char * vec,
                             double * probs,
                             unsigned int num_bits,
                             double shots)

cdef void internal_to_probs(map[string, double] * counts,
                            double * probs)
