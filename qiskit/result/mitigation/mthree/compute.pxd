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

cdef unsigned int within_distance(unsigned int row,
                                  unsigned int col,
                                  const unsigned char * bitstrings,
                                  unsigned int num_bits,
                                  unsigned int distance) nogil

cdef double compute_element(unsigned int row,
                            unsigned int col,
                            const unsigned char * bitstrings,
                            const double * cals,
                            unsigned int num_bits) nogil

cdef void compute_col_norms(double * col_norms,
                            const unsigned char * bitstrings,
                            const double * cals,
                            unsigned int num_bits,
                            unsigned int num_elems,
                            unsigned int distance) nogil
