# -*- coding: utf-8 -*-
#!python
#cython: language_level = 3, cdivision = True, nonecheck = False
#distutils: language = c++

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2018.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
from libcpp.vector cimport vector

# Numeric layout --------------------------------------------------------------
cdef class NLayout:
    cdef:
        unsigned int l2p_len
        unsigned int p2l_len
        unsigned int * logic_to_phys
        unsigned int * phys_to_logic

    # Methods
    cdef NLayout copy(self)
    cdef void swap(self, unsigned int idx1, unsigned int idx2)
    cpdef object to_layout(self, object dag)


cpdef NLayout nlayout_from_layout(object layout,
                                  object dag, 
                                  unsigned int physical_qubits)


# Edge collection -------------------------------------------------------------
cdef class EdgeCollection:
    cdef vector[unsigned int] _edges

    cpdef void add(self, unsigned int edge_start, unsigned int edge_end)
