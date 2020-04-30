# -*- coding: utf-8 -*-
#!python
#cython: language_level = 3
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

cimport cython
import numpy as np
from libc.stdlib cimport calloc, free
from libcpp.vector cimport vector

from qiskit.transpiler.layout import Layout
from qiskit.circuit import Qubit

cdef class EdgeCollection:
    """ A simple contain that contains a C++ vector
    representing edges in the coupling map that are
    found to be optimal by the swap mapper.  This allows
    us to keep the vector alive.
    """
    cpdef void add(self, unsigned int edge_start, unsigned int edge_end):
        """ Add two edges, in order, to the collection.

        Args:
            edge_start (int): The beginning edge.
            edge_end (int): The end of the edge.
        """
        self._edges.push_back(edge_start)
        self._edges.push_back(edge_end)
    
    @property
    def size(self):
        """ The size of the edge collection.
        Returns:
            int: Size of the edge collection.
        """
        return self._edges.size()

    @cython.boundscheck(False)
    def edges(self):
        """ Returns the vector of edges as a NumPy array.
        Returns:
            ndarray: Int array of edges.
        """
        cdef size_t kk
        out = np.zeros(self._edges.size(), dtype=np.uint32)
        for kk in range(self._edges.size()):
            out[kk] = self._edges[kk]
        return out


cdef class NLayout:
    """ A Numeric representation of a Qiskit Layout object.
    Here all qubit layouts are stored as int arrays.
    """
    def __cinit__(self, unsigned int num_logical,
                  unsigned int num_physical):
        """ Init object.
        Args:
            num_logical (int): Number of logical qubits.
            num_physical (int): Number of physical qubits.
        """
        self.l2p_len = num_logical
        self.p2l_len = num_physical
        self.logic_to_phys = <unsigned int *>calloc(num_logical,
                                                    sizeof(unsigned int))
        self.phys_to_logic = <unsigned int *>calloc(num_physical,
                                                    sizeof(unsigned int))
    
    def __dealloc__(self):
        """ Clears the pointers when finished.
        """
        if self.logic_to_phys is not NULL:
            free(self.logic_to_phys)
            self.logic_to_phys = NULL
        if self.phys_to_logic is not NULL:
            free(self.phys_to_logic)
            self.phys_to_logic = NULL
            
    @property
    def logic_to_phys(self):
        """ The array mapping logical to physical qubits.
        Returns:
            ndarray: Int array of logical to physical mappings.
        """
        cdef size_t kk
        out = np.zeros(self.l2p_len, dtype=np.int32)
        for kk in range(self.l2p_len):
            out[kk] = self.logic_to_phys[kk]
        return out
    
    @property
    def phys_to_logic(self):
        """ The array mapping physical to logical qubits.
        Returns:
            ndarray: Int array of physical to logical mappings.
        """
        cdef size_t kk
        out = np.zeros(self.p2l_len, dtype=np.int32)
        for kk in range(<unsigned int>self.p2l_len):
            out[kk] = self.phys_to_logic[kk]
        return out
    
    @cython.boundscheck(False)
    cdef NLayout copy(self):
        """ Returns a copy of the layout.

        Returns:
            NLayout: A copy of the layout.
        """
        cdef NLayout out = NLayout(self.l2p_len, self.p2l_len)
        cdef size_t kk
        for kk in range(<unsigned int>self.l2p_len):
            out.logic_to_phys[kk] = self.logic_to_phys[kk]
        for kk in range(<unsigned int>self.p2l_len):
            out.phys_to_logic[kk] = self.phys_to_logic[kk]
        return out
            
    @cython.boundscheck(False)
    cdef void swap(self, unsigned int idx1, unsigned int idx2):
        """ Swaps two indices in the Layout

        Args:
            idx1 (int): Index 1.
            idx2 (int): Index 2.
        """
        cdef unsigned int temp1, temp2
        temp1 = self.phys_to_logic[idx1]
        temp2 = self.phys_to_logic[idx2]
        self.phys_to_logic[idx1] = temp2
        self.phys_to_logic[idx2] = temp1
        self.logic_to_phys[self.phys_to_logic[idx1]] = idx1
        self.logic_to_phys[self.phys_to_logic[idx2]] = idx2
        
    @cython.boundscheck(False)
    cpdef object to_layout(self, object qregs):
        """ Converts numeric layout back to Qiskit Layout object.

        Args:
            qregs (OrderedDict): An ordered dict of Qubit instances.
        
        Returns:
            Layout: The corresponding Qiskit Layout object.
        """
        out = Layout()
        cdef unsigned int main_idx = 0
        cdef size_t idx
        for qreg in qregs.values():
            for idx in range(<unsigned int>qreg.size):
                out[qreg[idx]] = self.logic_to_phys[main_idx]
                main_idx += 1
        return out
    
    
cpdef NLayout nlayout_from_layout(object layout, object qregs, 
                                  unsigned int physical_qubits):
    """ Converts Qiskit Layout object to numerical NLayout.

    Args:
        layout (Layout): A Qiskit Layout instance.
        qregs (OrderedDict): An ordered dict of Qubit instances.
        physical_qubits (int): Number of physical qubits.
    Returns:
        NLayout: The corresponding numerical layout.
    """
    cdef size_t ind
    cdef list sizes = [qr.size for qr in qregs.values()]
    cdef int[::1] reg_idx = np.cumsum([0]+sizes, dtype=np.int32)
    cdef unsigned int logical_qubits = sum(sizes)

    cdef dict regint = {}
    for ind, qreg in enumerate(qregs.values()):
        regint[qreg] = ind

    cdef NLayout out = NLayout(logical_qubits, physical_qubits)
    cdef object key, val
    cdef dict merged_dict = {**layout._p2v, **layout._v2p}
    for key, val in merged_dict.items():
        if isinstance(key, Qubit):
            out.logic_to_phys[reg_idx[regint[key.register]]+key.index] = val
        else:
            out.phys_to_logic[key] = reg_idx[regint[val.register]]+val.index
    return out
