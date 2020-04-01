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
from libcpp.set cimport set as cset
from .utils cimport NLayout, EdgeCollection

@cython.boundscheck(False)
@cython.wraparound(False)
cdef double compute_cost(double[:, ::1] dist, unsigned int * logic_to_phys,
                          int[::1] gates, unsigned int num_gates) nogil:
    """ Computes the cost (distance) of a logical to physical mapping.
    
    Args:
        dist (ndarray): An array of doubles that specifies the distance.
        logic_to_phys (int *): Pointer to logical to physical array.
        gates (ndarray): Array of ints giving gates in layer.
        num_gates (int): The number of gates (length of gates//2).
    
    Returns:
        double: The distance calculated.
    """
    cdef unsigned int ii, jj, kk
    cdef double cost = 0.0
    for kk in range(num_gates):
        ii = logic_to_phys[gates[2*kk]]
        jj = logic_to_phys[gates[2*kk+1]]
        cost += dist[ii,jj]
    return cost

@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
cdef compute_random_scaling(double[:, ::1] scale, double[:, ::1] cdist2,
                            double * rand, unsigned int num_qubits):
    """ Computes the symmetric random scaling (perturbation) matrix, 
    and places the values in the 'scale' array.

    Args:
        scale (ndarray): An array of doubles where the values are to be stored.
        cdist2 (ndarray): Array representing the coupling map distance squared.
        rand (double *): Array of rands of length num_qubits*(num_qubits+1)//2.
        num_qubits (int): Number of physical qubits.
    """
    cdef size_t ii, jj, idx=0
    for ii in range(num_qubits):
        for jj in range(ii):
            scale[ii,jj] = rand[idx]*cdist2[ii,jj]
            scale[jj,ii] = scale[ii,jj]
            idx += 1


@cython.nonecheck(False)
@cython.boundscheck(False)
@cython.wraparound(False)
def swap_trial(int num_qubits, NLayout int_layout, int[::1] int_qubit_subset,
               int[::1] gates, double[:, ::1] cdist2, double[:, ::1] cdist, 
               int[::1] edges, double[:, ::1] scale, object rng):
    """ A single iteration of the tchastic swap mapping routine.

    Args:
        num_qubits (int): The number of physical qubits.
        int_layout (NLayout): The numeric (integer) representation of 
                              the initial_layout.
        int_qubit_subset (ndarray): Int ndarray listing qubits in set.
        gates (ndarray): Int array with integers giving qubits on which
                         two-qubits gates act on.
        cdist2 (ndarray): Array of doubles that gives the square of the 
                          distance graph.
        cdist (ndarray): Array of doubles that gives the distance graph.
        edges (ndarray): Int array of edges in coupling map.
        scale (ndarray): A double array that holds the perturbed cdist2 array.
        rng (RandomState): An instance of the NumPy RandomState.

    Returns:
        double: Best distance achieved in this trial.
        EdgeCollection: Collection of optimal edges found.
        NLayout: The optimal layout found.
        int: The number of depth steps required in mapping.
    """
    cdef EdgeCollection opt_edges = EdgeCollection()
    cdef NLayout optimal_layout, new_layout, trial_layout = int_layout.copy()
    
    cdef unsigned int num_gates = gates.shape[0]//2
    cdef unsigned int num_edges = edges.shape[0]//2
    
    cdef unsigned int need_copy, cost_reduced
    cdef unsigned int depth_step = 1
    cdef unsigned int depth_max = 2 * num_qubits + 1
    cdef double min_cost, new_cost, dist
    
    cdef unsigned int start_edge, end_edge, start_qubit, end_qubit
    cdef unsigned int optimal_start, optimal_end, optimal_start_qubit, optimal_end_qubit
    
    cdef size_t idx
    
    # Compute randomized distance
    cdef double[::1] rand = 1.0 + rng.normal(0.0, 1.0/num_qubits,
                                             size=num_qubits*(num_qubits+1)//2)
    
    compute_random_scaling(scale, cdist2, &rand[0], num_qubits)
    
    # Convert int qubit array to c++ set
    cdef cset[unsigned int] qubit_set
    cdef cset[unsigned int] input_qubit_set
    
    for idx in range(<unsigned int>int_qubit_subset.shape[0]):
        input_qubit_set.insert(int_qubit_subset[idx])
    
    # Loop over depths from 1 up to a maximum depth
    while depth_step < depth_max:
        qubit_set = input_qubit_set
        # While there are still qubits available
        while not qubit_set.empty():
            # Compute the objective function
            min_cost = compute_cost(scale, trial_layout.logic_to_phys,
                                   gates, num_gates)
            # Try to decrease objective function
            cost_reduced = 0

            # Loop over edges of coupling graph
            need_copy = 1
            for idx in range(num_edges):
                start_edge = edges[2*idx]
                end_edge = edges[2*idx+1]
                start_qubit = trial_layout.phys_to_logic[start_edge]
                end_qubit =  trial_layout.phys_to_logic[end_edge]
                # Are the qubits available?
                if  qubit_set.count(start_qubit) and qubit_set.count(end_qubit):
                    # Try this edge to reduce the cost
                    if need_copy:
                        new_layout = trial_layout.copy()
                        need_copy = 0
                    new_layout.swap(start_edge, end_edge)
                    # Compute the objective function
                    new_cost = compute_cost(scale, new_layout.logic_to_phys,
                                   gates, num_gates)
                    # Record progress if we succeed
                    if new_cost < min_cost:
                        cost_reduced = True
                        min_cost = new_cost
                        optimal_layout = new_layout
                        optimal_start = start_edge
                        optimal_end = end_edge
                        optimal_start_qubit = start_qubit
                        optimal_end_qubit = end_qubit
                        need_copy = 1
                    else:
                        new_layout.swap(start_edge, end_edge)

            # After going over all edges
            # Were there any good swap choices?
            if cost_reduced:
                qubit_set.erase(optimal_start_qubit)
                qubit_set.erase(optimal_end_qubit)
                trial_layout = optimal_layout
                opt_edges.add(optimal_start, optimal_end)
            else:
                break

        # We have either run out of swap pairs to try or
        # failed to improve the cost.

        # Compute the coupling graph distance
        dist = compute_cost(cdist, trial_layout.logic_to_phys,
                                   gates, num_gates)
        # If all gates can be applied now, we are finished.
        # Otherwise we need to consider a deeper swap circuit
        if dist == num_gates:
            break

        # Increment the depth
        depth_step += 1

    # Either we have succeeded at some depth d < dmax or failed
    dist = compute_cost(cdist, trial_layout.logic_to_phys,
                                   gates, num_gates)
    
    return dist, opt_edges, trial_layout, depth_step
