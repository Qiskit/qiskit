# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""A set of routines used by all routing methods"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as cs


def weighted_distance(num_qubits, coupling_map, properties,
                      alpha_dis=0.5, alpha_error=0.5, alpha_time=0):
    """Distance between qubits weighted by 2Q gate errors, 2Q SWAP shortest
       path and 2Q execution time.

    Parameters:
        num_qubits (int): Number of qubits.
        coupling_map (CouplingMap): Coupling map.
        properties (BackendProperties): A backend properties instance.
        alpha_dis (float): Weight of swap distance matrix.
        alpha_error (float): Weight of swap error rate matrix.
        alpha_time (float): Weight of swap execution time matrix.

    Additional Information:
        math:
            The error rate of a swap gate
                E(SWAP.q_i,SWAP.q_j) = 1 - F(q_i,q_j) * F(q_j,q_i) * max(F(q_i,q_j),F(q_j,q_i))
                F(q_i,q_j) = 1 - E(q_i,q_j)
            The execution time of a swap gate
                T(SWAP.q_i,SWAP.q_j) = T(q_i,q_j) + T(q_j,q_i) + min(T(q_i,q_j),T(q_j,q_i))
            The weighted distance matrix
                D = alpha_dis * swap_distance_matrix +
                    alpha_error * swap_error_rate_matrix +
                    alpha_time * swap_execution_time_matrix

    Returns:
        ndarray: 2D array of distances.
    """
    edges = coupling_map.get_edges()
    two_qubit_gates = []
    for gate in properties.gates:
        if len(gate.qubits) == 2:
            two_qubit_gates.append(gate)
    error_weights = []
    execution_time_weights = []
    cnot_fidelity = cnot_execution_time = reverse_cnot_fidelity = reverse_cnot_execution_time = 0
    for edge in edges:
        for gate in two_qubit_gates:
            if gate.qubits[0] == edge[0] and gate.qubits[1] == edge[1]:
                cnot_fidelity = 1 - gate.parameters[0].value
                cnot_execution_time = gate.parameters[1].value
                break
        for gate in two_qubit_gates:
            if gate.qubits[1] == edge[0] and gate.qubits[0] == edge[1]:
                reverse_cnot_fidelity = 1 - gate.parameters[0].value
                reverse_cnot_execution_time = gate.parameters[1].value
                break

        error_weights.append(1 - cnot_fidelity * reverse_cnot_fidelity
                             * max(cnot_fidelity, reverse_cnot_fidelity))
        execution_time_weights.append(cnot_execution_time + reverse_cnot_execution_time
                                      + min(cnot_execution_time, reverse_cnot_execution_time))

    error_weights = np.asarray(error_weights)
    execution_time_weights = np.asarray(execution_time_weights)
    coupling_map._compute_distance_matrix()
    swap_distance_weights = coupling_map._dist_matrix
    # normalize edge weights
    avg_error_weight = np.mean(error_weights[error_weights != 1])
    avg_execution_time_weight = np.mean(execution_time_weights[execution_time_weights != 1])
    avg_swap_distance_weight = np.linalg.norm(swap_distance_weights)

    normed_error_weights = error_weights / avg_error_weight
    normed_execution_time_weights = execution_time_weights / avg_execution_time_weight
    normed_swap_distance_weight = swap_distance_weights / avg_swap_distance_weight

    normed_weights = alpha_error * normed_error_weights + alpha_time * normed_execution_time_weights

    rows = [edge[0] for edge in edges]
    cols = [edge[1] for edge in edges]
    weighted_dist = sp.coo_matrix((normed_weights, (rows, cols)),
                                  shape=(num_qubits, num_qubits),
                                  dtype=float).tocsr()
    weighted_paths = cs.shortest_path(weighted_dist, directed=False,
                                      return_predecessors=False)

    weighted_paths = weighted_paths + alpha_dis * normed_swap_distance_weight

    return weighted_paths
