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


def weighted_distance(num_qubits, coupling_map, properties):
    """Distance between qubits weighted by 2Q gate errors.

    Parameters:
        num_qubits (int): Number of qubits.
        coupling_map (CouplingMap): Coupling map.
        properties (BackendProperties): A backend properties instance.

    Returns:
        ndarray: 2D array of distances.
    """
    edges = edges = coupling_map.get_edges()
    twoQ_gates = []
    for gate in properties.gates:
        if len(gate.qubits) == 2:
            twoQ_gates.append(gate)
    weights = []
    for edge in edges:
        for gate in twoQ_gates:
            if gate.qubits[0] == edge[0] and gate.qubits[1] == edge[1]:
                if gate.parameters[0].value == 1.0:
                    # If a gate reports bad weight it a lot
                    weights.append(1e99)
                else:
                    weights.append(gate.parameters[0].value)
                break
    weights = np.asarray(weights)
    # normalize edge weights
    avg_weight = np.mean(weights[weights != 1])
    normed_weights = weights / avg_weight
    rows = [edge[0] for edge in edges]
    cols = [edge[1] for edge in edges]
    weighted_dist = sp.coo_matrix((normed_weights, (rows, cols)),
                                  shape=(num_qubits, num_qubits),
                                  dtype=float).tocsr()

    weighted_paths = cs.shortest_path(weighted_dist, directed=False,
                                      return_predecessors=False)

    return weighted_paths
