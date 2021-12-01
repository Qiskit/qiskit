import logging

import numpy as np

from qiskit.quantum_info import Pauli
from functools import reduce
from qiskit.opflow import I, X, Y, Z
QISKIT_DICT = {"I": I, "X": X, "Y": Y, "Z": Z}


logger = logging.getLogger(__name__)


def get_operator(weight_matrix):
    from functools import reduce
    """Generate Hamiltonian for the max-cut problem of a graph.

    Args:
        weight_matrix (numpy.ndarray) : adjacency matrix.

    Returns:
        WeightedPauliOperator: operator for the Hamiltonian
        float: a constant shift for the obj function.

    """
    num_nodes = weight_matrix.shape[0]
    pauli_list = []
    shift = 0
    for i in range(num_nodes):
        for j in range(i):
            if weight_matrix[i, j] != 0:
                x_p = np.zeros(num_nodes, dtype=bool)
                z_p = np.zeros(num_nodes, dtype=bool)
                z_p[i] = True
                z_p[j] = True
                pauli_list.append([0.5 * weight_matrix[i, j], reduce(lambda a, b: a ^ b, [QISKIT_DICT[char.upper()] for char in Pauli(z_p, x_p).to_label()])])
                shift -= 0.5 * weight_matrix[i, j]
    return pauli_list, shift



def max_cut_value(x, w):
    """Compute the value of a cut.

    Args:
        x (numpy.ndarray): binary string as numpy array.
        w (numpy.ndarray): adjacency matrix.

    Returns:
        float: value of the cut.
    """
    # pylint: disable=invalid-name
    X = np.outer(x, (1 - x))
    return np.sum(w * X)


def get_graph_solution(x):
    """Get graph solution from binary string.

    Args:
        x (numpy.ndarray) : binary string as numpy array.

    Returns:
        numpy.ndarray: graph solution as binary numpy array.
    """
    return 1 - x

def max_cut_hamiltonian(D, nq):
    import random
    import networkx as nx
    G = nx.random_regular_graph(D, nq, seed=1234) # connectivity, vertices
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = random.randint(0,1000)/1000
    w = np.zeros([nq,nq])
    for i in range(nq):
        for j in range(nq):
            temp = G.get_edge_data(i,j,default=0)
            if temp != 0:
                w[i,j] = temp['weight']
    hc_pauli = get_operator(w)[0]
    return sum([coeff*w_op for coeff, w_op in hc_pauli])
