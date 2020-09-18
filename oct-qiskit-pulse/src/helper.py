import math
import re

from qiskit import pulse
from qutip import sigmaz, sigmax, sigmay, Qobj
import numpy as np
from typing import Mapping
from src.prefactor_parsing import prefactor_parser


def qubit_distribution(counts: Mapping[str, int]) -> Mapping[int, int]:
    """Convert statevector counts list to state distribution for each qubit
    individually

    Args: 
        counts: Statevector counts from IBM jobs

    Returns: Individual qubit state counts
    """
    qubits = list(range(len(list(counts.keys())[0])))
    dist = {key: {0: 0, 1: 0} for key in qubits}
    for state in counts.keys():
        for qubit in qubits:
            dist[qubit][int(state[qubit])] += int(counts[state])
    return dist


def d_sum(a: Qobj, b: Qobj) -> Qobj:
    """Perform the direct sum, or kroenecker sum of two matrices.
        Specifically here we use quantum objects. A direct sum
        simply involves taking two matrices and putting them in either 
        corner of a larger zero matrix.
        d_sum(a,b) :
        [ a   0 ]
        [ 0   b ]

    Args:
        a (Qobj): First matrix
        b (Qobj): Second matrix

    Returns:
        Qobj: Resulting matrix
    """
    tot_size = [a.shape[0] + b.shape[0], a.shape[1] + b.shape[1]]
    matrix = np.zeros(tot_size, dtype=complex)
    if isinstance(a, Qobj):
        a = a.full()
    if isinstance(b, Qobj):
        b = b.full()
    for i, row in enumerate(a):
        for j, val in enumerate(row):
            matrix[i][j] = val
    for i, row in enumerate(b):
        for j, val in enumerate(row):
            matrix[i + a.shape[0]][j + a.shape[1]] = val
    return Qobj(matrix)
