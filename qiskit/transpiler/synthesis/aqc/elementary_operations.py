# This code is part of Qiskit.
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

"""
These are a number of elementary functions that are required for
the aqc routines to work.
"""

# TODO: add more comments.

import numpy as np
from numpy import linalg as la
from qiskit.circuit.library import XGate, YGate, ZGate

# Avoid excessive deprecation warnings in Qiskit on Linux system.
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


# X = [[0, 1], [1, 0]]
X = XGate().to_matrix()
# Y = [[0, -1j], [1j, 0]]
Y = YGate().to_matrix()
# Z = [[1, 0], [0, -1]]
Z = ZGate().to_matrix()


# todo: some of them they may have been already defined in qiskit,
#  but for now we re-define them here
def unitary(u, n, j):
    return np.kron(np.kron(np.eye(2 ** (j - 1)), u), np.eye(2 ** (n - j)))


def CNOT(n, j, k):
    if j < k:
        V = np.kron(
            np.kron(np.eye(2 ** (j - 1)), [[1, 0], [0, 0]]), np.eye(2 ** (n - j))
        ) + np.kron(
            np.kron(
                np.kron(np.kron(np.eye(2 ** (j - 1)), [[0, 0], [0, 1]]), np.eye(2 ** (k - j - 1))),
                [[0, 1], [1, 0]],
            ),
            np.eye(2 ** (n - k)),
        )
    else:
        V = np.kron(
            np.kron(np.eye(2 ** (j - 1)), [[1, 0], [0, 0]]), np.eye(2 ** (n - j))
        ) + np.kron(
            np.kron(
                np.kron(np.kron(np.eye(2 ** (k - 1)), [[0, 1], [1, 0]]), np.eye(2 ** (j - k - 1))),
                [[0, 0], [0, 1]],
            ),
            np.eye(2 ** (n - j)),
        )
    return V


def Rx(phi):
    u = [[np.cos(phi / 2), -1j * np.sin(phi / 2)], [-1j * np.sin(phi / 2), np.cos(phi / 2)]]
    return u


def Ry(phi):
    u = [[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]]
    return u


def Rz(phi):
    u = [[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]]
    return u


def mcx_gate_matrix(nqubits: int, make_SU: bool = True) -> np.ndarray:
    """
    Generates a multi-control CX gate as a Numpy matrix.
    Equivalent to:
        qc = QuantumCircuit(nqubits)
        qc.mcx([0, ..., nqubits-2], nqubits-1)
        qiskit_matrix = Operator(qc.reverse_bits()).data

    N O T E, the above Qiskit code outputs generic unitary operator, whereas
    we optimize for SU one. In order to obtain SU matrix, the "qiskit_matrix"
    should be scaled accordingly, similar to what is done in the code below.

    Args:
        nqubits: total number of qubits, should be within [2 .. 16] interval.
        make_SU: generate SU matrix, if True, otherwise a generic unitary one.
    Returns:
        MCX gate matrix.
    """
    assert isinstance(nqubits, (int, np.int64)) and 2 <= nqubits <= 16
    assert isinstance(make_SU, bool)
    d = int(2 ** nqubits)
    U = np.eye(d, dtype=np.cfloat)
    U[d - 2 : d, d - 2 : d] = [[0, 1], [1, 0]]
    if make_SU:  # make SU matrix from the generic unitary one
        U /= (np.linalg.det(U) + 0j) ** (1 / d)
    return U


def toffoli_gate(n):
    # Generate a Toffoli gate
    d = int(2 ** n)
    U = np.eye(d)
    U[d - 2 : d, d - 2 : d] = [[0, 1], [1, 0]]
    U = U / ((la.det(U) + 0j) ** (1 / d))
    return U


def fredkin_gate():
    # Generate a Fredkin gate with 3 qubits
    n = 3
    d = int(2 ** n)
    U = np.eye(d)
    U[3:6, 3:6] = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    U = U / ((la.det(U) + 0j) ** (1 / d))
    return U


# transpile(qc.reverse_bits(), basis_gates=["rx", "ry", "rz", "cx"], optimization_level=3)
