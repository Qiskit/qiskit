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
These are a number of elementary functions that are required for the aqc routines to work.
"""

import numpy as np

from qiskit.circuit.library import XGate, YGate, ZGate, RXGate, RZGate

# todo: move all of them to gradient.py?
X = XGate().to_matrix()
Y = YGate().to_matrix()
Z = ZGate().to_matrix()


def op_unitary(unitary: np.ndarray, n: int, j: int) -> np.ndarray:
    """
    I(j - 1) tensor product U tensor product I(n - j).

    Args:
        unitary: 2x2, single qubit unitary or bigger?
        n: num qubits
        j: position where to place a unitary

    Returns:
        a unitary of n qubits with u in position j.
    """
    return np.kron(np.kron(np.eye(2 ** (j - 1)), unitary), np.eye(2 ** (n - j)))


def op_cnot(n: int, j: int, k: int) -> np.ndarray:
    """
    Places a CNOT from j to k (what is target, what is control?), todo: e.g. j = 1, k = 5

    Args:
        n: num qubits
        j: todo: target/control location of CNOT
        k: todo: target/control location of CNOT

    Returns:
        a unitary of n qubits with CNOT placed at ``j`` and ``k``.
    """
    if j < k:
        unitary = np.kron(
            np.kron(np.eye(2 ** (j - 1)), [[1, 0], [0, 0]]), np.eye(2 ** (n - j))
        ) + np.kron(
            np.kron(
                np.kron(np.kron(np.eye(2 ** (j - 1)), [[0, 0], [0, 1]]), np.eye(2 ** (k - j - 1))),
                [[0, 1], [1, 0]],
            ),
            np.eye(2 ** (n - k)),
        )
    else:
        unitary = np.kron(
            np.kron(np.eye(2 ** (j - 1)), [[1, 0], [0, 0]]), np.eye(2 ** (n - j))
        ) + np.kron(
            np.kron(
                np.kron(np.kron(np.eye(2 ** (k - 1)), [[0, 1], [1, 0]]), np.eye(2 ** (j - k - 1))),
                [[0, 0], [0, 1]],
            ),
            np.eye(2 ** (n - j)),
        )
    return unitary


# TODO: do we need a dedicated function?
def op_rx(phi) -> np.ndarray:
    """

    Args:
        phi:

    Returns:
        an RX rotation matrix
    """
    return RXGate(phi).to_matrix()
    # return np.array(
    #     [[np.cos(phi / 2), -1j * np.sin(phi / 2)], [-1j * np.sin(phi / 2), np.cos(phi / 2)]]
    # )


# TODO: replace with Qiskit
def op_ry(phi) -> np.ndarray:
    """

    Args:
        phi:

    Returns:
        an RY rotation matrix
    """
    return np.array([[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]])


def op_rz(phi) -> np.ndarray:
    """

    Args:
        phi:

    Returns:
        an RZ rotation matrix
    """
    return RZGate(phi).to_matrix()
    # return np.array([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]])


# TODO: replace with the Qiskit's implementation
# TODO: is not used
# def mcx_gate_matrix(num_qubits: int, make_su: bool = True) -> np.ndarray:
#     """
#     Generates a multi-control CX gate as a Numpy matrix.
#     Equivalent to:
#         qc = QuantumCircuit(nqubits)
#         qc.mcx([0, ..., nqubits-2], nqubits-1)
#         qiskit_matrix = Operator(qc.reverse_bits()).data
#
#     N O T E, the above Qiskit code outputs generic unitary operator, whereas
#     we optimize for SU one. In order to obtain SU matrix, the "qiskit_matrix"
#     should be scaled accordingly, similar to what is done in the code below.
#
#     Args:
#         num_qubits: total number of qubits, should be within [2 .. 16] interval.
#         make_su: generate SU matrix, if True, otherwise a generic unitary one.
#
#     Returns:
#         MCX gate matrix.
#     """
#     assert isinstance(num_qubits, (int, np.int64)) and 2 <= num_qubits <= 16
#     assert isinstance(make_su, bool)
#     d = int(2 ** num_qubits)
#     U = np.eye(d, dtype=np.cfloat)
#     U[d - 2 : d, d - 2 : d] = [[0, 1], [1, 0]]
#     if make_su:  # make SU matrix from the generic unitary one
#         U /= (np.linalg.det(U) + 0j) ** (1 / d)
#     return U


# TODO: replace with Qiskit
# TODO: is not used
# def toffoli_gate(n) -> np.ndarray:
#     # Generate a Toffoli gate
#     d = int(2 ** n)
#     U = np.eye(d)
#     U[d - 2 : d, d - 2 : d] = [[0, 1], [1, 0]]
#     U = U / ((la.det(U) + 0j) ** (1 / d))
#     return U


# TODO: replace with Qiskit
# TODO: is not used
# def fredkin_gate() -> np.ndarray:
#     # Generate a Fredkin gate with 3 qubits
#     n = 3
#     d = int(2 ** n)
#     U = np.eye(d)
#     U[3:6, 3:6] = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
#     U = U / ((la.det(U) + 0j) ** (1 / d))
#     return U
