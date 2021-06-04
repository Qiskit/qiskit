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

# Avoid excessive deprecation warnings in Qiskit on Linux system.
import warnings

import numpy as np
from scipy.stats import unitary_group

from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

warnings.filterwarnings("ignore", category=DeprecationWarning)


def check_num_qubits(nqubits: int) -> bool:
    """
    Checks validity of the number of qubits.
    """
    assert isinstance(nqubits, (int, np.int64))
    assert 2 <= nqubits <= 16
    return True


def circuit_to_numpy(circuit: (np.ndarray, QuantumCircuit)) -> np.ndarray:
    """
    Converts quantum circuit to Numpy matrix or returns just a copy,
    if the input is already a Numpy matrix.
    Args:
        circuit: the circuit to be converted into Numpy matrix.
    Returns:
        Numpy matrix underlying the input circuit.
    """
    if isinstance(circuit, QuantumCircuit):
        return Operator(circuit).data

    assert isinstance(circuit, np.ndarray)
    assert circuit.ndim == 2
    assert circuit.shape[0] == circuit.shape[1]
    assert circuit.dtype == np.cfloat
    return circuit.copy()  # returns a copy of the input matrix!


def compare_circuits(
    target_circuit: (np.ndarray, QuantumCircuit), approx_circuit: (np.ndarray, QuantumCircuit)
) -> float:
    """
    Compares two circuits (or their underlying matrices) for equivalence
    up to a global phase factor.
    Args:
        target_circuit: the circuit that we try to approximate.
        approx_circuit: the circuit obtained by approximate compiling.
    Returns:
        relative difference between two circuits.
    """
    U_t = circuit_to_numpy(target_circuit)
    U_a = circuit_to_numpy(approx_circuit)
    HS = np.trace(np.dot(U_a.conj().T, U_t))  # Hilbert–Schmidt inner product

    _BY_DOT = False
    if _BY_DOT:
        res = 1.0 - abs(HS) / U_t.shape[0]
    else:
        alpha = np.angle(HS)
        U_t *= np.exp(-1j * alpha)
        res = np.linalg.norm(U_t - U_a, "fro") / np.linalg.norm(U_t, "fro")
    return res


# def fidelity(target_circuit: (np.ndarray, QuantumCircuit),
#              approx_circuit: (np.ndarray, QuantumCircuit)) -> float:
#     """
#     Compares two circuits (or their underlying matrices) for equivalence
#     up to a global phase factor.
#     Args:
#         target_circuit: the circuit that we try to approximate.
#         approx_circuit: the circuit obtained by approximate compiling.
#     Returns:
#         relative difference between two circuits.
#     """
#     U_t = circuit_to_numpy(target_circuit)
#     U_a = circuit_to_numpy(approx_circuit)
#     HS = np.trace(np.dot(U_a.conj().T, U_t))    # Hilbert–Schmidt inner product
#     return (float(np.abs(HS)) / float(U_t.shape[0])) ** 2


def random_SU(nqubits: int) -> np.ndarray:
    """
    Generates a random SU matrix.
    Args:
        nqubits: number of qubits.
    Returns
        random SU matrix of size 2^n x 2^n.
    """
    EPS = float(np.sqrt(np.finfo(np.float64).eps))
    assert isinstance(nqubits, (int, np.int64)) and nqubits >= 2
    d = int(2 ** nqubits)
    U = unitary_group.rvs(d)
    U = U / (np.linalg.det(U) ** (1.0 / float(d)))
    assert U.dtype == np.cfloat
    assert abs(np.linalg.det(U) - 1.0) < EPS
    assert np.allclose(np.conj(U).T @ U, np.eye(d), atol=EPS, rtol=EPS)
    assert np.allclose(U @ np.conj(U).T, np.eye(d), atol=EPS, rtol=EPS)
    return U
