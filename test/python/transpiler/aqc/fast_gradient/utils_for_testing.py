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
Utility functions for debugging and testing.
"""

from typing import Tuple
import numpy as np
from scipy.stats import unitary_group
import qiskit.transpiler.synthesis.aqc.fast_gradient.fast_grad_utils as myu


def _pp(perm: np.ndarray):
    """
    Prints a permutation. The least significant bit is on the left.
    """
    # assert isinstance(perm, np.ndarray) and perm.dtype == np.int64
    # assert np.all(perm >= 0)
    if np.amax(perm) >= 2**8:
        print("WARNING: index in permutation should occupy 8 bits or less")
        return
    for i in range(perm.size):
        print("{:3d}".format(i), " " * 6, end="")
    print("")
    for i in range(perm.size):
        print("{0:08b}".format(i)[::-1], " ", end="")
    print("")
    for i in range(perm.size):
        print("|  ||  |  ", end="")
    print("")
    for i in range(perm.size):
        print("{0:08b}".format(perm[i])[::-1], " ", end="")
    print("")
    for i in range(perm.size):
        print("{:3d}".format(perm[i]), " " * 6, end="")
    print("", flush=True)


def _pb(x: int):
    """
    Prints an integer number as a binary string.
    The least significant bit is on the left.
    """
    # assert isinstance(x, int) and x >= 0
    if x >= 2**8:
        print("WARNING: index value should occupy 8 bits or less")
        return
    print("{0:08b}".format(x)[::-1])


def relative_error(a_mat: np.ndarray, b_mat: np.ndarray) -> float:
    """
    Computes relative residual between two matrices in Frobenius norm.
    """
    # assert isinstance(a_mat, np.ndarray) and isinstance(b_mat, np.ndarray)
    return float(np.linalg.norm(a_mat - b_mat, "fro")) / float(np.linalg.norm(b_mat, "fro"))


def make_unit_vector(pos: int, nbits: int) -> np.ndarray:
    """
    Makes a unit vector e = (0 ... 0 1 0 ... 0) of size 2^nbits with
    unit at the position 'num'.
    N O T E: result depends on bit ordering.

    Args:
        pos: position of unit in vector.
        nbits: number of meaningful bit in the number 'pos'.

    Returns:
        unit vector of size 2^nbits.
    """
    # assert isinstance(pos, int) and isinstance(nbits, int)
    # assert 0 <= pos < 2**nbits
    vec = np.full((2**nbits,), fill_value=0, dtype=np.int64)
    vec[myu.reverse_bits(pos, nbits, enable=not myu.is_natural_bit_ordering())] = 1
    return vec


def identity_matrix(n: int) -> np.ndarray:
    """
    Creates a unit matrix with integer entries.

    Args:
        n: number of bits.

    Returns:
        unit matrix of size 2^n with integer entries.
    """
    # assert isinstance(n, int) and n >= 0
    return np.eye(2**n, dtype=np.int64)


def kron3(a_mat: np.ndarray, b_mat: np.ndarray, c_mat: np.ndarray) -> np.ndarray:
    """
    Computes Kronecker product of 3 matrices.
    """
    return np.kron(a_mat, np.kron(b_mat, c_mat))


def kron5(
    a_mat: np.ndarray, b_mat: np.ndarray, c_mat: np.ndarray, d_mat: np.ndarray, e_mat: np.ndarray
) -> np.ndarray:
    """
    Computes Kronecker product of 5 matrices.
    """
    return np.kron(np.kron(np.kron(np.kron(a_mat, b_mat), c_mat), d_mat), e_mat)


def rand_matrix(dim: int, kind: str = "complex") -> np.ndarray:
    """
    Generates a random complex or integer value matrix.

    Args:
        dim: matrix size dim-x-dim.
        kind: "complex" or "randint".

    Returns:
        a random matrix.
    """
    # assert isinstance(dim, int) and dim > 0
    # assert isinstance(kind, str) and (kind in {"complex", "randint"})
    if kind == "complex":
        return (
            np.random.rand(dim, dim).astype(np.cfloat)
            + np.random.rand(dim, dim).astype(np.cfloat) * 1j
        )
    else:
        return np.random.randint(low=1, high=100, size=(dim, dim), dtype=np.int64)


def make_test_matrices2x2(n: int, k: int, kind: str = "complex") -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a 2^n x 2^n random matrix made as a Kronecker product of identity
    ones and a single 1-qubit gate. This models a layer in quantum circuit with
    an arbitrary 1-qubit gate somewhere in the middle.

    Args:
        n: number of qubits.
        k: index of qubit a 1-qubit gate is acting on.
        kind: entries of the output matrix are defined as:
              "complex", "primes" or "randint".
    Returns:
        A tuple of (1) 2^n x 2^n random matrix; (2) 2 x 2 matrix of 1-qubit
          gate used for matrix construction.
    TODO: bit ordering might be an issue in case of natural ordering???
    """
    # assert isinstance(n, int) and isinstance(k, int)
    # assert 0 <= k < n and 2 <= n <= myu.get_max_num_bits()
    # assert isinstance(kind, str) and (kind in {"complex", "primes", "randint"})
    if kind == "primes":
        a_mat = np.array([[2, 3], [5, 7]], dtype=np.int64)
    else:
        a_mat = rand_matrix(dim=2, kind=kind)
    m_mat = kron3(identity_matrix(k), a_mat, identity_matrix(n - k - 1))
    return m_mat, a_mat


def make_test_matrices4x4(
    n: int, j: int, k: int, kind: str = "complex"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates a 2^n x 2^n random matrix made as a Kronecker product of identity
    ones and a single 2-qubit gate. This models a layer in quantum circuit with
    an arbitrary 2-qubit gate somewhere in the middle.

    Args:
        n: number of qubits.
        j: index of the first qubit the 2-qubit gate acting on.
        k: index of the second qubit the 2-qubit gate acting on.
        kind: entries of the output matrix are defined as:
              "complex", "primes" or "randint".

    Returns:
        A tuple of (1) 2^n x 2^n random matrix; (2) 4 x 4 matrix of
        2-qubit gate used for matrix construction.
    TODO: bit ordering might be an issue in case of natural ordering.
    """
    # assert isinstance(n, int) and isinstance(j, int) and isinstance(k, int)
    # assert j != k and 0 <= j < n and 0 <= k < n and 2 <= n <= myu.get_max_num_bits()
    # assert isinstance(kind, str) and (kind in {"complex", "primes", "randint"})
    if kind == "primes":
        a_mat = np.array([[2, 3], [5, 7]], dtype=np.int64)
        b_mat = np.array([[11, 13], [17, 19]], dtype=np.int64)
        c_mat = np.array([[47, 53], [41, 43]], dtype=np.int64)
        d_mat = np.array([[31, 37], [23, 29]], dtype=np.int64)
    else:
        a_mat, b_mat = rand_matrix(dim=2, kind=kind), rand_matrix(dim=2, kind=kind)
        c_mat, d_mat = rand_matrix(dim=2, kind=kind), rand_matrix(dim=2, kind=kind)

    if j < k:
        m_mat = kron5(
            identity_matrix(j), a_mat, identity_matrix(k - j - 1), b_mat, identity_matrix(n - k - 1)
        ) + kron5(
            identity_matrix(j), c_mat, identity_matrix(k - j - 1), d_mat, identity_matrix(n - k - 1)
        )
    else:
        m_mat = kron5(
            identity_matrix(k), b_mat, identity_matrix(j - k - 1), a_mat, identity_matrix(n - j - 1)
        ) + kron5(
            identity_matrix(k), d_mat, identity_matrix(j - k - 1), c_mat, identity_matrix(n - j - 1)
        )
    g_mat = np.kron(a_mat, b_mat) + np.kron(c_mat, d_mat)
    return m_mat, g_mat


def rand_circuit(num_qubits: int, depth: int) -> np.ndarray:
    """
    Generates a random circuit of unit blocks for debugging and testing.
    """
    blocks = np.tile(np.arange(num_qubits).reshape(num_qubits, 1), depth)
    for i in range(depth):
        np.random.shuffle(blocks[:, i])
    return blocks[0:2, :].copy()


def rand_su_mat(dim: int) -> np.ndarray:
    """
    Generates a random SU matrix.
    Args:
        dim: matrix size dim-x-dim.
    Returns:
        random SU matrix.
    """
    # tol = float(np.sqrt(np.finfo(np.float64).eps))
    # assert isinstance(dim, (int, np.int64)) and dim >= 4
    u_mat = unitary_group.rvs(dim)
    u_mat /= np.linalg.det(u_mat) ** (1.0 / float(dim))
    # assert u_mat.dtype == np.cfloat
    # assert abs(np.linalg.det(u_mat) - 1.0) < tol
    # assert np.allclose(np.conj(u_mat).T @ u_mat, np.eye(dim), atol=tol, rtol=tol)
    # assert np.allclose(u_mat @ np.conj(u_mat).T, np.eye(dim), atol=tol, rtol=tol)
    return u_mat
