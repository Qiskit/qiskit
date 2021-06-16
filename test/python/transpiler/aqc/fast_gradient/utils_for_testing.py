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

import os
import sys

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import numpy as np
import qiskit.transpiler.synthesis.aqc.fast_gradient.fast_grad_utils as myu


def _pp(perm: np.ndarray):
    """Prints a permutation. The least significant bit is on the left."""
    assert isinstance(perm, np.ndarray) and perm.dtype == np.int64
    assert np.all(perm >= 0)
    if np.amax(perm) >= 2 ** 8:
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
    """Prints an integer number as a binary string.
    The least significant bit is on the left."""
    assert isinstance(x, int) and 0 <= x
    if x >= 2 ** 8:
        print("WARNING: index value should occupy 8 bits or less")
        return
    print("{0:08b}".format(x)[::-1])


def RelativeError(A: np.ndarray, B: np.ndarray) -> float:
    """
    Computes relative residual between two matrices in Frobenius norm.
    """
    assert isinstance(A, np.ndarray) and isinstance(B, np.ndarray)
    return float(np.linalg.norm(A - B, "fro")) / float(np.linalg.norm(B, "fro"))


def MakeUnitVector(pos: int, nbits: int) -> np.ndarray:
    """
    Makes a unit vector e = (0 ... 0 1 0 ... 0) of size 2^nbits
    with unit at the position 'num'.
    N O T E: result depends on bit ordering.
    :param pos: position of unit in vector.
    :param nbits: number of meaningful bit in the number 'pos'.
    :return: unit vector of size 2^nbits.
    """
    assert isinstance(pos, int) and isinstance(nbits, int)
    assert 0 <= pos < 2 ** nbits
    vec = np.full((2 ** nbits,), fill_value=0, dtype=np.int64)
    vec[myu.reverse_bits(pos, nbits, enable=not myu.is_natural_bit_ordering())] = 1
    return vec


def I(n: int):
    """
    Creates a unit matrix with integer entries.
    :param n: number of bits.
    :return: unit matrix of size 2^n with integer entries.
    """
    assert isinstance(n, int) and n >= 0
    return np.eye(2 ** n, dtype=np.int64)


def Kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """
    Computes Kronecker product of 3 matrices.
    """
    return np.kron(a, np.kron(b, c))


def Kron5(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, e: np.ndarray) -> np.ndarray:
    """
    Computes Kronecker product of 5 matrices.
    """
    return np.kron(np.kron(np.kron(np.kron(a, b), c), d), e)


def RandMatrix(dim: int, kind: str = "complex") -> np.ndarray:
    """
    Generates a random complex or integer value matrix.
    :param dim: matrix size dim-x-dim.
    :param kind: "complex" or "randint".
    """
    assert isinstance(dim, int) and dim > 0
    assert isinstance(kind, str) and (kind in {"complex", "randint"})
    if kind == "complex":
        return (
            np.random.rand(dim, dim).astype(np.cfloat)
            + np.random.rand(dim, dim).astype(np.cfloat) * 1j
        )
    else:
        return np.random.randint(low=1, high=100, size=(dim, dim), dtype=np.int64)


def MakeTestMatrices2x2(n: int, k: int, kind: str = "complex") -> (np.ndarray, np.ndarray):
    """
    Creates a 2^n x 2^n random matrix made as a Kronecker product of identity
    ones and a single 1-qubit gate. This models a layer in quantum circuit with
    an arbitrary 1-qubit gate somewhere in the middle.
    :param n: number of qubits.
    :param k: index of qubit a 1-qubit gate is acting on.
    :param kind: entries of the output matrix are defined as:
                 "complex", "primes" or "randint".
    :return: (1) 2^n x 2^n random matrix; (2) 2 x 2 matrix of 1-qubit gate
             used for matrix construction.
    TODO: bit ordering might be an issue in case of natural ordering???
    """
    assert isinstance(n, int) and isinstance(k, int)
    assert 0 <= k < n and 2 <= n <= myu.get_max_num_bits()
    assert isinstance(kind, str) and (kind in {"complex", "primes", "randint"})
    if kind == "primes":
        A = np.array([[2, 3], [5, 7]], dtype=np.int64)
    else:
        A = RandMatrix(dim=2, kind=kind)
    M = Kron3(I(k), A, I(n - k - 1))
    return M, A


def MakeTestMatrices4x4(n: int, j: int, k: int, kind: str = "complex") -> (np.ndarray, np.ndarray):
    """
    Creates a 2^n x 2^n random matrix made as a Kronecker product of identity
    ones and a single 2-qubit gate. This models a layer in quantum circuit with
    an arbitrary 2-qubit gate somewhere in the middle.
    :param n: number of qubits.
    :param j: index of the first qubit the 2-qubit gate acting on.
    :param k: index of the second qubit the 2-qubit gate acting on.
    :param kind: entries of the output matrix are defined as:
                 "complex", "primes" or "randint".
    :return: (1) 2^n x 2^n random matrix; (2) 4 x 4 matrix of 2-qubit gate
             used for matrix construction.
    TODO: bit ordering might be an issue in case of natural ordering.
    """
    assert isinstance(n, int) and isinstance(j, int) and isinstance(k, int)
    assert j != k and 0 <= j < n and 0 <= k < n and 2 <= n <= myu.get_max_num_bits()
    assert isinstance(kind, str) and (kind in {"complex", "primes", "randint"})
    if kind == "primes":
        A = np.array([[2, 3], [5, 7]], dtype=np.int64)
        B = np.array([[11, 13], [17, 19]], dtype=np.int64)
        C = np.array([[47, 53], [41, 43]], dtype=np.int64)
        D = np.array([[31, 37], [23, 29]], dtype=np.int64)
    else:
        A, B = RandMatrix(dim=2, kind=kind), RandMatrix(dim=2, kind=kind)
        C, D = RandMatrix(dim=2, kind=kind), RandMatrix(dim=2, kind=kind)

    if j < k:
        M = Kron5(I(j), A, I(k - j - 1), B, I(n - k - 1)) + Kron5(
            I(j), C, I(k - j - 1), D, I(n - k - 1)
        )
    else:
        M = Kron5(I(k), B, I(j - k - 1), A, I(n - j - 1)) + Kron5(
            I(k), D, I(j - k - 1), C, I(n - j - 1)
        )
    G = np.kron(A, B) + np.kron(C, D)
    return M, G


if __name__ == "__main__":
    np.set_printoptions(precision=6, linewidth=256)
