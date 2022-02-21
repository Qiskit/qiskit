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
Utility functions in the fast gradient implementation.
"""

from typing import Union
import numpy as np


def get_max_num_bits() -> int:
    """
    Returns the maximum supported number of qubits handled by this approach.
    Note, 2^16 means that the size of circuit matrix is 2^16 x 2^16 = 2^32,
    which is huge.

    Returns:
        maximum supported number of qubits.
    """
    return int(16)


def is_natural_bit_ordering() -> bool:
    """
    Returns True, if so called "natural" bit ordering is adopted.
    I M P O R T A N T:
    The current code relies on "non-natural" bit ordering convention, where
    the state vector is defined as a chain of Kronecker products of individual
    qubit states as follows:
    |state> = |least-significant-qubit> kron ... kron |most-significant-qubit>
    In quantum computation another ("natural") bit ordering is adopted:
    |state> = |most-significant-qubit> kron ... kron |least-significant-qubit>
    Non-natural bit ordering is more linear algebra friendly. However, if we
    consider a "natural" representation of a number:
    |x> = |x1 x2 ... xn> = x1 * 2^{n-1} + x2 * 2^{n-2} + ... + xn * 2^{0}, where
    x1 is a 0/1 value of the most significant bit, xn is a 0/1 value of the
    least significant one, then representation becomes intuitive for enumerating
    the quantum states. For example, |0> = (1 0 ... 0), |1> = (0 1 ... 0),
    |2^n - 1> = (0 0 ... 1). In the case of non-natural settings, in order to
    make a state |k> we have to flip bit ordering in the bit representation
    of the number 'k'. This is the rational for the function ReverseBits().
    I M P O R T A N T:
    At the moment we apply ReverseBits() even in case of natural ordering,
    because we traverse a number starting from the least-significant bit. Had
    it done the other way, the ReverseBits() would be unnecessary.
    TODO: implement it.

    Returns:
        True, if so called "natural" bit ordering is adopted,
        otherwise returns False.
    """
    return False


def is_permutation(x: np.ndarray) -> bool:
    """
    Checks if array is really an index permutation.

    Args:
        1D-array of integers that supposedly represents a permutation.

    Returns:
        True, if array is really a permutation of indices.
    """
    return (
        isinstance(x, np.ndarray)
        and x.ndim == 1
        and x.dtype == np.int64
        and np.all(np.sort(x) == np.arange(x.size, dtype=np.int64))
    )


def reverse_bits(x: Union[int, np.ndarray], nbits: int, enable: bool) -> Union[int, np.ndarray]:
    """
    Reverses the bit order in a number of 'nbits' length.
    If 'x' is an array, then operation is applied to every entry.

    Args:
        x: either a single integer or an array of integers.
        nbits: number of meaningful bits in the number ``x``.
        enable: apply reverse operation, if enabled, otherwise leave unchanged.

    Returns:
        a number or array of numbers with reversed bits.
    """
    # assert isinstance(nbits, int) and 1 <= nbits <= get_max_num_bits()
    # assert isinstance(enable, bool)

    if not enable:
        if isinstance(x, int):
            # assert 0 <= x < 2**nbits
            pass
        else:
            # assert isinstance(x, np.ndarray) and x.dtype == np.int64
            # assert np.all((x >= 0) & (x < 2**nbits))
            x = x.copy()
        return x

    if isinstance(x, int):
        # assert 0 <= x < 2**nbits
        res = int(0)
    else:
        # assert isinstance(x, np.ndarray) and x.dtype == np.int64
        # assert np.all((x >= 0) & (x < 2**nbits))
        x = x.copy()
        res = np.full_like(x, fill_value=0)

    for _ in range(nbits):
        res <<= 1
        res |= x & 1
        x >>= 1
    return res


def swap_bits(num: int, a: int, b: int) -> int:
    """
    Swaps the bits at positions 'a' and 'b' in the number 'num'.

    Args:
        num: an integer number where bits should be swapped.
        a: index of the first bit to be swapped.
        b: index of the second bit to be swapped.

    Returns:
        the number with swapped bits.
    """
    x = ((num >> a) ^ (num >> b)) & 1
    return num ^ ((x << a) | (x << b))


def bit_permutation_1q(n: int, k: int) -> np.ndarray:
    """
    Constructs index permutation that brings a circuit consisting of a single
    1-qubit gate to "standard form": kron(I(2^n/2), G), as we call it. Here n
    is the number of qubits, G is a 2x2 gate matrix, I(2^n/2) is the identity
    matrix of size (2^n/2)x(2^n/2), and the full size of the circuit matrix is
    (2^n)x(2^n). Circuit matrix in standard form becomes block-diagonal (with
    sub-matrices G on the main diagonal). Multiplication of such a matrix and
    a dense one is much faster than generic dense-dense product. Moreover,
    we do not need to keep the entire circuit matrix in memory but just 2x2 G
    one. This saves a lot of memory when the number of qubits is large.

    Args:
        n: number of qubits.
        k: index of qubit where single 1-qubit gate is applied.

    Returns:
        permutation that brings the whole layer to the standard form.
    """
    # assert isinstance(n, int) and isinstance(k, int)
    # assert 0 <= k < n <= get_max_num_bits()
    perm = np.arange(2**n, dtype=np.int64)
    if k != n - 1:
        for v in range(2**n):
            perm[v] = swap_bits(v, k, n - 1)
    return perm


def bit_permutation_2q(n: int, j: int, k: int) -> np.ndarray:
    """
    Constructs index permutation that brings a circuit consisting of a single
    2-qubit gate to "standard form": kron(I(2^n/4), G), as we call it. Here n
    is the number of qubits, G is a 4x4 gate matrix, I(2^n/4) is the identity
    matrix of size (2^n/4)x(2^n/4), and the full size of the circuit matrix is
    (2^n)x(2^n). Circuit matrix in standard form becomes block-diagonal (with
    sub-matrices G on the main diagonal). Multiplication of such a matrix and
    a dense one is much faster than generic dense-dense product. Moreover,
    we do not need to keep the entire circuit matrix in memory but just 4x4 G
    one. This saves a lot of memory when the number of qubits is large.

    Args:
        n: number of qubits.
        j: index of control qubit where single 2-qubit gate is applied.
        k: index of target qubit where single 2-qubit gate is applied.

    Returns:
        permutation that brings the whole layer to the standard form.
    """
    # assert isinstance(n, int) and isinstance(j, int) and isinstance(k, int)
    # assert j != k and 0 <= j < n and 0 <= k < n and 2 <= n <= get_max_num_bits()
    dim = 2**n
    perm = np.arange(dim, dtype=np.int64)
    if j < n - 2:
        if k < n - 2:
            for v in range(dim):
                perm[v] = swap_bits(swap_bits(v, j, n - 2), k, n - 1)
        elif k == n - 2:
            for v in range(dim):
                perm[v] = swap_bits(swap_bits(v, n - 2, n - 1), j, n - 2)
        else:
            # assert k == n - 1
            for v in range(dim):
                perm[v] = swap_bits(v, j, n - 2)
    elif j == n - 2:
        if k < n - 2:
            for v in range(dim):
                perm[v] = swap_bits(v, k, n - 1)
        else:
            # assert k == n - 1
            pass
    else:
        # assert j == n - 1
        if k < n - 2:
            for v in range(dim):
                perm[v] = swap_bits(swap_bits(v, n - 2, n - 1), k, n - 1)
        else:
            # assert k == n - 2
            for v in range(dim):
                perm[v] = swap_bits(v, n - 2, n - 1)
    return perm


def inverse_permutation(perm: np.ndarray) -> np.ndarray:
    """
    Returns inverse permutation.

    Args:
        perm: permutation to be reversed.

    Returns:
        inverse permutation.
    """
    # assert is_permutation(perm)
    inv = np.zeros_like(perm)
    inv[perm] = np.arange(perm.size, dtype=np.int64)
    return inv


def make_rx(phi: float, out: np.ndarray) -> np.ndarray:
    """
    Makes a 2x2 matrix that corresponds to X-rotation gate.

    Args:
        phi: rotation angle.
        out: placeholder for the result (2x2, complex-valued matrix).

    Returns:
        rotation gate, same object as referenced by "out".
    """
    # assert isinstance(phi, float) and isinstance(out, np.ndarray)
    # assert out.shape == (2, 2) and out.dtype == np.cfloat
    a = 0.5 * phi
    cs, sn = np.cos(a).item(), -1j * np.sin(a).item()
    out[0, 0] = cs
    out[0, 1] = sn
    out[1, 0] = sn
    out[1, 1] = cs
    return out


def make_ry(phi: float, out: np.ndarray) -> np.ndarray:
    """
    Makes a 2x2 matrix that corresponds to Y-rotation gate.

    Args:
        phi: rotation angle.
        out: placeholder for the result (2x2, complex-valued matrix).

    Returns:
        rotation gate, same object as referenced by "out".
    """
    # assert isinstance(phi, float) and isinstance(out, np.ndarray)
    # assert out.shape == (2, 2) and out.dtype == np.cfloat
    a = 0.5 * phi
    cs, sn = np.cos(a).item(), np.sin(a).item()
    out[0, 0] = cs
    out[0, 1] = -sn
    out[1, 0] = sn
    out[1, 1] = cs
    return out


def make_rz(phi: float, out: np.ndarray) -> np.ndarray:
    """
    Makes a 2x2 matrix that corresponds to Z-rotation gate.

    Args:
        phi: rotation angle.
        out: placeholder for the result (2x2, complex-valued matrix).

    Returns:
        rotation gate, same object as referenced by "out".
    """
    # assert isinstance(phi, float) and isinstance(out, np.ndarray)
    # assert out.shape == (2, 2) and out.dtype == np.cfloat
    exp = np.exp(0.5j * phi).item()
    out[0, 0] = 1 / exp
    out[0, 1] = 0
    out[1, 0] = 0
    out[1, 1] = exp
    return out
