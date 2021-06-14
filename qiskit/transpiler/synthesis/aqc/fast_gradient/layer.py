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

import numpy as np

# from typing import Union
import fast_grad_utils as myu


class LayerBase:
    """
    Base class for any layer implementation.
    """

    def __init__(self):
        pass

    def set_from_matrix(self, _: np.ndarray):
        raise NotImplementedError("set_from_matrix()")

    def get_attr(self) -> (np.ndarray, np.ndarray, np.ndarray):
        raise NotImplementedError("get_attr()")


class Layer1Q(LayerBase):
    """
    Layer represents a simple circuit where 1-qubit gate matrix (of size 2x2)
    interleaves with identity matrices.
    """

    def __init__(self, nbits: int, k: int, g2x2: (np.ndarray, None) = None):
        """
        Constructor.
        """
        super(Layer1Q, self).__init__()
        assert isinstance(nbits, int) and 1 <= nbits <= myu.get_max_num_bits()
        assert isinstance(k, int) and 0 <= k < nbits

        self._n = nbits  # number of bits
        self._k = k  # index of the bit where gate is applied

        # 2x2 gate matrix (1-qubit gate).
        self._g = np.full((2, 2), fill_value=0, dtype=np.cfloat)
        self._tmp_g = np.full_like(self._g, fill_value=0)
        if isinstance(g2x2, np.ndarray):
            assert g2x2.shape == (2, 2)
            np.copyto(self._g, g2x2)

        bit_flip = True
        N = 2 ** nbits
        row_perm = myu.reverse_bits(myu.bit_permutation_1q(n=nbits, k=k), nbits=nbits, enable=bit_flip)
        col_perm = myu.reverse_bits(np.arange(N, dtype=np.int64), nbits=nbits, enable=bit_flip)
        self._perm = np.full((N,), fill_value=0, dtype=np.int64)
        self._perm[row_perm] = col_perm
        self._inv_perm = myu.inverse_permutation(self._perm)

    def set_from_matrix(self, g2x2: np.ndarray):
        """Update this layer from an external 2x2 gate matrix."""
        assert isinstance(g2x2, np.ndarray) and g2x2.shape == (2, 2)
        np.copyto(self._g, g2x2)

    def get_attr(self) -> (np.ndarray, np.ndarray, np.ndarray):
        """Returns gate matrix, direct and inverse permutations."""
        return self._g, self._perm, self._inv_perm


class Layer2Q(LayerBase):
    """
    Layer represents a simple circuit where 2-qubit gate matrix interleaves
    with identity matrices. The i
    """

    def __init__(self, nbits: int, j: int, k: int, g4x4: (np.ndarray, None) = None):
        """
        Constructor.
        """
        super(Layer2Q, self).__init__()
        assert isinstance(nbits, int) and 2 <= nbits <= myu.get_max_num_bits()
        assert isinstance(j, int) and isinstance(k, int) and j != k
        assert 0 <= j < nbits and 0 <= k < nbits

        self._n = nbits  # number of bits
        self._j = j  # index of the first (control) bit
        self._k = k  # index of the second (target) bit

        # 4x4 gate matrix (2-qubit gate).
        self._g = np.full((4, 4), fill_value=0, dtype=np.cfloat)
        self._tmp_g = np.full_like(self._g, fill_value=0)
        if isinstance(g4x4, np.ndarray):
            assert g4x4.shape == (4, 4)
            np.copyto(self._g, g4x4)

        bit_flip = True  # isNaturalBitOrdering()
        N = 2 ** nbits
        row_perm = myu.reverse_bits(
            myu.bit_permutation_2q(n=nbits, j=j, k=k), nbits=nbits, enable=bit_flip
        )
        col_perm = myu.reverse_bits(np.arange(N, dtype=np.int64), nbits=nbits, enable=bit_flip)
        self._perm = np.full((N,), fill_value=0, dtype=np.int64)
        self._perm[row_perm] = col_perm
        self._inv_perm = myu.inverse_permutation(self._perm)

    def set_from_matrix(self, g4x4: np.ndarray):
        assert isinstance(g4x4, np.ndarray) and g4x4.shape == (4, 4)
        np.copyto(self._g, g4x4)

    def get_attr(self) -> (np.ndarray, np.ndarray, np.ndarray):
        return self._g, self._perm, self._inv_perm


def InitLayer1QMatrices(thetas: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Initializes 4 x 4 matrices of 2-qubit gates defined in the paper.
    :param thetas: L x 4 matrix of gate parameters for every layer, where
                   L is the number of layers.
    :param dst: destination array of size L x 4 x 4 that will receive gate
                matrices of each layer.
    :return: dst.
    """
    assert isinstance(thetas, np.ndarray) and isinstance(dst, np.ndarray)
    n = thetas.shape[0]
    assert thetas.shape == (n, 3) and thetas.dtype == np.float64
    assert dst.shape == (n, 2, 2) and dst.dtype == np.cfloat
    tmp = np.full((4, 2, 2), fill_value=0, dtype=np.cfloat)
    for k in range(n):
        th = thetas[k]
        a = myu.Rz(th[0], out=tmp[0])
        b = myu.Ry(th[1], out=tmp[1])
        c = myu.Rz(th[2], out=tmp[2])
        np.dot(np.dot(a, b, out=tmp[3]), c, out=dst[k])
    return dst


def InitLayer1QDerivativeMatrices(thetas: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Initializes 4 x 4 derivative matrices of 2-qubit gates defined in the paper.
    :param thetas: L x 4 matrix of gate parameters for every layer, where
                   L is the number of layers.
    :param dst: destination array of size L x 4 x 4 x 4 that will receive gate
                derivative matrices of each layer; there are 4 parameters per
                gate, hence, 4 derivative matrices per layer.
    :return: dst.
    """
    assert isinstance(thetas, np.ndarray) and isinstance(dst, np.ndarray)
    n = thetas.shape[0]
    assert thetas.shape == (n, 3) and thetas.dtype == np.float64
    assert dst.shape == (n, 3, 2, 2) and dst.dtype == np.cfloat
    y = np.array([[0, -0.5], [0.5, 0]], dtype=np.cfloat)
    z = np.array([[-0.5j, 0], [0, 0.5j]], dtype=np.cfloat)
    tmp = np.full((5, 2, 2), fill_value=0, dtype=np.cfloat)
    for k in range(n):
        th = thetas[k]
        a = myu.Rz(th[0], out=tmp[0])
        b = myu.Ry(th[1], out=tmp[1])
        c = myu.Rz(th[2], out=tmp[2])

        za = np.dot(z, a, out=tmp[3])
        np.dot(np.dot(za, b, out=tmp[4]), c, out=dst[k, 0])
        yb = np.dot(y, b, out=tmp[3])
        np.dot(a, np.dot(yb, c, out=tmp[4]), out=dst[k, 1])
        zc = np.dot(z, c, out=tmp[3])
        np.dot(a, np.dot(b, zc, out=tmp[4]), out=dst[k, 2])
    return dst


def InitLayer2QMatrices(thetas: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Initializes 4 x 4 matrices of 2-qubit gates defined in the paper.
    :param thetas: L x 4 matrix of gate parameters for every layer, where
                   L is the number of layers.
    :param dst: destination array of size L x 4 x 4 that will receive gate
                matrices of each layer.
    :return: dst.
    """
    assert isinstance(thetas, np.ndarray) and isinstance(dst, np.ndarray)
    L = thetas.shape[0]
    assert thetas.shape == (L, 4) and thetas.dtype == np.float64
    assert dst.shape == (L, 4, 4) and dst.dtype == np.cfloat
    for k in range(L):
        th = thetas[k]
        cs0 = np.cos(0.5 * th[0]).item()
        sn0 = np.sin(0.5 * th[0]).item()
        ep1 = np.exp(0.5j * th[1]).item()
        en1 = np.exp(-0.5j * th[1]).item()
        cs2 = np.cos(0.5 * th[2]).item()
        sn2 = np.sin(0.5 * th[2]).item()
        cs3 = np.cos(0.5 * th[3]).item()
        sn3 = np.sin(0.5 * th[3]).item()
        ep1cs0 = ep1 * cs0
        ep1sn0 = ep1 * sn0
        en1cs0 = en1 * cs0
        en1sn0 = en1 * sn0
        sn2cs3 = sn2 * cs3
        sn2sn3 = sn2 * sn3
        cs2cs3 = cs2 * cs3
        sn3cs2j = 1j * sn3 * cs2
        sn2sn3j = 1j * sn2sn3

        S = dst[k].ravel()
        S[:] = [
            -(sn2sn3j - cs2cs3) * en1cs0,
            -(sn2cs3 + sn3cs2j) * en1cs0,
            (sn2cs3 + sn3cs2j) * en1sn0,
            (sn2sn3j - cs2cs3) * en1sn0,
            (sn2cs3 - sn3cs2j) * en1cs0,
            (sn2sn3j + cs2cs3) * en1cs0,
            -(sn2sn3j + cs2cs3) * en1sn0,
            (-sn2cs3 + sn3cs2j) * en1sn0,
            (-sn2sn3j + cs2cs3) * ep1sn0,
            -(sn2cs3 + sn3cs2j) * ep1sn0,
            -(sn2cs3 + sn3cs2j) * ep1cs0,
            (-sn2sn3j + cs2cs3) * ep1cs0,
            (sn2cs3 - sn3cs2j) * ep1sn0,
            (sn2sn3j + cs2cs3) * ep1sn0,
            (sn2sn3j + cs2cs3) * ep1cs0,
            (sn2cs3 - sn3cs2j) * ep1cs0,
        ]
    return dst


def InitLayer2QDerivativeMatrices(thetas: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """
    Initializes 4 x 4 derivative matrices of 2-qubit gates defined in the paper.
    :param thetas: L x 4 matrix of gate parameters for every layer, where
                   L is the number of layers.
    :param dst: destination array of size L x 4 x 4 x 4 that will receive gate
                derivative matrices of each layer; there are 4 parameters per
                gate, hence, 4 derivative matrices per layer.
    :return: dst.
    """
    assert isinstance(thetas, np.ndarray) and isinstance(dst, np.ndarray)
    L = thetas.shape[0]
    assert thetas.shape == (L, 4) and thetas.dtype == np.float64
    assert dst.shape == (L, 4, 4, 4) and dst.dtype == np.cfloat
    for k in range(L):
        th = thetas[k]
        cs0 = np.cos(0.5 * th[0]).item()
        sn0 = np.sin(0.5 * th[0]).item()
        ep1 = np.exp(0.5j * th[1]).item() * 0.5
        en1 = np.exp(-0.5j * th[1]).item() * 0.5
        cs2 = np.cos(0.5 * th[2]).item()
        sn2 = np.sin(0.5 * th[2]).item()
        cs3 = np.cos(0.5 * th[3]).item()
        sn3 = np.sin(0.5 * th[3]).item()
        ep1cs0 = ep1 * cs0
        ep1sn0 = ep1 * sn0
        en1cs0 = en1 * cs0
        en1sn0 = en1 * sn0
        sn2cs3 = sn2 * cs3
        sn2sn3 = sn2 * sn3
        sn3cs2 = sn3 * cs2
        cs2cs3 = cs2 * cs3
        sn2cs3j = 1j * sn2cs3
        sn2sn3j = 1j * sn2sn3
        sn3cs2j = 1j * sn3cs2
        cs2cs3j = 1j * cs2cs3

        S = dst[k, 0].ravel()
        S[:] = [
            (sn2sn3j - cs2cs3) * en1sn0,
            (sn2cs3 + sn3cs2j) * en1sn0,
            (sn2cs3 + sn3cs2j) * en1cs0,
            (sn2sn3j - cs2cs3) * en1cs0,
            (-sn2cs3 + sn3cs2j) * en1sn0,
            -(sn2sn3j + cs2cs3) * en1sn0,
            -(sn2sn3j + cs2cs3) * en1cs0,
            (-sn2cs3 + sn3cs2j) * en1cs0,
            (-sn2sn3j + cs2cs3) * ep1cs0,
            -(sn2cs3 + sn3cs2j) * ep1cs0,
            (sn2cs3 + sn3cs2j) * ep1sn0,
            (sn2sn3j - cs2cs3) * ep1sn0,
            (sn2cs3 - sn3cs2j) * ep1cs0,
            (sn2sn3j + cs2cs3) * ep1cs0,
            -(sn2sn3j + cs2cs3) * ep1sn0,
            (-sn2cs3 + sn3cs2j) * ep1sn0,
        ]

        S = dst[k, 1].ravel()
        S[:] = [
            -(sn2sn3 + cs2cs3j) * en1cs0,
            (sn2cs3j - sn3cs2) * en1cs0,
            -(sn2cs3j - sn3cs2) * en1sn0,
            (sn2sn3 + cs2cs3j) * en1sn0,
            -(sn2cs3j + sn3cs2) * en1cs0,
            (sn2sn3 - cs2cs3j) * en1cs0,
            (-sn2sn3 + cs2cs3j) * en1sn0,
            (sn2cs3j + sn3cs2) * en1sn0,
            (sn2sn3 + cs2cs3j) * ep1sn0,
            (-sn2cs3j + sn3cs2) * ep1sn0,
            (-sn2cs3j + sn3cs2) * ep1cs0,
            (sn2sn3 + cs2cs3j) * ep1cs0,
            (sn2cs3j + sn3cs2) * ep1sn0,
            (-sn2sn3 + cs2cs3j) * ep1sn0,
            (-sn2sn3 + cs2cs3j) * ep1cs0,
            (sn2cs3j + sn3cs2) * ep1cs0,
        ]

        S = dst[k, 2].ravel()
        S[:] = [
            -(sn2cs3 + sn3cs2j) * en1cs0,
            (sn2sn3j - cs2cs3) * en1cs0,
            -(sn2sn3j - cs2cs3) * en1sn0,
            (sn2cs3 + sn3cs2j) * en1sn0,
            (sn2sn3j + cs2cs3) * en1cs0,
            (-sn2cs3 + sn3cs2j) * en1cs0,
            (sn2cs3 - sn3cs2j) * en1sn0,
            -(sn2sn3j + cs2cs3) * en1sn0,
            -(sn2cs3 + sn3cs2j) * ep1sn0,
            (sn2sn3j - cs2cs3) * ep1sn0,
            (sn2sn3j - cs2cs3) * ep1cs0,
            -(sn2cs3 + sn3cs2j) * ep1cs0,
            (sn2sn3j + cs2cs3) * ep1sn0,
            (-sn2cs3 + sn3cs2j) * ep1sn0,
            (-sn2cs3 + sn3cs2j) * ep1cs0,
            (sn2sn3j + cs2cs3) * ep1cs0,
        ]

        S = dst[k, 3].ravel()
        S[:] = [
            -(sn2cs3j + sn3cs2) * en1cs0,
            (sn2sn3 - cs2cs3j) * en1cs0,
            (-sn2sn3 + cs2cs3j) * en1sn0,
            (sn2cs3j + sn3cs2) * en1sn0,
            -(sn2sn3 + cs2cs3j) * en1cs0,
            (sn2cs3j - sn3cs2) * en1cs0,
            -(sn2cs3j - sn3cs2) * en1sn0,
            (sn2sn3 + cs2cs3j) * en1sn0,
            -(sn2cs3j + sn3cs2) * ep1sn0,
            (sn2sn3 - cs2cs3j) * ep1sn0,
            (sn2sn3 - cs2cs3j) * ep1cs0,
            -(sn2cs3j + sn3cs2) * ep1cs0,
            -(sn2sn3 + cs2cs3j) * ep1sn0,
            (sn2cs3j - sn3cs2) * ep1sn0,
            (sn2cs3j - sn3cs2) * ep1cs0,
            -(sn2sn3 + cs2cs3j) * ep1cs0,
        ]
    return dst
