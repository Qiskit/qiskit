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
Tests for utility functions.
"""

# import os, sys; sys.path.append(os.getcwd()); print(sys.path) # to run in console

import sys
import random
from time import perf_counter
import unittest
import test.python.transpiler.aqc.fast_gradient.utils_for_testing as tut
import numpy as np
import qiskit.transpiler.synthesis.aqc.fast_gradient.fast_grad_utils as myu
from qiskit.test import QiskitTestCase

__glo_verbose__ = False


def _rx(phi: float) -> np.ndarray:
    """
    Returns Rx gate matrix. Function has obvious implementation but creates
    a temporary array upon every call. This can be inefficient in some cases.
    """
    return np.array(
        [[np.cos(phi / 2), -1j * np.sin(phi / 2)], [-1j * np.sin(phi / 2), np.cos(phi / 2)]],
        dtype=np.cfloat,
    )


def _ry(phi: float) -> np.ndarray:
    """
    Returns Ry gate matrix. Function has obvious implementation but creates
    a temporary array upon every call. This can be inefficient in some cases.
    """
    return np.array(
        [[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]], dtype=np.cfloat
    )


def _rz(phi: float) -> np.ndarray:
    """
    Returns Rz gate matrix. Function has obvious implementation but creates
    a temporary array upon every call. This can be inefficient in some cases.
    """
    return np.array([[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]], dtype=np.cfloat)


class TestUtils(QiskitTestCase):
    """
    Tests utility functions used by the fast gradient implementation.
    """

    def test_reverse_bits(self):
        """
        Tests the function utils.reverse_bits().
        """
        if __glo_verbose__:
            print("\nrunning {:s}() ...".format(self.test_reverse_bits.__name__))

        start = perf_counter()
        nbits = myu.get_max_num_bits() if __glo_verbose__ else 7
        for x in range(2**nbits):
            y = myu.reverse_bits(x, nbits, enable=True)
            self.assertTrue(x == myu.reverse_bits(y, nbits, enable=True))
            for i in range(nbits):
                self.assertTrue(((x >> i) & 1) == ((y >> (nbits - i - 1)) & 1))

        x = np.arange(2**nbits, dtype=np.int64)
        y = myu.reverse_bits(x, nbits, enable=True)
        self.assertTrue(np.all(myu.reverse_bits(y, nbits, enable=True) == x))

        if __glo_verbose__:
            print("test execution time:", perf_counter() - start)

    def test_make_unit_vector(self):
        """
        Tests the function utils_for_testing.make_unit_vector().
        """
        if __glo_verbose__:
            print("\nrunning {:s}() ...".format(self.test_make_unit_vector.__name__))

        start = perf_counter()

        # Computes unit vector as Kronecker product of (1 0) or (0 1) sub-vectors.
        def _make_unit_vector_simple(num: int, nbits: int) -> np.ndarray:
            self.assertTrue(isinstance(num, int) and isinstance(nbits, int))
            self.assertTrue(0 <= num < 2**nbits)
            vec = 1
            for i in range(nbits):
                x = (num >> i) & 1
                if myu.is_natural_bit_ordering():
                    vec = np.kron(np.array([1 - x, x], dtype=np.int64), vec)
                else:
                    vec = np.kron(vec, np.array([1 - x, x], dtype=np.int64))
            self.assertTrue(vec.dtype == np.int64 and vec.shape == (2**nbits,))
            return vec

        for n in range(1, (11 if __glo_verbose__ else 5) + 1):
            for k in range(2**n):
                v1 = _make_unit_vector_simple(k, n)
                v2 = tut.make_unit_vector(k, n)
                self.assertTrue(np.all(v1 == v2))

            if __glo_verbose__:
                print(".", end="", flush=True)

        if __glo_verbose__:
            print("")
            print("test execution time:", perf_counter() - start)

    def test_swap(self):
        """
        Tests the function utils.swap_bits().
        """
        if __glo_verbose__:
            print("\nrunning {:s}() ...".format(self.test_swap.__name__))

        start = perf_counter()
        nbits = myu.get_max_num_bits() if __glo_verbose__ else 7
        for x in range(2**nbits):
            for a in range(nbits):
                for b in range(nbits):
                    y = myu.swap_bits(x, a, b)
                    self.assertTrue(x == myu.swap_bits(y, a, b))
                    self.assertTrue(((x >> a) & 1) == ((y >> b) & 1))
                    self.assertTrue(((x >> b) & 1) == ((y >> a) & 1))

        if __glo_verbose__:
            print("test execution time:", perf_counter() - start)

    def test_permutations(self):
        """
        Tests various properties of permutations and their compositions.
        """
        if __glo_verbose__:
            print("\nrunning {:s}() ...".format(self.test_permutations.__name__))

        total_start = perf_counter()

        def _perm2_mat(perm: np.ndarray) -> np.ndarray:
            """Creates permutation matrix from a permutation."""
            return np.eye(perm.size, dtype=np.int64)[perm]

        def _permutation_and_matrix(size: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
            """Creates direct and inverse permutations and their matrices."""
            perm = np.random.permutation(np.arange(size, dtype=np.int64))
            inv_perm = myu.inverse_permutation(perm)
            p_mat, q_mat = _perm2_mat(perm), _perm2_mat(inv_perm)
            return perm, inv_perm, p_mat, q_mat

        for n in range(1, (8 if __glo_verbose__ else 5) + 1):
            if __glo_verbose__:
                print("n =", n, end="", flush=True)

            start = perf_counter()

            dim = 2**n
            for _ in range(100):
                perm1, inv_perm1, p1_mat, q1_mat = _permutation_and_matrix(dim)
                perm2, inv_perm2, p2_mat, q2_mat = _permutation_and_matrix(dim)

                iden = tut.identity_matrix(n)
                self.assertTrue(np.all(p1_mat == q1_mat.T) and np.all(p1_mat.T == q1_mat))
                self.assertTrue(np.all(p2_mat == q2_mat.T) and np.all(p2_mat.T == q2_mat))
                self.assertTrue(np.all(p1_mat @ q1_mat == iden) and np.all(q1_mat @ p1_mat == iden))
                self.assertTrue(np.all(p2_mat @ q2_mat == iden) and np.all(q2_mat @ p2_mat == iden))

                self.assertTrue(np.all(perm2[perm1] == np.take(perm2, perm1)))
                self.assertTrue(np.all(perm1[perm2] == np.take(perm1, perm2)))

                self.assertTrue(np.all(p1_mat @ p2_mat == _perm2_mat(perm2[perm1])))
                self.assertTrue(np.all(p2_mat @ p1_mat == _perm2_mat(perm1[perm2])))

                self.assertTrue(np.all(p1_mat @ q2_mat == _perm2_mat(inv_perm2[perm1])))
                self.assertTrue(np.all(q2_mat @ p1_mat == _perm2_mat(perm1[inv_perm2])))

                self.assertTrue(np.all(p2_mat @ q1_mat == _perm2_mat(inv_perm1[perm2])))
                self.assertTrue(np.all(q1_mat @ p2_mat == _perm2_mat(perm2[inv_perm1])))

                _, inv_perm3, p3_mat, _ = _permutation_and_matrix(dim)
                perm4, _, _, q4_mat = _permutation_and_matrix(dim)

                mat = tut.rand_matrix(dim=dim, kind="randint")

                pmq = p1_mat @ mat @ q2_mat
                self.assertTrue(np.all(pmq == mat[perm1][:, perm2]))
                self.assertTrue(np.all(pmq == np.take(np.take(mat, perm1, axis=0), perm2, axis=1)))
                pmq = p1_mat @ mat @ p2_mat
                self.assertTrue(np.all(pmq == mat[perm1][:, inv_perm2]))
                self.assertTrue(
                    np.all(pmq == np.take(np.take(mat, perm1, axis=0), inv_perm2, axis=1))
                )

                ppmpq = p1_mat @ p2_mat @ mat @ p3_mat @ q4_mat
                self.assertTrue(np.all(ppmpq == mat[perm2][perm1][:, inv_perm3][:, perm4]))
                self.assertTrue(np.all(ppmpq == mat[perm2[perm1]][:, inv_perm3[perm4]]))
                self.assertTrue(
                    np.all(
                        ppmpq
                        == np.take(np.take(mat, perm2[perm1], axis=0), inv_perm3[perm4], axis=1)
                    )
                )

            if __glo_verbose__:
                print(", time = {:0.6f}".format(perf_counter() - start))
        if __glo_verbose__:
            print("test execution time:", perf_counter() - total_start)

    def test_gates2x2(self):
        """
        Tests implementation of 2x2 gate matrices vs the ones that create and
        return temporary array upon every call (but have more compact code).
        """
        if __glo_verbose__:
            print("\nrunning {:s}() ...".format(self.test_gates2x2.__name__))

        total_start = perf_counter()
        _eps = np.finfo(np.float64).eps * 2.0
        out = np.full((2, 2), fill_value=0, dtype=np.cfloat)
        for test in range(1000 if __glo_verbose__ else 100):
            phi = random.random() * 2.0 * np.pi
            self.assertTrue(np.allclose(myu.make_rx(phi, out=out), _rx(phi), atol=_eps, rtol=_eps))
            self.assertTrue(np.allclose(myu.make_ry(phi, out=out), _ry(phi), atol=_eps, rtol=_eps))
            self.assertTrue(np.allclose(myu.make_rz(phi, out=out), _rz(phi), atol=_eps, rtol=_eps))
            if test % 100 == 0:
                if __glo_verbose__:
                    print(".", end="", flush=True)

        if __glo_verbose__:
            print("\ntest execution time:", perf_counter() - total_start)


if __name__ == "__main__":
    __glo_verbose__ = ("-v" in sys.argv) or ("--verbose" in sys.argv)
    np.set_printoptions(precision=6, linewidth=256)
    unittest.main()
