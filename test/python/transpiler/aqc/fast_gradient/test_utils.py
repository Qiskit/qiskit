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
print("\n{:s}\n{:s}\n{:s}\n".format("@" * 80, __doc__, "@" * 80))

import sys, os, time, traceback

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
import unittest
import numpy as np
import utils_for_testing as tut
import qiskit.transpiler.synthesis.aqc.fast_gradient.fast_grad_utils as myu
import random


def _Rx_old(phi: float) -> np.ndarray:
    u = [[np.cos(phi / 2), -1j * np.sin(phi / 2)], [-1j * np.sin(phi / 2), np.cos(phi / 2)]]
    return np.array(u, dtype=np.cfloat)


def _Ry_old(phi: float) -> np.ndarray:
    u = [[np.cos(phi / 2), -np.sin(phi / 2)], [np.sin(phi / 2), np.cos(phi / 2)]]
    return np.array(u, dtype=np.cfloat)


def _Rz_old(phi: float) -> np.ndarray:
    u = [[np.exp(-1j * phi / 2), 0], [0, np.exp(1j * phi / 2)]]
    return np.array(u, dtype=np.cfloat)


class TestUtils(unittest.TestCase):
    def test_reverse_bits(self):
        """
        Tests the function utils.ReverseBits().
        """
        print("\nrunning {:s}() ...".format(self.test_reverse_bits.__name__))
        start = time.time()
        nbits = myu.get_max_num_bits()
        for x in range(2 ** nbits):
            y = myu.reverse_bits(x, nbits, enable=True)
            self.assertTrue(x == myu.reverse_bits(y, nbits, enable=True))
            for i in range(nbits):
                self.assertTrue(((x >> i) & 1) == ((y >> (nbits - i - 1)) & 1))

        x = np.arange(2 ** nbits, dtype=np.int64)
        y = myu.reverse_bits(x, nbits, enable=True)
        self.assertTrue(np.all(myu.reverse_bits(y, nbits, enable=True) == x))
        print("test execution time:", time.time() - start)

    def test_make_unit_vector(self):
        """
        Tests the function utils_for_testing.MakeUnitVector().
        """
        print("\nrunning {:s}() ...".format(self.test_make_unit_vector.__name__))
        start = time.time()
        # Computes unit vector as Kronecker product of (1 0) or (0 1) sub-vectors.
        def _MakeUnitVectorStraightforward(num: int, nbits: int) -> np.ndarray:
            self.assertTrue(isinstance(num, int) and isinstance(nbits, int))
            self.assertTrue(0 <= num < 2 ** nbits)
            vec = 1
            for i in range(nbits):
                x = (num >> i) & 1
                if myu.is_natural_bit_ordering():
                    vec = np.kron(np.array([1 - x, x], dtype=np.int64), vec)
                else:
                    vec = np.kron(vec, np.array([1 - x, x], dtype=np.int64))
            self.assertTrue(vec.dtype == np.int64 and vec.shape == (2 ** nbits,))
            return vec

        max_nbits = 11
        for n in range(1, max_nbits + 1):
            for k in range(2 ** n):
                v1 = _MakeUnitVectorStraightforward(k, n)
                v2 = tut.MakeUnitVector(k, n)
                self.assertTrue(np.all(v1 == v2))
            print(".", end="", flush=True)
        print("")
        print("test execution time:", time.time() - start)

    def test_swap(self):
        """
        Tests the function utils.SwapBits().
        """
        print("\nrunning {:s}() ...".format(self.test_swap.__name__))
        start = time.time()
        nbits = myu.get_max_num_bits()
        for x in range(2 ** nbits):
            for a in range(nbits):
                for b in range(nbits):
                    y = myu.swap_bits(x, a, b)
                    self.assertTrue(x == myu.swap_bits(y, a, b))
                    self.assertTrue(((x >> a) & 1) == ((y >> b) & 1))
                    self.assertTrue(((x >> b) & 1) == ((y >> a) & 1))
        print("test execution time:", time.time() - start)

    def test_permutations(self):
        """
        Tests various properties of permutations and their compositions.
        """
        print("\nrunning {:s}() ...".format(self.test_permutations.__name__))
        total_start = time.time()

        def _Perm2Mat(perm: np.ndarray) -> np.ndarray:
            """Creates permutation matrix from a permutation."""
            return np.eye(perm.size, dtype=np.int64)[perm]

        def _PermutationAndMatrix(size: int) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
            """Creates direct and inverse permutations and their matrices."""
            perm = np.random.permutation(np.arange(size, dtype=np.int64))
            inv_perm = myu.inverse_permutation(perm)
            P, Q = _Perm2Mat(perm), _Perm2Mat(inv_perm)
            return perm, inv_perm, P, Q

        for n in range(1, 8 + 1):
            print("n =", n, end="", flush=True)
            start = time.time()

            N = 2 ** n
            for _ in range(100):
                perm1, inv_perm1, P1, Q1 = _PermutationAndMatrix(N)
                perm2, inv_perm2, P2, Q2 = _PermutationAndMatrix(N)

                I = tut.I(n)
                self.assertTrue(np.all(P1 == Q1.T) and np.all(P1.T == Q1))
                self.assertTrue(np.all(P2 == Q2.T) and np.all(P2.T == Q2))
                self.assertTrue(np.all(P1 @ Q1 == I) and np.all(Q1 @ P1 == I))
                self.assertTrue(np.all(P2 @ Q2 == I) and np.all(Q2 @ P2 == I))

                self.assertTrue(np.all(perm2[perm1] == np.take(perm2, perm1)))
                self.assertTrue(np.all(perm1[perm2] == np.take(perm1, perm2)))

                self.assertTrue(np.all(P1 @ P2 == _Perm2Mat(perm2[perm1])))
                self.assertTrue(np.all(P2 @ P1 == _Perm2Mat(perm1[perm2])))

                self.assertTrue(np.all(P1 @ Q2 == _Perm2Mat(inv_perm2[perm1])))
                self.assertTrue(np.all(Q2 @ P1 == _Perm2Mat(perm1[inv_perm2])))

                self.assertTrue(np.all(P2 @ Q1 == _Perm2Mat(inv_perm1[perm2])))
                self.assertTrue(np.all(Q1 @ P2 == _Perm2Mat(perm2[inv_perm1])))

                perm3, inv_perm3, P3, Q3 = _PermutationAndMatrix(N)
                perm4, inv_perm4, P4, Q4 = _PermutationAndMatrix(N)

                M = tut.RandMatrix(dim=N, kind="randint")

                PMQ = P1 @ M @ Q2
                self.assertTrue(np.all(PMQ == M[perm1][:, perm2]))
                self.assertTrue(np.all(PMQ == np.take(np.take(M, perm1, axis=0), perm2, axis=1)))
                PMQ = P1 @ M @ P2
                self.assertTrue(np.all(PMQ == M[perm1][:, inv_perm2]))
                self.assertTrue(
                    np.all(PMQ == np.take(np.take(M, perm1, axis=0), inv_perm2, axis=1))
                )

                PPMPQ = P1 @ P2 @ M @ P3 @ Q4
                self.assertTrue(np.all(PPMPQ == M[perm2][perm1][:, inv_perm3][:, perm4]))
                self.assertTrue(np.all(PPMPQ == M[perm2[perm1]][:, inv_perm3[perm4]]))
                self.assertTrue(
                    np.all(
                        PPMPQ == np.take(np.take(M, perm2[perm1], axis=0), inv_perm3[perm4], axis=1)
                    )
                )

            print(", time = {:0.6f}".format(time.time() - start))
        print("test execution time:", time.time() - total_start)

    def test_gates2x2(self):
        """
        Tests new implementation of 2x2 gate matrices vs the old one.
        """
        print("\nrunning {:s}() ...".format(self.test_gates2x2.__name__))
        total_start = time.time()
        _EPS = np.finfo(np.float64).eps * 2.0
        out = np.full((2, 2), fill_value=0, dtype=np.cfloat)
        for test in range(1000):
            phi = random.random() * 2.0 * np.pi
            self.assertTrue(np.allclose(myu.Rx(phi, out=out), _Rx_old(phi), atol=_EPS, rtol=_EPS))
            self.assertTrue(np.allclose(myu.Ry(phi, out=out), _Ry_old(phi), atol=_EPS, rtol=_EPS))
            self.assertTrue(np.allclose(myu.Rz(phi, out=out), _Rz_old(phi), atol=_EPS, rtol=_EPS))
            if test % 100 == 0:
                print(".", end="", flush=True)
        print("\ntest execution time:", time.time() - total_start)


if __name__ == "__main__":
    np.set_printoptions(precision=6, linewidth=256)
    try:
        unittest.main()
    except Exception as ex:
        print("message length:", len(str(ex)))
        traceback.print_exc()
