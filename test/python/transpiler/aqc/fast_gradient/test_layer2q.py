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
Tests for Layer2Q implementation.
"""
# TODO: remove print("\n{:s}\n{:s}\n{:s}\n".format("@" * 80, __doc__, "@" * 80))

import os
import sys
import time
import traceback

from qiskit.test import QiskitTestCase

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from random import randint
import numpy as np
import unittest
import qiskit.transpiler.synthesis.aqc.fast_gradient.layer as lr
from qiskit.transpiler.synthesis.aqc.fast_gradient.pmatrix import PMatrix
import utils_for_testing as tut


class TestLayer2q(QiskitTestCase):
    def test_layer2q_matrix(self):
        """
        Tests: (1) the correctness of Layer2Q matrix construction;
        (2) matrix multiplication interleaved with permutations.
        TODO: test fails for natural bit ordering.
        """
        print("\nrunning {:s}() ...".format(self.test_layer2q_matrix.__name__))
        start = time.time()
        mat_kind = "complex"
        EPS = 100.0 * np.finfo(float).eps
        max_rel_err = 0.0
        for n in range(2, 8 + 1):
            print("n:", n)
            N = 2 ** n
            I = tut.I(n)
            for j in range(n):
                for k in range(n):
                    if j == k:
                        continue
                    M = tut.RandMatrix(dim=N, kind=mat_kind)
                    T, G = tut.MakeTestMatrices4x4(n=n, j=j, k=k, kind=mat_kind)
                    lmat = lr.Layer2Q(nbits=n, j=j, k=k, g4x4=G)
                    G2, perm, inv_perm = lmat.get_attr()
                    self.assertTrue(M.dtype == T.dtype == G.dtype == G2.dtype)
                    self.assertTrue(np.all(G == G2))
                    self.assertTrue(np.all(I[perm].T == I[inv_perm]))

                    G = np.kron(tut.I(n - 2), G)

                    # T == P^t @ G @ P.
                    err = tut.RelativeError(T, I[perm].T @ G @ I[perm])
                    self.assertLess(err, EPS, "err = {:0.16f}".format(err))
                    max_rel_err = max(max_rel_err, err)

                    # Multiplication by permutation matrix of the left can be
                    # replaced by row permutations.
                    TM = T @ M
                    err1 = tut.RelativeError(I[perm].T @ G @ M[perm], TM)
                    err2 = tut.RelativeError((G @ M[perm])[inv_perm], TM)

                    # Multiplication by permutation matrix of the right can be
                    # replaced by column permutations.
                    MT = M @ T
                    err3 = tut.RelativeError(M @ I[perm].T @ G @ I[perm], MT)
                    err4 = tut.RelativeError((M[:, perm] @ G)[:, inv_perm], MT)

                    self.assertTrue(
                        err1 < EPS and err2 < EPS and err3 < EPS and err4 < EPS,
                        "err1 = {:f},  err2 = {:f},  "
                        "err3 = {:f},  err4 = {:f}".format(err1, err2, err3, err4),
                    )
                    max_rel_err = max(max_rel_err, err1, err2, err3, err4)
        print("test execution time: {:0.6f}".format(time.time() - start))
        print("max. relative error: {:0.16f}".format(max_rel_err))

    def test_pmatrix_class(self):
        """
        Test the class PMatrix.
        """
        print("\nrunning {:s}() ...".format(self.test_pmatrix_class.__name__))
        start = time.time()
        EPS = 100.0 * np.finfo(float).eps
        mat_kind = "complex"
        max_rel_err = 0.0
        for n in range(2, 8 + 1):
            print("n:", n)
            N = 2 ** n
            tmpA = np.ndarray((N, N), dtype=np.cfloat)
            tmpB = tmpA.copy()
            for _ in range(200):
                j0 = randint(0, n - 1)
                k0 = (j0 + randint(1, n - 1)) % n
                j1 = randint(0, n - 1)
                k1 = (j1 + randint(1, n - 1)) % n
                j2 = randint(0, n - 1)
                k2 = (j2 + randint(1, n - 1)) % n
                j3 = randint(0, n - 1)
                k3 = (j3 + randint(1, n - 1)) % n
                j4 = randint(0, n - 1)
                k4 = (j4 + randint(1, n - 1)) % n

                T0, G0 = tut.MakeTestMatrices4x4(n=n, j=j0, k=k0, kind=mat_kind)
                T1, G1 = tut.MakeTestMatrices4x4(n=n, j=j1, k=k1, kind=mat_kind)
                T2, G2 = tut.MakeTestMatrices4x4(n=n, j=j2, k=k2, kind=mat_kind)
                T3, G3 = tut.MakeTestMatrices4x4(n=n, j=j3, k=k3, kind=mat_kind)
                T4, G4 = tut.MakeTestMatrices4x4(n=n, j=j4, k=k4, kind=mat_kind)

                C0 = lr.Layer2Q(nbits=n, j=j0, k=k0, g4x4=G0)
                C1 = lr.Layer2Q(nbits=n, j=j1, k=k1, g4x4=G1)
                C2 = lr.Layer2Q(nbits=n, j=j2, k=k2, g4x4=G2)
                C3 = lr.Layer2Q(nbits=n, j=j3, k=k3, g4x4=G3)
                C4 = lr.Layer2Q(nbits=n, j=j4, k=k4, g4x4=G4)

                M = tut.RandMatrix(dim=N, kind=mat_kind)
                TTMTT = T0 @ T1 @ M @ np.conj(T2).T @ np.conj(T3).T

                pmat = PMatrix(M=M.copy())  # we use copy for testing (!)
                pmat.mul_left_q2(L=C1, temp_mat=tmpA)
                pmat.mul_left_q2(L=C0, temp_mat=tmpA)
                pmat.mul_right_q2(L=C2, temp_mat=tmpA)
                pmat.mul_right_q2(L=C3, temp_mat=tmpA)
                alt_TTMTT = pmat.finalize(temp_mat=tmpA)

                err1 = tut.RelativeError(alt_TTMTT, TTMTT)
                self.assertLess(err1, EPS, "relative error: {:f}".format(err1))

                prod = np.cfloat(np.trace(TTMTT @ T4))
                alt_prod = pmat.product_q2(L=C4, tmpA=tmpA, tmpB=tmpB)
                err2 = abs(alt_prod - prod) / abs(prod)
                self.assertLess(err2, EPS, "relative error: {:f}".format(err2))

                max_rel_err = max(max_rel_err, err1, err2)

        print("test execution time: {:0.6}".format(time.time() - start))
        print("max. relative error: {:0.16f}".format(max_rel_err))


if __name__ == "__main__":
    np.set_printoptions(precision=6, linewidth=256)
    try:
        unittest.main()
    except Exception as ex:
        print("message length:", len(str(ex)))
        traceback.print_exc()
