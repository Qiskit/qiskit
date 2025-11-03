# This code is part of Qiskit.
#
# (C) Copyright IBM 2025.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Tests for operator Schmidt decomposition utility (pure unittest, no pytest).
"""

from __future__ import annotations

import itertools
import os
import unittest
import numpy as np
import numpy.testing as npt
from test import QiskitTestCase

from qiskit.exceptions import QiskitError
from qiskit.quantum_info import random_unitary
from qiskit.quantum_info.operators.operator_schmidt_decomposition import (
    operator_schmidt_decomposition,
)

# Tolerances consistent with Terra’s double-precision checks.
ATOL = 1e-12
RTOL = 1e-12

# Always-on small seed set (keeps default runs fast and deterministic).
SEEDS_FAST = [7, 11, 19, 23, 42]

# Optional heavy stress, controlled by env var (e.g., QISKIT_SLOW_TESTS=1).
SEEDS_STRESS = list(range(100, 120))
RUN_STRESS = os.getenv("QISKIT_SLOW_TESTS", "0") not in ("", "0", "false", "False")


def _fro_error(a: np.ndarray, b: np.ndarray) -> float:
    return np.linalg.norm(a - b, ord="fro")


class TestOperatorSchmidtDecomposition(QiskitTestCase):

    def test_exact_reconstruction_random_unitary_multiple_seeds(self):
        """Exact reconstruction (full Schmidt sum) for random unitaries over several seeds."""
        for seed in SEEDS_FAST:
            for n in (1, 2, 3):
                U = np.array(random_unitary(2**n, seed=seed), dtype=complex)
                for r in range(1, n):
                    for S in itertools.combinations(range(n), r):
                        out = operator_schmidt_decomposition(U, S, return_reconstruction=True)
                        self.assertAlmostEqual(_fro_error(U, out["reconstruction"]), 0.0, delta=ATOL)

    def test_exact_reconstruction_random_dense_multiple_seeds(self):
        """Exact reconstruction for random dense (nonunitary) operators over several seeds."""
        for seed in SEEDS_FAST:
            for n in (2, 3):
                rng = np.random.default_rng(seed)
                A = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
                for r in range(1, n):
                    for S in itertools.combinations(range(n), r):
                        out = operator_schmidt_decomposition(A, S, return_reconstruction=True)
                        self.assertAlmostEqual(_fro_error(A, out["reconstruction"]), 0.0, delta=ATOL)

    def test_qargs_order_irrelevant(self):
        """Singular values are invariant under reordering within the same subset."""
        n = 3
        for seed in SEEDS_FAST:
            U = np.array(random_unitary(2**n, seed=seed), dtype=complex)
            S1 = (2, 0)  # same subset as (0, 2), different order
            S2 = (0, 2)
            out1 = operator_schmidt_decomposition(U, S1)
            out2 = operator_schmidt_decomposition(U, S2)
            npt.assert_allclose(
                np.sort(out1["singular_values"]),
                np.sort(out2["singular_values"]),
                rtol=RTOL,
                atol=ATOL,
            )

    def test_schmidt_factors_orthogonality(self):
        """Hilbert–Schmidt orthogonality and normalization of Schmidt factors."""
        n = 3
        for seed in SEEDS_FAST:
            rng = np.random.default_rng(seed)
            A = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
            S = (0, 2)
            out = operator_schmidt_decomposition(A, S)
            s = out["singular_values"]
            A_list = out["A_factors"]
            B_list = out["B_factors"]

            # Gram matrices under HS inner product -> diag(s)
            GA = np.array([[np.vdot(X.ravel(), Y.ravel()) for Y in A_list] for X in A_list])
            GB = np.array([[np.vdot(X.ravel(), Y.ravel()) for Y in B_list] for X in B_list])
            D = np.diag(s)
            npt.assert_allclose(GA, D, rtol=RTOL, atol=ATOL)
            npt.assert_allclose(GB, D, rtol=RTOL, atol=ATOL)

            # Normalized factors are orthonormal
            A_norm = [X / np.sqrt(s[i]) if s[i] > 0 else X for i, X in enumerate(A_list)]
            B_norm = [Y / np.sqrt(s[i]) if s[i] > 0 else Y for i, Y in enumerate(B_list)]
            GA_n = np.array([[np.vdot(X.ravel(), Y.ravel()) for Y in A_norm] for X in A_norm])
            GB_n = np.array([[np.vdot(X.ravel(), Y.ravel()) for Y in B_norm] for X in B_norm])
            I = np.eye(len(s), dtype=complex)
            npt.assert_allclose(GA_n, I, rtol=RTOL, atol=ATOL)
            npt.assert_allclose(GB_n, I, rtol=RTOL, atol=ATOL)

    def test_singular_values_properties(self):
        """SV sanity: nonnegative, descending, and Frobenius identity."""
        n = 3
        for seed in SEEDS_FAST:
            rng = np.random.default_rng(seed)
            U = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
            S = (1,)

            out = operator_schmidt_decomposition(U, S)
            s = out["singular_values"]

            # Nonnegative up to tiny numerical noise; descending from np.linalg.svd
            self.assertTrue(np.all(s >= -ATOL))
            self.assertTrue(np.all(s[:-1] + ATOL >= s[1:]))

            # Frobenius identity: sum(s**2) == ||U||_F^2
            fro2 = np.linalg.norm(U, ord="fro") ** 2
            self.assertAlmostEqual(np.sum(s**2), fro2, delta=max(ATOL, RTOL * abs(fro2)))

    def test_rank1_kron_has_single_singular_value(self):
        """A ⊗ B has a single nonzero Schmidt value: ||A||_F * ||B||_F."""
        for seed in SEEDS_FAST:
            rng = np.random.default_rng(seed)
            A = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            B = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
            U = np.kron(A, B)
            out = operator_schmidt_decomposition(U, qargs=[1])
            s = out["singular_values"]
            s0 = np.linalg.norm(A, ord="fro") * np.linalg.norm(B, ord="fro")
            self.assertAlmostEqual(s[0], s0, delta=max(ATOL, RTOL * abs(s0)))
            if len(s) > 1:
                npt.assert_allclose(s[1:], np.zeros_like(s[1:]), rtol=RTOL, atol=ATOL)

    def test_errors(self):
        """Input validation paths."""
        with self.assertRaises(QiskitError):
            operator_schmidt_decomposition(np.eye(6), [0])  # non power-of-two
        with self.assertRaises(QiskitError):
            operator_schmidt_decomposition(np.eye(4), [])  # empty S
        with self.assertRaises(QiskitError):
            operator_schmidt_decomposition(np.eye(4), [0, 1])  # full set (no complement)
        with self.assertRaises(QiskitError):
            operator_schmidt_decomposition(np.eye(4), [2])  # out of range for n=2

class TestOperatorSchmidtDecompositionStress(QiskitTestCase):
    @unittest.skipUnless(RUN_STRESS, "Set QISKIT_SLOW_TESTS=1 to enable stress tests")
    def test_exact_reconstruction_unitary_stress(self):
        n = 3
        for seed in SEEDS_STRESS:
            U = np.array(random_unitary(2**n, seed=seed), dtype=complex)
            for S in [(0,), (1,), (2,), (0, 2)]:
                out = operator_schmidt_decomposition(U, S, return_reconstruction=True)
                self.assertAlmostEqual(_fro_error(U, out["reconstruction"]), 0.0, delta=ATOL)

    @unittest.skipUnless(RUN_STRESS, "Set QISKIT_SLOW_TESTS=1 to enable stress tests")
    def test_singular_values_properties_stress(self):
        n = 3
        for seed in SEEDS_STRESS:
            rng = np.random.default_rng(seed)
            U = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
            out = operator_schmidt_decomposition(U, [1])
            s = out["singular_values"]
            self.assertTrue(np.all(s >= -ATOL))
            self.assertTrue(np.all(s[:-1] + ATOL >= s[1:]))
            fro2 = np.linalg.norm(U, ord="fro") ** 2
            self.assertAlmostEqual(np.sum(s**2), fro2, delta=max(ATOL, RTOL * abs(fro2)))


if __name__ == "__main__":
    unittest.main()