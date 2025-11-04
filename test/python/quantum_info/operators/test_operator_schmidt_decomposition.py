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

    # -------- Tests for truncation and permutation --------

    def test_truncation_frobenius_optimality_and_metadata(self):
        """Top-k truncation gives Frobenius-optimal tail error and correct metadata."""
        n = 3
        k = 3  # truncate
        for seed in SEEDS_FAST:
            rng = np.random.default_rng(seed)
            U = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
            S = (1,)  # arbitrary nontrivial bipartition
            out = operator_schmidt_decomposition(U, S, k=k, return_reconstruction=True)

            # Full spectrum (descending)
            s = out["singular_values"]
            total_terms = len(s)
            self.assertEqual(out["truncation"]["kept_terms"], min(k, total_terms))
            self.assertEqual(out["truncation"]["discarded_terms"], max(0, total_terms - k))

            # Expected Frobenius tail error from discarded singular values.
            expected_tail = np.sqrt(np.sum(s[k:] ** 2)) if k < total_terms else 0.0
            fro_err = out["truncation"]["frobenius_error"]
            self.assertAlmostEqual(fro_err, expected_tail, delta=max(ATOL, RTOL * max(1.0, expected_tail)))

            # Relative error uses ||s||_2 = sqrt(sum s^2) = ||U||_F
            denom = np.linalg.norm(s)
            rel_expected = (expected_tail / denom) if denom > 0 else 0.0
            self.assertAlmostEqual(
                out["truncation"]["relative_frobenius_error"],
                rel_expected,
                delta=max(ATOL, RTOL * max(1.0, rel_expected)),
            )

            # Reconstruction is returned in original order; its error equals tail Frobenius error.
            U_rec = out["reconstruction"]
            self.assertIsInstance(U_rec, np.ndarray)
            self.assertAlmostEqual(_fro_error(U, U_rec), expected_tail, delta=max(ATOL, RTOL * max(1.0, expected_tail)))

            # Kept factor lists have length k (or less if spectrum shorter).
            self.assertEqual(len(out["A_factors"]), min(k, total_terms))
            self.assertEqual(len(out["B_factors"]), min(k, total_terms))

    def test_truncation_exact_low_rank_sum_of_krons(self):
        """Operators with Schmidt rank p are reconstructed exactly when k >= p."""
        # Build U = sum_{i=1}^p kron(A_i, B_i) on n=2 with S=[1] (identity permutation).
        n = 2
        p = 2
        rng = np.random.default_rng(123)
        A1 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        B1 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        A2 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        B2 = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        U = np.kron(A1, B1) + np.kron(A2, B2)

        # With k=p we should reconstruct exactly (up to numerical tolerance).
        out = operator_schmidt_decomposition(U, qargs=[1], k=p, return_reconstruction=True)
        self.assertAlmostEqual(_fro_error(U, out["reconstruction"]), 0.0, delta=1e-10)

        # With k=1 we should incur nonzero error that matches the discarded tail.
        out_k1 = operator_schmidt_decomposition(U, qargs=[1], k=1, return_reconstruction=True)
        s = out_k1["singular_values"]
        expected_tail = np.sqrt(np.sum(s[1:] ** 2))
        self.assertGreater(expected_tail, 0.0)
        self.assertAlmostEqual(
            _fro_error(U, out_k1["reconstruction"]),
            expected_tail,
            delta=max(ATOL, RTOL * expected_tail),
        )

    def test_permutation_new_order_and_matrix_contract(self):
        """Check that new_order == Sc + S and P maps U to the basis where sum kron(A,B) holds."""
        n = 3
        for seed in SEEDS_FAST:
            U = np.array(random_unitary(2**n, seed=seed), dtype=complex)
            # Try several bipartitions
            for S in [(0,), (1,), (2,), (0, 2)]:
                out = operator_schmidt_decomposition(U, S)
                part = out["partition"]
                perm = out["permutation"]

                # new_order must equal Sc + S
                expected_order = tuple(part["Sc"]) + tuple(part["S"])
                self.assertEqual(tuple(perm["new_order"]), expected_order)

                # P must be a proper permutation: real 0/1; orthogonal (P P^T = I = P^T P)
                P = perm["matrix"]
                self.assertEqual(P.shape, (2**n, 2**n))
                npt.assert_allclose(P @ P.T, np.eye(2**n), rtol=RTOL, atol=ATOL)
                npt.assert_allclose(P.T @ P, np.eye(2**n), rtol=RTOL, atol=ATOL)
                # Entries are 0/1 (within tolerance)
                self.assertTrue(np.all((np.abs(P) < ATOL) | (np.abs(P - 1) < ATOL)))

                # In the permuted basis, Up == sum_i kron(A_i, B_i) (full, untruncated case).
                # Recompute with full set explicitly to be certain.
                out_full = operator_schmidt_decomposition(U, S, k=None)
                A_list = out_full["A_factors"]
                B_list = out_full["B_factors"]

                Up_from_factors = np.zeros_like(U, dtype=np.complex128)
                for Ai, Bi in zip(A_list, B_list):
                    Up_from_factors += np.kron(Ai, Bi)

                Up_direct = P @ U @ P.T
                npt.assert_allclose(Up_from_factors, Up_direct, rtol=1e-11, atol=1e-11)

    def test_k_validation(self):
        """Non-positive k should raise QiskitError."""
        U = np.eye(4, dtype=complex)
        with self.assertRaises(QiskitError):
            operator_schmidt_decomposition(U, [0], k=0)
        with self.assertRaises(QiskitError):
            operator_schmidt_decomposition(U, [0], k=-3)


class TestOperatorSchmidtDecompositionStress(QiskitTestCase):
    @unittest.skipUnless(RUN_STRESS, "Set QISKIT_SLOW_TESTS=1 to enable stress tests")
    def test_exact_reconstruction_unitary_stress(self):
        n = 3
        for seed in SEEDS_STRESS:
            U = np.array(random_unitary(2**n, seed=seed), dtype=complex)
            for S in [(0,), (1,), (2,), (0, 1), (0, 2), (1, 2)]:
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