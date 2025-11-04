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
Tests for operator Schmidt decomposition utility (unittest + DDT parameterization).
"""
from __future__ import annotations

import itertools
import os
import unittest
from typing import Iterable, Tuple

import numpy as np
import numpy.testing as npt
from ddt import ddt, data, idata, unpack  # DDT: data-driven unittest

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


# ---------- Helper case generators (evaluated at import time for DDT) ----------

def _cases_exact_unitary() -> Iterable[Tuple[int, int, Tuple[int, ...]]]:
    # (seed, n, S)
    for seed in SEEDS_FAST:
        for n in (1, 2, 3):
            for r in range(1, n):
                for S in itertools.combinations(range(n), r):
                    yield (seed, n, S)


def _cases_exact_dense() -> Iterable[Tuple[int, int, Tuple[int, ...]]]:
    for seed in SEEDS_FAST:
        for n in (2, 3):  # n=1 would be trivial
            for r in range(1, n):
                for S in itertools.combinations(range(n), r):
                    yield (seed, n, S)


def _cases_qargs_order_irrelevant() -> Iterable[Tuple[int, Tuple[int, ...], Tuple[int, ...]]]:
    n = 3
    for seed in SEEDS_FAST:
        # Same subsets, different order:
        yield (seed, (2, 0), (0, 2))
        yield (seed, (1, 0), (0, 1))
        yield (seed, (2, 1), (1, 2))


def _cases_singular_values_props() -> Iterable[Tuple[int]]:
    for seed in SEEDS_FAST:
        yield (seed,)


def _cases_rank1_kron() -> Iterable[Tuple[int]]:
    for seed in SEEDS_FAST:
        yield (seed,)


def _cases_truncation_meta() -> Iterable[Tuple[int, int, Tuple[int, ...]]]:
    # (seed, n, S), with several n/S combos; truncation k is provided per-test below
    for seed in SEEDS_FAST:
        yield (seed, 3, (1,))
        yield (seed, 3, (0, 2))
        yield (seed, 4, (1, 2))


def _cases_truncation_low_rank() -> Iterable[Tuple[int]]:
    # vary rank p via randomized inputs inside the test if needed; keep simple here
    return [(2,), (3,)]  # number of terms p to build; n=2 fixed inside


def _cases_permutation(seed_list=None) -> Iterable[Tuple[int, Tuple[int, ...]]]:
    if seed_list is None:
        seed_list = SEEDS_FAST
    n = 3
    for seed in seed_list:
        for S in [(0,), (1,), (2,), (0, 2)]:
            yield (seed, S)


def _cases_k_validation() -> Iterable[Tuple[int]]:
    return [(0,), (-3,)]


# ------------------------- Main test class (fast set) --------------------------

@ddt
class TestOperatorSchmidtDecomposition(QiskitTestCase):

    @idata(list(_cases_exact_unitary()))
    @unpack
    def test_exact_reconstruction_random_unitary(self, seed: int, n: int, S: Tuple[int, ...]):
        """Exact reconstruction (full Schmidt sum) for random unitaries (parameterized)."""
        U = np.array(random_unitary(2**n, seed=seed), dtype=complex)
        out = operator_schmidt_decomposition(U, S, return_reconstruction=True)
        self.assertAlmostEqual(_fro_error(U, out["reconstruction"]), 0.0, delta=ATOL)

    @idata(list(_cases_exact_dense()))
    @unpack
    def test_exact_reconstruction_random_dense(self, seed: int, n: int, S: Tuple[int, ...]):
        """Exact reconstruction for random dense (nonunitary) operators (parameterized)."""
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
        out = operator_schmidt_decomposition(A, S, return_reconstruction=True)
        self.assertAlmostEqual(_fro_error(A, out["reconstruction"]), 0.0, delta=ATOL)

    @idata(list(_cases_qargs_order_irrelevant()))
    @unpack
    def test_qargs_order_irrelevant(self, seed: int, S1: Tuple[int, ...], S2: Tuple[int, ...]):
        """Singular values are invariant under reordering within the same subset."""
        n = 3
        U = np.array(random_unitary(2**n, seed=seed), dtype=complex)
        out1 = operator_schmidt_decomposition(U, S1)
        out2 = operator_schmidt_decomposition(U, S2)
        npt.assert_allclose(
            np.sort(out1["singular_values"]),
            np.sort(out2["singular_values"]),
            rtol=RTOL,
            atol=ATOL,
        )

    @idata(list(_cases_singular_values_props()))
    @unpack
    def test_singular_values_properties(self, seed: int):
        """SV sanity: nonnegative, descending, and Frobenius identity."""
        n = 3
        rng = np.random.default_rng(seed)
        U = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
        out = operator_schmidt_decomposition(U, (1,))
        s = out["singular_values"]
        self.assertTrue(np.all(s >= -ATOL))  # nonnegative (within numerical noise)
        self.assertTrue(np.all(s[:-1] + ATOL >= s[1:]))  # descending from SVD
        fro2 = np.linalg.norm(U, ord="fro") ** 2
        self.assertAlmostEqual(np.sum(s**2), fro2, delta=max(ATOL, RTOL * abs(fro2)))

    @idata(list(_cases_singular_values_props()))
    @unpack
    def test_schmidt_factors_orthogonality(self, seed: int):
        """Hilbert–Schmidt orthogonality and normalization of Schmidt factors."""
        n = 3
        rng = np.random.default_rng(seed)
        A = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))
        S = (0, 2)
        out = operator_schmidt_decomposition(A, S)
        s = out["singular_values"]
        A_list = out["A_factors"]
        B_list = out["B_factors"]
        GA = np.array([[np.vdot(X.ravel(), Y.ravel()) for Y in A_list] for X in A_list])
        GB = np.array([[np.vdot(X.ravel(), Y.ravel()) for Y in B_list] for X in B_list])
        D = np.diag(s)
        npt.assert_allclose(GA, D, rtol=RTOL, atol=ATOL)
        npt.assert_allclose(GB, D, rtol=RTOL, atol=ATOL)
        # Orthonormality after normalization
        A_norm = [X / np.sqrt(s[i]) if s[i] > 0 else X for i, X in enumerate(A_list)]
        B_norm = [Y / np.sqrt(s[i]) if s[i] > 0 else Y for i, Y in enumerate(B_list)]
        GA_n = np.array([[np.vdot(X.ravel(), Y.ravel()) for Y in A_norm] for X in A_norm])
        GB_n = np.array([[np.vdot(X.ravel(), Y.ravel()) for Y in B_norm] for X in B_norm])
        I = np.eye(len(s), dtype=complex)
        npt.assert_allclose(GA_n, I, rtol=RTOL, atol=ATOL)
        npt.assert_allclose(GB_n, I, rtol=RTOL, atol=ATOL)

    @idata(list(_cases_rank1_kron()))
    @unpack
    def test_rank1_kron_has_single_singular_value(self, seed: int):
        """A ⊗ B has a single nonzero Schmidt value: ||A||_F * ||B||_F."""
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

    @idata(list(_cases_k_validation()))
    @unpack
    def test_k_validation(self, k_bad: int):
        """Non-positive k should raise QiskitError (parameterized)."""
        U = np.eye(4, dtype=complex)
        with self.assertRaises(QiskitError):
            operator_schmidt_decomposition(U, [0], k=k_bad)

    # --- Truncation & permutation ---

    @idata(list(_cases_truncation_meta()))
    @unpack
    def test_truncation_frobenius_optimality_and_metadata(self, seed: int, n: int, S: Tuple[int, ...]):
        """Top-k truncation gives Frobenius-optimal tail error and correct metadata."""
        rng = np.random.default_rng(seed)
        U = rng.normal(size=(2**n, 2**n)) + 1j * rng.normal(size=(2**n, 2**n))

        # Try multiple k values per case for robustness
        for k in (1, 2, 3):
            out = operator_schmidt_decomposition(U, S, k=k, return_reconstruction=True)
            s = out["singular_values"]
            total_terms = len(s)
            kept = out["truncation"]["kept_terms"]
            disc = out["truncation"]["discarded_terms"]
            self.assertEqual(kept, min(k, total_terms))
            self.assertEqual(disc, max(0, total_terms - k))
            expected_tail = np.sqrt(np.sum(s[k:] ** 2)) if k < total_terms else 0.0
            fro_err = out["truncation"]["frobenius_error"]
            self.assertAlmostEqual(fro_err, expected_tail, delta=max(ATOL, RTOL * max(1.0, expected_tail)))
            denom = np.linalg.norm(s)
            rel_expected = (expected_tail / denom) if denom > 0 else 0.0
            self.assertAlmostEqual(
                out["truncation"]["relative_frobenius_error"],
                rel_expected,
                delta=max(ATOL, RTOL * max(1.0, rel_expected)),
            )
            # Reconstruction is in original order; its error equals tail error.
            U_rec = out["reconstruction"]
            self.assertIsInstance(U_rec, np.ndarray)
            self.assertAlmostEqual(_fro_error(U, U_rec), expected_tail, delta=max(ATOL, RTOL * max(1.0, expected_tail)))

    @idata(list(_cases_truncation_low_rank()))
    @unpack
    def test_truncation_exact_low_rank_sum_of_krons(self, p: int):
        """Operators with Schmidt rank p are reconstructed exactly when k >= p."""
        n = 2
        rng = np.random.default_rng(123 + p)  # vary seed with p
        # Build U = sum_{i=1}^p kron(A_i, B_i)
        A_list = [rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) for _ in range(p)]
        B_list = [rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) for _ in range(p)]
        U = sum(np.kron(Ai, Bi) for Ai, Bi in zip(A_list, B_list))

        out = operator_schmidt_decomposition(U, qargs=[1], k=p, return_reconstruction=True)
        self.assertAlmostEqual(_fro_error(U, out["reconstruction"]), 0.0, delta=1e-10)

        out_k1 = operator_schmidt_decomposition(U, qargs=[1], k=1, return_reconstruction=True)
        s = out_k1["singular_values"]
        expected_tail = np.sqrt(np.sum(s[1:] ** 2)) if len(s) > 1 else 0.0
        if expected_tail > 0:
            self.assertAlmostEqual(
                _fro_error(U, out_k1["reconstruction"]),
                expected_tail,
                delta=max(ATOL, RTOL * max(1.0, expected_tail)),
            )

    @idata(list(_cases_permutation()))
    @unpack
    def test_permutation_new_order_and_matrix_contract(self, seed: int, S: Tuple[int, ...]):
        """new_order == Sc + S and P maps U to the basis where sum kron(A,B) holds."""
        n = 3
        U = np.array(random_unitary(2**n, seed=seed), dtype=complex)
        out = operator_schmidt_decomposition(U, S)
        part = out["partition"]
        perm = out["permutation"]

        expected_order = tuple(part["Sc"]) + tuple(part["S"])
        self.assertEqual(tuple(perm["new_order"]), expected_order)

        P = perm["matrix"]
        self.assertEqual(P.shape, (2**n, 2**n))
        npt.assert_allclose(P @ P.T, np.eye(2**n), rtol=RTOL, atol=ATOL)
        npt.assert_allclose(P.T @ P, np.eye(2**n), rtol=RTOL, atol=ATOL)
        self.assertTrue(np.all((np.abs(P) < ATOL) | (np.abs(P - 1) < ATOL)))

        # In the permuted basis, Up == sum_i kron(A_i, B_i) (full, untruncated case).
        out_full = operator_schmidt_decomposition(U, S, k=None)
        A_f = out_full["A_factors"]
        B_f = out_full["B_factors"]
        Up_from_factors = np.zeros_like(U, dtype=np.complex128)
        for Ai, Bi in zip(A_f, B_f):
            Up_from_factors += np.kron(Ai, Bi)
        Up_direct = P @ U @ P.T
        npt.assert_allclose(Up_from_factors, Up_direct, rtol=1e-11, atol=1e-11)


# ----------------------------- Stress tests (DDT) ------------------------------

@ddt
class TestOperatorSchmidtDecompositionStress(QiskitTestCase):

    @unittest.skipUnless(RUN_STRESS, "Set QISKIT_SLOW_TESTS=1 to enable stress tests")
    @idata(list(_cases_permutation(SEEDS_STRESS)))
    @unpack
    def test_exact_reconstruction_unitary_stress(self, seed: int, S: Tuple[int, ...]):
        n = 3
        U = np.array(random_unitary(2**n, seed=seed), dtype=complex)
        out = operator_schmidt_decomposition(U, S, return_reconstruction=True)
        self.assertAlmostEqual(_fro_error(U, out["reconstruction"]), 0.0, delta=ATOL)

    @unittest.skipUnless(RUN_STRESS, "Set QISKIT_SLOW_TESTS=1 to enable stress tests")
    @idata([(seed,) for seed in SEEDS_STRESS])
    @unpack
    def test_singular_values_properties_stress(self, seed: int):
        n = 3
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