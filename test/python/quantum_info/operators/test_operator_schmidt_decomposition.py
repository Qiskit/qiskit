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

from test import QiskitTestCase, slow_test

import itertools
import unittest
from collections.abc import Iterable

import numpy as np
import numpy.testing as npt
from ddt import ddt, idata, unpack

from qiskit.exceptions import QiskitError
from qiskit.quantum_info import random_unitary
from qiskit.quantum_info.operators.operator_schmidt_decomposition import (
    operator_schmidt_decomposition,
)

# Tolerances consistent with Qiskit’s double-precision checks.
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT

ATOL = ATOL_DEFAULT
RTOL = RTOL_DEFAULT

# Always-on small seed set (keeps default runs fast and deterministic).
SEEDS_FAST = [7, 11, 19, 23, 42]

# Optional heavy stress, controlled by env var (e.g., QISKIT_SLOW_TESTS=1).
SEEDS_STRESS = list(range(100, 120))


def _fro_error(a_mat: np.ndarray, b_mat: np.ndarray) -> float:
    return np.linalg.norm(a_mat - b_mat, ord="fro")


# ---------- Helper case generators (evaluated at import time for DDT) ----------


def _cases_exact_unitary() -> Iterable[tuple[int, int, tuple[int, ...]]]:
    # (seed, n, subset_a)
    for seed in SEEDS_FAST:
        for n_qubits in (1, 2, 3):
            for r_size in range(1, n_qubits):
                for subset_a in itertools.combinations(range(n_qubits), r_size):
                    yield (seed, n_qubits, subset_a)


def _cases_exact_dense() -> Iterable[tuple[int, int, tuple[int, ...]]]:
    for seed in SEEDS_FAST:
        for n_qubits in (2, 3):  # n=1 would be trivial
            for r_size in range(1, n_qubits):
                for subset_a in itertools.combinations(range(n_qubits), r_size):
                    yield (seed, n_qubits, subset_a)


def _cases_qargs_order_irrelevant() -> Iterable[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    # removed unused local 'n_qubits'
    for seed in SEEDS_FAST:
        # Same subsets, different order:
        yield (seed, (2, 0), (0, 2))
        yield (seed, (1, 0), (0, 1))
        yield (seed, (2, 1), (1, 2))


def _cases_singular_values_props() -> Iterable[tuple[int]]:
    for seed in SEEDS_FAST:
        yield (seed,)


def _cases_rank1_kron() -> Iterable[tuple[int]]:
    for seed in SEEDS_FAST:
        yield (seed,)


def _cases_truncation_meta() -> Iterable[tuple[int, int, tuple[int, ...]]]:
    # (seed, n, subset_a)
    for seed in SEEDS_FAST:
        yield (seed, 3, (1,))
        yield (seed, 3, (0, 2))
        yield (seed, 4, (1, 2))


def _cases_truncation_low_rank() -> Iterable[tuple[int]]:
    # vary rank p; n=2 fixed inside
    return [(2,), (3,)]


def _cases_permutation(seed_list=None) -> Iterable[tuple[int, tuple[int, ...]]]:
    if seed_list is None:
        seed_list = SEEDS_FAST
    for seed in seed_list:
        for subset_a in [(0,), (1,), (2,), (0, 2)]:
            yield (seed, subset_a)


def _cases_k_validation() -> Iterable[tuple[int]]:
    return [(0,), (-3,)]


# ------------------------- Main test class (fast set) --------------------------


@ddt
class TestOperatorSchmidtDecomposition(QiskitTestCase):
    """Fast test suite for OSD."""

    @idata(list(_cases_exact_unitary()))
    @unpack
    def test_exact_reconstruction_random_unitary(
        self, seed: int, n_qubits: int, subset_a: tuple[int, ...]
    ):
        """Exact reconstruction (full sum) for random unitaries."""
        unitary = np.array(random_unitary(2**n_qubits, seed=seed), dtype=complex)
        out = operator_schmidt_decomposition(unitary, subset_a, return_reconstruction=True)
        self.assertAlmostEqual(_fro_error(unitary, out["reconstruction"]), 0.0, delta=ATOL)

    @idata(list(_cases_exact_dense()))
    @unpack
    def test_exact_reconstruction_random_dense(
        self, seed: int, n_qubits: int, subset_a: tuple[int, ...]
    ):
        """Exact reconstruction for random dense (nonunitary) operators."""
        rng = np.random.default_rng(seed)
        dim = 2**n_qubits
        op = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        out = operator_schmidt_decomposition(op, subset_a, return_reconstruction=True)
        self.assertAlmostEqual(_fro_error(op, out["reconstruction"]), 0.0, delta=ATOL)

    @idata(list(_cases_qargs_order_irrelevant()))
    @unpack
    def test_qargs_order_irrelevant(
        self, seed: int, subset_a_perm1: tuple[int, ...], subset_a_perm2: tuple[int, ...]
    ):
        """Singular values are invariant under reordering within the same subset."""
        n_qubits = 3
        unitary = np.array(random_unitary(2**n_qubits, seed=seed), dtype=complex)
        out1 = operator_schmidt_decomposition(unitary, subset_a_perm1)
        out2 = operator_schmidt_decomposition(unitary, subset_a_perm2)
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
        n_qubits = 3
        rng = np.random.default_rng(seed)
        dim = 2**n_qubits
        op = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        out = operator_schmidt_decomposition(op, (1,))
        sing_vals = out["singular_values"]
        self.assertTrue(np.all(sing_vals >= -ATOL))  # nonnegative (within numerical noise)
        self.assertTrue(np.all(sing_vals[:-1] + ATOL >= sing_vals[1:]))  # descending
        fro_sq = np.linalg.norm(op, ord="fro") ** 2
        self.assertAlmostEqual(np.sum(sing_vals**2), fro_sq, delta=max(ATOL, RTOL * abs(fro_sq)))

    @idata(list(_cases_singular_values_props()))
    @unpack
    def test_schmidt_factors_orthogonality(self, seed: int):
        """Hilbert–Schmidt orthogonality and normalization of Schmidt factors."""
        n_qubits = 3
        rng = np.random.default_rng(seed)
        dim = 2**n_qubits
        op = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        subset_a = (0, 2)
        out = operator_schmidt_decomposition(op, subset_a)
        sing_vals = out["singular_values"]
        a_factors = out["A_factors"]
        b_factors = out["B_factors"]

        gram_a = np.array(
            [[np.vdot(x_mat.ravel(), y_mat.ravel()) for y_mat in a_factors] for x_mat in a_factors]
        )
        gram_b = np.array(
            [[np.vdot(x_mat.ravel(), y_mat.ravel()) for y_mat in b_factors] for x_mat in b_factors]
        )
        diag_s = np.diag(sing_vals)
        npt.assert_allclose(gram_a, diag_s, rtol=RTOL, atol=ATOL)
        npt.assert_allclose(gram_b, diag_s, rtol=RTOL, atol=ATOL)

        # Orthonormality after normalization
        a_norm = [
            x_mat / np.sqrt(sing_vals[i]) if sing_vals[i] > 0 else x_mat
            for i, x_mat in enumerate(a_factors)
        ]
        b_norm = [
            y_mat / np.sqrt(sing_vals[i]) if sing_vals[i] > 0 else y_mat
            for i, y_mat in enumerate(b_factors)
        ]
        gram_a_n = np.array(
            [[np.vdot(x_mat.ravel(), y_mat.ravel()) for y_mat in a_norm] for x_mat in a_norm]
        )
        gram_b_n = np.array(
            [[np.vdot(x_mat.ravel(), y_mat.ravel()) for y_mat in b_norm] for x_mat in b_norm]
        )
        identity_mat = np.eye(len(sing_vals), dtype=complex)
        npt.assert_allclose(gram_a_n, identity_mat, rtol=RTOL, atol=ATOL)
        npt.assert_allclose(gram_b_n, identity_mat, rtol=RTOL, atol=ATOL)

    @idata(list(_cases_rank1_kron()))
    @unpack
    def test_rank1_kron_has_single_singular_value(self, seed: int):
        """A ⊗ B has a single nonzero Schmidt value: ||A||_F * ||B||_F."""
        rng = np.random.default_rng(seed)
        a_mat = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        b_mat = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        op = np.kron(a_mat, b_mat)
        out = operator_schmidt_decomposition(op, qargs=[1])
        sing_vals = out["singular_values"]
        s0 = np.linalg.norm(a_mat, ord="fro") * np.linalg.norm(b_mat, ord="fro")
        self.assertAlmostEqual(sing_vals[0], s0, delta=max(ATOL, RTOL * abs(s0)))
        if len(sing_vals) > 1:
            npt.assert_allclose(sing_vals[1:], np.zeros_like(sing_vals[1:]), rtol=RTOL, atol=ATOL)

    @idata(list(_cases_k_validation()))
    @unpack
    def test_k_validation(self, k_bad: int):
        """Non‑positive k should raise QiskitError (parameterized)."""
        op = np.eye(4, dtype=complex)
        with self.assertRaises(QiskitError):
            operator_schmidt_decomposition(op, [0], k=k_bad)

    # --- Truncation & permutation ---

    @idata(list(_cases_truncation_meta()))
    @unpack
    def test_truncation_frobenius_optimality_and_metadata(
        self, seed: int, n_qubits: int, subset_a: tuple[int, ...]
    ):
        """Top‑k truncation gives Frobenius‑optimal tail error and correct metadata."""
        rng = np.random.default_rng(seed)
        dim = 2**n_qubits
        op = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))

        # Try multiple k values per case for robustness
        for k_val in (1, 2, 3):
            out = operator_schmidt_decomposition(op, subset_a, k=k_val, return_reconstruction=True)
            sing_vals = out["singular_values"]
            total_terms = len(sing_vals)

            kept = out["truncation"]["kept_terms"]
            discarded = out["truncation"]["discarded_terms"]
            self.assertEqual(kept, min(k_val, total_terms))
            self.assertEqual(discarded, max(0, total_terms - k_val))

            expected_tail = np.sqrt(np.sum(sing_vals[k_val:] ** 2)) if k_val < total_terms else 0.0
            fro_err = out["truncation"]["frobenius_error"]
            self.assertAlmostEqual(
                fro_err, expected_tail, delta=max(ATOL, RTOL * max(1.0, expected_tail))
            )

            denom = np.linalg.norm(sing_vals)
            rel_expected = (expected_tail / denom) if denom > 0 else 0.0
            self.assertAlmostEqual(
                out["truncation"]["relative_frobenius_error"],
                rel_expected,
                delta=max(ATOL, RTOL * max(1.0, rel_expected)),
            )

            # Reconstruction is in original order; its error equals tail error.
            op_rec = out["reconstruction"]
            self.assertIsInstance(op_rec, np.ndarray)
            self.assertAlmostEqual(
                _fro_error(op, op_rec),
                expected_tail,
                delta=max(ATOL, RTOL * max(1.0, expected_tail)),
            )

    @idata(list(_cases_truncation_low_rank()))
    @unpack
    def test_truncation_exact_low_rank_sum_of_krons(self, rank_terms: int):
        """Operators with Schmidt rank p are reconstructed exactly when k >= p."""
        rng = np.random.default_rng(123 + rank_terms)  # vary seed with p
        a_list = [rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) for _ in range(rank_terms)]
        b_list = [rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2)) for _ in range(rank_terms)]
        op = sum(np.kron(a_mat, b_mat) for a_mat, b_mat in zip(a_list, b_list))

        out_full = operator_schmidt_decomposition(
            op, qargs=[1], k=rank_terms, return_reconstruction=True
        )
        self.assertAlmostEqual(_fro_error(op, out_full["reconstruction"]), 0.0, delta=1e-10)

        out_k1 = operator_schmidt_decomposition(op, qargs=[1], k=1, return_reconstruction=True)
        sing_vals = out_k1["singular_values"]
        expected_tail = np.sqrt(np.sum(sing_vals[1:] ** 2)) if len(sing_vals) > 1 else 0.0
        if expected_tail > 0:
            self.assertAlmostEqual(
                _fro_error(op, out_k1["reconstruction"]),
                expected_tail,
                delta=max(ATOL, RTOL * max(1.0, expected_tail)),
            )

    @idata(list(_cases_permutation()))
    @unpack
    def test_permutation_new_order_and_matrix_contract(self, seed: int, subset_a: tuple[int, ...]):
        """new_order == Sc + S and P maps U to the basis where sum kron(A,B) holds."""
        n_qubits = 3
        unitary = np.array(random_unitary(2**n_qubits, seed=seed), dtype=complex)
        out = operator_schmidt_decomposition(unitary, subset_a)
        part_info = out["partition"]
        perm_info = out["permutation"]

        expected_order = tuple(part_info["Sc"]) + tuple(part_info["S"])
        self.assertEqual(tuple(perm_info["new_order"]), expected_order)

        perm_matrix = perm_info["matrix"]
        self.assertEqual(perm_matrix.shape, (2**n_qubits, 2**n_qubits))
        npt.assert_allclose(perm_matrix @ perm_matrix.T, np.eye(2**n_qubits), rtol=RTOL, atol=ATOL)
        npt.assert_allclose(perm_matrix.T @ perm_matrix, np.eye(2**n_qubits), rtol=RTOL, atol=ATOL)
        self.assertTrue(np.all((np.abs(perm_matrix) < ATOL) | (np.abs(perm_matrix - 1) < ATOL)))

        # In the permuted basis, Up == sum_i kron(A_i, B_i) (full, untruncated case).
        out_full = operator_schmidt_decomposition(unitary, subset_a, k=None)
        a_factors = out_full["A_factors"]
        b_factors = out_full["B_factors"]
        up_from_factors = np.zeros_like(unitary, dtype=np.complex128)
        for a_mat, b_mat in zip(a_factors, b_factors):
            up_from_factors += np.kron(a_mat, b_mat)
        up_direct = perm_matrix @ unitary @ perm_matrix.T
        npt.assert_allclose(up_from_factors, up_direct, rtol=1e-11, atol=1e-11)


# ----------------------------- Stress tests (DDT) ------------------------------


@ddt
class TestOperatorSchmidtDecompositionStress(QiskitTestCase):
    """Stress tests (marked as @slow_test)."""

    @slow_test
    @idata(list(_cases_permutation(SEEDS_STRESS)))
    @unpack
    def test_exact_reconstruction_unitary_stress(self, seed: int, subset_a: tuple[int, ...]):
        """Stress: exact reconstruction over many seeds/partitions (unitary inputs)."""
        n_qubits = 3
        unitary = np.array(random_unitary(2**n_qubits, seed=seed), dtype=complex)
        out = operator_schmidt_decomposition(unitary, subset_a, return_reconstruction=True)
        self.assertAlmostEqual(_fro_error(unitary, out["reconstruction"]), 0.0, delta=ATOL)

    @slow_test
    @idata([(seed,) for seed in SEEDS_STRESS])
    @unpack
    def test_singular_values_properties_stress(self, seed: int):
        """Stress: SV nonnegativity/ordering and Frobenius identity (dense random inputs)."""
        n_qubits = 3
        rng = np.random.default_rng(seed)
        dim = 2**n_qubits
        op = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        out = operator_schmidt_decomposition(op, [1])
        sing_vals = out["singular_values"]
        self.assertTrue(np.all(sing_vals >= -ATOL))
        self.assertTrue(np.all(sing_vals[:-1] + ATOL >= sing_vals[1:]))
        fro_sq = np.linalg.norm(op, ord="fro") ** 2
        self.assertAlmostEqual(np.sum(sing_vals**2), fro_sq, delta=max(ATOL, RTOL * abs(fro_sq)))


if __name__ == "__main__":
    unittest.main()
