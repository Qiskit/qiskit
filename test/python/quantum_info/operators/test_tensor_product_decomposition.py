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

"""Robust tests for tensor_product_decomposition.

Covers:
- All bipartitions for n up to MAX_N (both search modes).
- All set partitions for n <= FULL_SET_PARTITIONS_UP_TO (both search modes).
- Sampled set partitions for FULL_SET_PARTITIONS_UP_TO < n <= MAX_N.
- Random Haar operators in best_only mode.
"""

import itertools
import unittest

from ddt import ddt, data
import numpy as np

from qiskit.quantum_info import random_unitary
from qiskit.quantum_info.operators.tensor_product_decomposition import (
    tensor_product_decomposition,
)

# =============================================================================
# ----------------------------- CONFIG KNOBS ----------------------------------
# =============================================================================
# Change MAX_N here; the rest adapts automatically.
# Don't set too high (>10) as tests will take a lot of time.
MAX_N = 7

# Enumerate ALL set partitions for n <= this value (kept modest for CI time).
FULL_SET_PARTITIONS_UP_TO = min(6, MAX_N)

# Number of random set partitions per n for FULL_SET_PARTITIONS_UP_TO < n <= MAX_N.
# (Adjust per CI/runtime needs; defaults are modest.)
SAMPLED_SET_PARTITIONS_PER_N = {
    7: 10,
    8: 10,  # useful if you bump MAX_N
    9: 10,
    10: 10,
}

# Seeds/budgets
BIPARTITION_SEEDS_PER_N = 2  # seeds per bipartition (product-building variety)
BEST_ONLY_RANDOM_SEEDS_PER_N = 1  # random Haar operators per n for best_only sanity

SEARCH_MODES = ("small_to_big", "big_to_small")


# =============================================================================
# -------------------------------- HELPERS ------------------------------------
# =============================================================================


def kron_msb_to_lsb(*factors: np.ndarray) -> np.ndarray:
    """Kronecker in MSB->LSB order (left arg acts on MSB)."""
    out = factors[0]
    for fac in factors[1:]:
        out = np.kron(out, fac)
    return out


def blocks_cover_and_disjoint(blocks, n_qubits: int) -> bool:
    """Return True iff `blocks` form a disjoint partition of {0,...,n_qubits-1}."""
    seen = set()
    for blk in blocks:
        for qb in blk:
            if qb in seen:
                return False
            seen.add(qb)
    return seen == set(range(n_qubits))


def canonicalize_blocks(blocks):
    """Return blocks canonicalized (sort items in blocks; sort blocks by (len, tuple))."""
    return tuple(sorted((tuple(sorted(b)) for b in blocks), key=lambda t: (len(t), t)))


def _perm_matrix_from_qubit_order(new_order, n_qubits):
    """Boolean permutation matrix P for little-endian qubits: U_new = P U_old P^T."""
    if len(new_order) != n_qubits or set(new_order) != set(range(n_qubits)):
        raise ValueError("new_order must be a permutation of range(n)")
    dim = 2**n_qubits
    indices = np.arange(dim, dtype=np.int64)
    bits = (indices[:, None] >> np.arange(n_qubits, dtype=np.int64)) & 1
    reordered_bits = bits[:, new_order]
    new_indices = np.sum(reordered_bits << np.arange(n_qubits, dtype=np.int64), axis=1)
    return np.eye(dim, dtype=bool)[:, new_indices]


def build_product_from_blocks(n_qubits, blocks, seeds):
    """Build exact product U for `blocks` (LSB->MSB) in the original little-endian order."""
    assert len(seeds) == len(blocks)
    mats = [random_unitary(2 ** len(blk), seed=sd).data for blk, sd in zip(blocks, seeds)]

    # Build MSB->LSB Kronecker so that block 0 truly lands on LSB after permutation.
    u_perm = mats[-1]  # start from MSB block
    for mat in reversed(mats[:-1]):  # fold down to LSB
        u_perm = np.kron(u_perm, mat)

    new_order = tuple(q for blk in blocks for q in blk)  # LSB block first
    perm_matrix = _perm_matrix_from_qubit_order(new_order, n_qubits)
    return perm_matrix.T @ u_perm @ perm_matrix


def enumerate_set_partitions(n_qubits):
    """Yield ALL set partitions of {0,...,n_qubits-1} (order-insensitive)."""

    def _backtrack(idx, acc_blocks):
        if idx == n_qubits:
            sorted_blocks = [tuple(sorted(b)) for b in acc_blocks]
            sorted_blocks = tuple(sorted(sorted_blocks, key=lambda t: (min(t), len(t), t)))
            yield sorted_blocks
            return

        # place idx into existing blocks
        for blk in acc_blocks:
            blk.append(idx)
            yield from _backtrack(idx + 1, acc_blocks)
            blk.pop()

        # start a new block with idx
        acc_blocks.append([idx])
        yield from _backtrack(idx + 1, acc_blocks)
        acc_blocks.pop()

    if n_qubits <= 0:
        return
    seen = set()
    for part in _backtrack(0, []):
        if part not in seen:
            seen.add(part)
            yield part


def sample_set_partitions(n_qubits, count, rng):
    """Sample `count` random set partitions of {0,...,n_qubits-1} via RGS sampling."""
    seen = set()
    attempts = 0
    max_attempts = max(100, 20 * count)
    while len(seen) < count and attempts < max_attempts:
        attempts += 1
        # random restricted-growth string (RGS)
        rgs = [0]
        max_label = 0
        for _ in range(1, n_qubits):
            lab = rng.integers(0, max_label + 2)
            rgs.append(lab)
            if lab == max_label + 1:
                max_label += 1
        # convert to blocks
        num_labels = max(rgs) + 1
        blocks = [[] for _ in range(num_labels)]
        for idx, lab in enumerate(rgs):
            blocks[lab].append(idx)
        part = tuple(
            sorted(
                (tuple(sorted(b)) for b in blocks),
                key=lambda t: (min(t), len(t), t),
            )
        )
        if part not in seen:
            seen.add(part)
    return list(seen)


# =============================================================================
# -------------------------------- TEST SUITES --------------------------------
# =============================================================================


@ddt
class TestTPDAllBipartitionsExact(unittest.TestCase):
    """Exhaustive bipartition tests for n in [2..MAX_N] in both search modes."""

    @data(*range(2, MAX_N + 1))
    def test_all_bipartitions_exact_both_modes(self, n_qubits):
        """All S|Sc for this n: exactness, canonical blocks, permutation, reconstruction."""
        for k_size in range(1, n_qubits // 2 + 1):
            for subset_s in itertools.combinations(range(n_qubits), k_size):
                subset_sc = tuple(q for q in range(n_qubits) if q not in subset_s)
                # LSB->MSB blocks = (subset_s, subset_sc) for deterministic product-building
                blocks = (tuple(sorted(subset_s)), tuple(sorted(subset_sc)))
                for seed_offset in range(BIPARTITION_SEEDS_PER_N):
                    seeds = tuple(1000 + seed_offset * 10 + i for i, _ in enumerate(blocks))
                    u_op = build_product_from_blocks(n_qubits, blocks, seeds)
                    for search in SEARCH_MODES:
                        res = tensor_product_decomposition(
                            u_op,
                            mode="exact_only",
                            search=search,
                            return_operator=True,
                        )
                        with self.subTest(
                            n=n_qubits,
                            k=k_size,
                            s=subset_s,
                            seed_offset=seed_offset,
                            search=search,
                        ):
                            self.assertTrue(res.is_exact)
                            self.assertEqual(
                                canonicalize_blocks(res.blocks),
                                canonicalize_blocks(blocks),
                            )
                            self.assertTrue(blocks_cover_and_disjoint(res.blocks, n_qubits))
                            # Permutation & reconstruction
                            new_order = res.permutation["new_order"]
                            self.assertEqual(
                                new_order,
                                tuple(q for blk in res.blocks for q in blk),
                            )
                            perm_matrix = res.permutation["matrix"]
                            self.assertEqual(perm_matrix.dtype, bool)
                            np.testing.assert_allclose(
                                perm_matrix.T @ perm_matrix,
                                np.eye(2**n_qubits),
                                atol=1e-12,
                            )
                            np.testing.assert_allclose(res.reconstruction, u_op, atol=1e-12)


@ddt
class TestTPDAllAndSampledSetPartitions(unittest.TestCase):
    """All set partitions for small n; sampled partitions for larger n (up to MAX_N)."""

    @data(*range(2, FULL_SET_PARTITIONS_UP_TO + 1))
    def test_all_set_partitions_exact_both_modes(self, n_qubits):
        """Exactness & equality to expected blocks for every set partition (both modes)."""
        seed_base = 3000 + n_qubits * 100
        for idx, blocks in enumerate(enumerate_set_partitions(n_qubits)):
            seeds = tuple(seed_base + idx * 10 + i for i, _ in enumerate(blocks))
            u_op = build_product_from_blocks(n_qubits, blocks, seeds)
            for search in SEARCH_MODES:
                res = tensor_product_decomposition(
                    u_op, mode="exact_only", search=search, return_operator=True
                )
                with self.subTest(n=n_qubits, idx=idx, search=search):
                    self.assertTrue(res.is_exact)
                    self.assertEqual(
                        canonicalize_blocks(res.blocks),
                        canonicalize_blocks(blocks),
                    )
                    self.assertTrue(blocks_cover_and_disjoint(res.blocks, n_qubits))
                    np.testing.assert_allclose(res.reconstruction, u_op, atol=1e-12)

    @data(*range(FULL_SET_PARTITIONS_UP_TO + 1, MAX_N + 1))
    def test_sampled_set_partitions_exact_small_to_big(self, n_qubits):
        """Exactness & equality for sampled partitions in small_to_big (runtime-friendly)."""
        # Might be empty if MAX_N == FULL_SET_PARTITIONS_UP_TO
        samples = SAMPLED_SET_PARTITIONS_PER_N.get(n_qubits, 0)
        if samples <= 0:
            self.skipTest(f"No samples configured for n={n_qubits}")
        rng = np.random.default_rng(4000 + n_qubits)
        parts = sample_set_partitions(n_qubits, samples, rng)
        for idx, blocks in enumerate(parts):
            seeds = tuple(5000 + n_qubits * 100 + idx * 10 + i for i, _ in enumerate(blocks))
            u_op = build_product_from_blocks(n_qubits, blocks, seeds)
            res = tensor_product_decomposition(
                u_op, mode="exact_only", search="small_to_big", return_operator=True
            )
            with self.subTest(n=n_qubits, idx=idx):
                self.assertTrue(res.is_exact)
                self.assertEqual(
                    canonicalize_blocks(res.blocks),
                    canonicalize_blocks(blocks),
                )
                self.assertTrue(blocks_cover_and_disjoint(res.blocks, n_qubits))
                np.testing.assert_allclose(res.reconstruction, u_op, atol=1e-12)

    @data(*range(FULL_SET_PARTITIONS_UP_TO + 1, MAX_N + 1))
    def test_big_to_small_matches_small_on_sampled_partitions(self, n_qubits):
        """For sampled partitions, small_to_big and big_to_small must agree on blocks."""
        # Compare both modes on a small random subset (at most 15 partitions)
        samples = min(15, SAMPLED_SET_PARTITIONS_PER_N.get(n_qubits, 0))
        if samples <= 0:
            self.skipTest(f"No samples configured for n={n_qubits}")
        rng = np.random.default_rng(4500 + n_qubits)
        parts = sample_set_partitions(n_qubits, samples, rng)
        for idx, blocks in enumerate(parts):
            seeds = tuple(6000 + n_qubits * 100 + idx * 10 + i for i, _ in enumerate(blocks))
            u_op = build_product_from_blocks(n_qubits, blocks, seeds)
            res_small = tensor_product_decomposition(u_op, mode="exact_only", search="small_to_big")
            res_big = tensor_product_decomposition(u_op, mode="exact_only", search="big_to_small")
            with self.subTest(n=n_qubits, idx=idx):
                self.assertTrue(res_small.is_exact and res_big.is_exact)
                self.assertEqual(
                    canonicalize_blocks(res_small.blocks),
                    canonicalize_blocks(blocks),
                )
                self.assertEqual(
                    canonicalize_blocks(res_big.blocks),
                    canonicalize_blocks(blocks),
                )


@ddt
class TestTPDRandomBestOnlyUpToMaxN(unittest.TestCase):
    """Random Haar unitary sanity for best_only: two blocks and inexact."""

    @data(*range(2, MAX_N + 1))
    def test_random_best_only_two_block(self, n_qubits):
        """Random U: best_only must return exactly 2 blocks and report inexactness."""
        dim = 2**n_qubits
        for seed_idx in range(BEST_ONLY_RANDOM_SEEDS_PER_N):
            seed = 7000 + n_qubits * 10 + seed_idx
            u_op = random_unitary(dim, seed=seed).data
            # One mode here for time; both modes are exercised elsewhere.
            res = tensor_product_decomposition(u_op, mode="best_only", search="small_to_big")
            with self.subTest(n=n_qubits, seed=seed):
                self.assertFalse(res.is_exact)
                self.assertEqual(len(res.blocks), 2)
                self.assertTrue(blocks_cover_and_disjoint(res.blocks, n_qubits))
                # Unitaries after polar projection can be up to 2.0 away in relative Frobenius.
                self.assertTrue(0.0 < res.relative_residual <= 2.0)


if __name__ == "__main__":
    unittest.main()
