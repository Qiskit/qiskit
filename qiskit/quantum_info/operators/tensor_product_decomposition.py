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
Tensor-product block decomposition for n-qubit operators.

Given an operator U acting on n qubits (unitary or not), this module provides a function that
finds the *most granular* tensor-product factorization over disjoint qubit sets when it exists,
and otherwise returns the *best* bipartition (Frobenius-optimal rank-1 Operator Schmidt term).

It relies on the Operator Schmidt Decomposition (OSD) utility available in
``qiskit.quantum_info``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Sequence

import itertools
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT, RTOL_DEFAULT

from .operator_schmidt_decomposition import operator_schmidt_decomposition


# -----------------------------------------------------------------------------
# Result dataclasses
# -----------------------------------------------------------------------------


@dataclass
class CutInfo:
    """Diagnostics for a tested/accepted bipartition."""

    subset_s: tuple[int, ...]
    subset_sc: tuple[int, ...]
    singular_values: np.ndarray
    frobenius_error: float
    relative_frobenius_error: float
    kept_terms: int = 1


@dataclass
class TensorProductDecomposition:
    """
    Structured result returned by :func:`tensor_product_decomposition`.

    Attributes
    ----------
    blocks:
        A partition of the qubit indices into non-overlapping blocks, listed in the *same order*
        used to build the final permutation (see below).
    factors:
        Local operators (np.ndarray) for each block, aligned with ``blocks``.
        With the ordering guarantees in this module, you can reconstruct the *permuted* operator as
        ``kron(factors[0], factors[1], ..., factors[m-1])`` (i.e., LSB block first in Kronecker
        order).
    operator_factors:
        Same as ``factors`` but wrapped as :class:`~qiskit.quantum_info.Operator` when
        ``return_operator=True``.
    is_exact:
        ``True`` when the factorization is exact within tolerances.
    residual, relative_residual:
        Frobenius norm of ``U - reconstruction`` and its normalized variant.
    cuts:
        List of :class:`CutInfo` captured during recursion (accepted splits). Useful to inspect
        singular-value spectra and errors per cut.
    permutation:
        Dict with:
          * ``new_order``: tuple[int] – the qubit order used by the final permutation (concatenation
            of ``blocks`` in the returned order; LSB block comes first).
          * ``matrix``: np.ndarray (dtype ``bool``) – permutation matrix ``P`` such that
            ``U = P^T U_perm P`` and ``U_perm = kron(factors[0],...,factors[m-1])``.
    reconstruction:
        The reconstructed operator (np.ndarray) in the *original* qubit order, present only when
        ``return_operator=True``.
    """

    blocks: tuple[tuple[int, ...], ...]
    factors: tuple[np.ndarray, ...]
    operator_factors: tuple[Operator, ...] | None
    is_exact: bool
    residual: float
    relative_residual: float
    cuts: list[CutInfo] = field(default_factory=list)
    permutation: dict[str, Any] = field(default_factory=dict)
    reconstruction: np.ndarray | None = None


# -----------------------------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------------------------


def _closest_unitary(x: np.ndarray) -> np.ndarray:
    """Polar decomposition: return the unitary closest to x in Frobenius norm."""
    u_mat, _, vh_mat = np.linalg.svd(x, full_matrices=False)
    return u_mat @ vh_mat


def _permutation_matrix_from_qubit_order(
    new_order: Sequence[int],
    n: int,
) -> np.ndarray:
    """
    Return the (2**n x 2**n) boolean permutation matrix P that reorders **little‑endian** qubits.

    Little‑endian convention: qubit 0 is the **least significant bit** (LSB).

    - ``new_order[k]`` is the original qubit that becomes bit-position ``k`` in the *new*
      representation (with ``k=0`` the LSB).
    - States: ``|psi_new> = P |psi_old>``
    - Operators: ``U_new = P U_old P^T``  (P is real, so P^T = P.conj().T)

    This is identical to the permutation used in the OSD module.
    """
    if len(new_order) != n or set(new_order) != set(range(n)):
        raise QiskitError("`new_order` must be a permutation of range(n).")
    dim = 2**n
    indices = np.arange(dim, dtype=np.int64)
    bits = (indices[:, None] >> np.arange(n, dtype=np.int64)) & 1
    reordered_bits = bits[:, new_order]
    new_indices = np.sum(reordered_bits << np.arange(n, dtype=np.int64), axis=1)
    return np.eye(dim, dtype=bool)[:, new_indices]


def _pick_bipartitions(
    m: int,
    order: Literal["small_to_big", "big_to_small"],
) -> Iterable[tuple[tuple[int, ...], tuple[int, ...]]]:
    """
    Yield nontrivial bipartitions (s, sc) over local indices 0..m-1, up to symmetry.

    The order of |s| depends on ``order``:
      - "small_to_big": 1, 2, ..., floor(m/2)
      - "big_to_small": floor(m/2), ..., 2, 1
    """
    if m <= 1:
        return
    sizes = range(1, m // 2 + 1) if order == "small_to_big" else range(m // 2, 0, -1)
    local_qubits = tuple(range(m))
    for k_size in sizes:
        for subset in itertools.combinations(local_qubits, k_size):
            sc = tuple(q for q in local_qubits if q not in subset)
            yield tuple(sorted(subset)), tuple(sorted(sc))


def _best_bipartition_all_sizes(
    op_local: np.ndarray,
    qubits_local: tuple[int, ...],
    max_partitions: int | None,
) -> tuple[tuple[int, ...], CutInfo, np.ndarray, np.ndarray]:
    """
    Scan *all* nontrivial bipartitions (up to symmetry) of the current subset, regardless of
    search order, and return the best (minimum Frobenius tail).
    """
    m_size = len(qubits_local)
    best = (None,) * 5
    tested = 0
    for k_size in range(1, m_size // 2 + 1):
        for s_local in itertools.combinations(range(m_size), k_size):
            if max_partitions is not None and tested >= max_partitions:
                break
            tested += 1
            res_osd = operator_schmidt_decomposition(
                op_local, qargs=s_local, k=1, return_reconstruction=False
            )
            svals = res_osd["singular_values"]
            tail = svals[1:]
            fro_err = float(np.sqrt(np.sum(tail**2))) if tail.size else 0.0
            rel_err = float(fro_err / (np.linalg.norm(svals) + 1e-16))
            s_global = tuple(qubits_local[i] for i in s_local)
            sc_global = tuple(q for q in qubits_local if q not in s_global)
            info = CutInfo(
                subset_s=tuple(sorted(s_global)),
                subset_sc=tuple(sorted(sc_global)),
                singular_values=svals,
                frobenius_error=fro_err,
                relative_frobenius_error=rel_err,
                kept_terms=1,
            )
            a_factor = res_osd["A_factors"][0]
            b_factor = res_osd["B_factors"][0]
            if best[0] is None or fro_err < best[0].frobenius_error:
                best = (info, a_factor, b_factor, s_global, sc_global)

    if best[0] is None:
        # m>=2 should always produce at least one candidate; this is a safety fallback.
        raise QiskitError("No bipartition candidates were evaluated.")

    info, a_factor, b_factor, s_global, sc_global = best
    return s_global, info, a_factor, b_factor


# -----------------------------------------------------------------------------
# Core recursion
# -----------------------------------------------------------------------------


def _factorize_recursive(
    op_local: np.ndarray,
    qubits_local: tuple[int, ...],
    *,
    atol: float,
    rtol: float,
    mode: Literal["exact_only", "exact_or_best", "best_only"],
    search: Literal["small_to_big", "big_to_small"],
    max_partitions: int | None,
    cuts_accum: list[CutInfo],
) -> tuple[list[tuple[int, ...]], list[np.ndarray], bool]:
    """
    Recursively decompose `op_local` acting on `qubits_local` into product blocks.

    Returns
    -------
    blocks, factors, is_exact
      - `blocks`: list of global qubit tuples (kept in recursion order)
      - `factors`: list of ndarray factors, aligned with `blocks`
      - `is_exact`: True iff all splits inside this subtree were exact
    """
    m_size = len(qubits_local)
    if m_size == 1:
        return [qubits_local], [op_local], True

    # 1) Try exact rank-1 splits first (via OSD k=1); pass LOCAL indices to OSD.
    for s_local, _ in _pick_bipartitions(m_size, search):
        res_osd = operator_schmidt_decomposition(
            op_local, qargs=s_local, k=1, return_reconstruction=False
        )
        svals = res_osd["singular_values"]
        tail = svals[1:]
        fro_err = float(np.sqrt(np.sum(tail**2))) if tail.size else 0.0
        if fro_err <= atol:
            # Accept exact split. OSD returns A on S, B on Sc in the [Sc, S] permuted basis.
            a_factor = res_osd["A_factors"][0]
            b_factor = res_osd["B_factors"][0]
            unitary_a = _closest_unitary(a_factor)
            unitary_b = _closest_unitary(b_factor)

            # Map LOCAL → GLOBAL qubit indices for the two sides.
            s_global = tuple(qubits_local[i] for i in s_local)
            sc_global = tuple(q for q in qubits_local if q not in s_global)

            # Recurse on each side. IMPORTANT: keep factor/block order consistent with OSD layout:
            # we return blocks = [S, Sc] but factors = [B, A] so that kron(factors[0],...,)
            # (LSB block first) matches the final new_order and requires no extra swaps.
            blocks_a, ops_a, exact_a = _factorize_recursive(
                unitary_a,
                tuple(sorted(s_global)),
                atol=atol,
                rtol=rtol,
                mode=mode,
                search=search,
                max_partitions=max_partitions,
                cuts_accum=cuts_accum,
            )
            blocks_b, ops_b, exact_b = _factorize_recursive(
                unitary_b,
                tuple(sorted(sc_global)),
                atol=atol,
                rtol=rtol,
                mode=mode,
                search=search,
                max_partitions=max_partitions,
                cuts_accum=cuts_accum,
            )
            # Return in the convention: blocks [S, Sc], factors [B, A]
            return blocks_a + blocks_b, ops_b + ops_a, (exact_a and exact_b)

    # 2) No exact split under this subset.
    if mode == "exact_only":
        # Leaf at this granularity is *exact* (we reconstruct this block exactly).
        # Overall exactness will be True iff *all* recursive branches were exact.
        return [qubits_local], [op_local], True

    # 3) Approximate best split: pick the *globally best* bipartition across *all* sizes,
    #    and STOP here (no deeper approximate recursion).
    s_global, info, a_factor, b_factor = _best_bipartition_all_sizes(
        op_local, qubits_local, max_partitions=max_partitions
    )
    cuts_accum.append(info)
    unitary_a = _closest_unitary(a_factor)
    unitary_b = _closest_unitary(b_factor)
    return (
        [tuple(sorted(s_global)), tuple(sorted(set(qubits_local) - set(s_global)))],
        [unitary_b, unitary_a],
        False,
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------


def tensor_product_decomposition(
    op: np.ndarray,
    *,
    atol: float | None = None,
    rtol: float | None = None,
    mode: Literal["exact_only", "exact_or_best", "best_only"] = "exact_or_best",
    search: Literal["small_to_big", "big_to_small"] = "small_to_big",
    max_partitions: int | None = None,
    return_operator: bool = False,
) -> TensorProductDecomposition:
    r"""
    Find the most granular tensor-product *block* decomposition of an operator over disjoint
    qubit sets. If an exact decomposition exists, return it; otherwise (when permitted) return
    the best bipartition in Frobenius norm using the top operator-Schmidt term.

    **Semantics and ordering.**
    We return disjoint blocks ``(S_0, S_1, ..., S_{m-1})`` and corresponding local factors
    ``(U_{S_0}, U_{S_1}, ..., U_{S_{m-1}})`` such that:

    - The concatenation ``new_order = S_0 + S_1 + ... + S_{m-1}`` (LSB block first) defines a
      permutation matrix ``P`` with ``U = P^T U_perm P``.
    - The *permuted* operator factors as a plain Kronecker product in **LSB→MSB order**:
      ``U_perm = kron(U_{S_0}, U_{S_1}, ..., U_{S_{m-1}})``.

    Internally rely on :func:`operator_schmidt_decomposition` which computes the OSD in a
    permuted basis ``[Sc, S]`` with the B-factor (``Sc``) occupying the LSB block and the A-factor
    (``S``) occupying the MSB block. Keep the recursion output ordered so that reconstruction
    needs only a *single* final permutation for ``new_order`` (no extra swaps).

    Parameters
    ----------
    op : np.ndarray
        Square matrix of shape ``(2**n, 2**n)`` in the standard little-endian basis
        (qubit 0 is LSB).
    atol : float or None
        Absolute tolerance used to accept rank-1 (exact) operator-Schmidt splits and in the final
        exactness check for the reconstruction. If ``None``, defaults to
        :data:`qiskit.quantum_info.operators.predicates.ATOL_DEFAULT`.
    rtol : float or None
        Relative tolerance used together with ``atol`` to accept exactness (i.e.,
        a check of the form ``residual <= atol + rtol * ||U||_F``). If ``None``, defaults to
        :data:`qiskit.quantum_info.operators.predicates.RTOL_DEFAULT`.
    mode : {"exact_only", "exact_or_best", "best_only"}
        - "exact_only": return exact blocks only; if a subset has no exact split, it becomes
          a leaf block.
        - "exact_or_best": recursively split *exactly* wherever possible; when a node has no exact
          split, choose the *single best* bipartition across all sizes and STOP at that node
          (no deeper approximate recursion).
        - "best_only": return a single best bipartition without deeper recursion.
    search : {"small_to_big","big_to_small"}
        Controls the order of subset sizes |S| to try at each recursion node for **exact** splits.
        "small_to_big": 1, 2, ..., floor(m/2). "big_to_small": floor(m/2), ..., 2, 1.
    max_partitions : int or None
        Optional cap on the number of bipartitions to evaluate per recursion call (for large n).
    return_operator : bool
        If True, also supply ``operator_factors`` (as :class:`Operator`) and the full
        ``reconstruction`` matrix in the original qubit order.

    Returns
    -------
    TensorProductDecomposition
        See :class:`TensorProductDecomposition`. ``blocks`` and ``factors`` are aligned as
        described above.

    Notes
    -----
    *Uses OSD.* For a bipartition ``S|Sc``, the OSD rank is 1 iff the operator is a simple tensor
    across that cut; we test this via :func:`operator_schmidt_decomposition(op, qargs=S, k=1)` and
    check the Frobenius norm of the tail singular values (``<= atol`` to accept exactness). The
    leading factors are polar-projected to unitaries for stable outputs.

    *Future work.* We may add a heuristic to prioritize promising cuts (e.g., coarse metrics on
    reshapes, sparsity/symmetry hints). This will be an internal optimization with no API changes.
    """
    if not isinstance(op, np.ndarray) or op.ndim != 2 or op.shape[0] != op.shape[1]:
        raise QiskitError("`op` must be a square np.ndarray.")
    n_float = np.log2(op.shape[0])
    if not np.isclose(n_float, int(n_float)):
        raise QiskitError("`op` dimension must be a power of 2.")
    n_qubits = int(round(n_float))
    if n_qubits == 0:
        raise QiskitError(
            "0-qubit (1x1) operators are not supported by tensor_product_decomposition."
        )

    # Default tolerances from predicates
    if atol is None:
        atol = float(ATOL_DEFAULT)
    if rtol is None:
        rtol = float(RTOL_DEFAULT)

    qubits = tuple(range(n_qubits))
    cuts: list[CutInfo] = []

    blocks, local_ops, is_exact_subtree = _factorize_recursive(
        op,
        qubits,
        atol=atol,
        rtol=rtol,
        mode=mode,
        search=search,
        max_partitions=max_partitions,
        cuts_accum=cuts,
    )

    # Keep the recursion ordering. Build u_perm as kron(factors[0],...,factors[m-1]) (LSB→MSB).
    factors_ordered = tuple(local_ops)
    blocks_ordered = tuple(blocks)

    if len(factors_ordered) == 0:
        u_perm = np.eye(op.shape[0], dtype=complex)
    else:
        u_perm = factors_ordered[0]
        for mat in factors_ordered[1:]:
            u_perm = np.kron(u_perm, mat)

    # Build final permutation new_order by concatenating the blocks (LSB block first).
    new_order = tuple(q for blk in blocks_ordered for q in blk)
    perm_matrix = _permutation_matrix_from_qubit_order(new_order, n_qubits)

    reconstruction = perm_matrix.T @ u_perm @ perm_matrix
    diff = op - reconstruction
    residual = float(np.linalg.norm(diff, ord="fro"))
    denom = float(np.linalg.norm(op, ord="fro"))
    relative_residual = (residual / denom) if denom != 0.0 else 0.0

    operator_factors = tuple(Operator(m) for m in factors_ordered) if return_operator else None

    return TensorProductDecomposition(
        blocks=blocks_ordered,
        factors=factors_ordered,
        operator_factors=operator_factors,
        is_exact=bool(is_exact_subtree and residual <= atol + rtol * denom),
        residual=residual,
        relative_residual=relative_residual,
        cuts=cuts,
        permutation={"new_order": new_order, "matrix": perm_matrix},
        reconstruction=reconstruction if return_operator else None,
    )
