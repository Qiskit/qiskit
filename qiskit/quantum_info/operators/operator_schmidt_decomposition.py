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
Operator Schmidt decomposition utilities.
"""
from __future__ import annotations

from typing import Any
from collections.abc import Sequence
import numpy as np

from qiskit.exceptions import QiskitError


def _permutation_matrix_from_qubit_order(new_order: Sequence[int], n: int) -> np.ndarray:
    """
    Return the ``(2**n) x (2**n)`` permutation matrix ``P`` that reorders **little‑endian** qubits.

    Little‑endian convention: qubit 0 is the **least significant bit** (LSB).

    Mapping (bits → indices):
      * ``new_order[k]`` gives which **original** qubit becomes bit‑position ``k`` in the **new**
        representation (with ``k=0`` the LSB).
      * For a computational basis state with original bitstring
        ``b = (b_{n-1} ... b_1 b_0)`` we form the new bitstring ``b'`` by
        ``b'_k = b_{ new_order[k] }``.
      * Index mapping: ``i' = sum_k b'_k 2^k``.

    Action:
      * States: ``|psi_new> = P |psi_old>``.
      * Operators: ``U_new = P U_old P^T`` (``P`` is real; ``P^T = P.conj().T``).

    Args:
        new_order: A permutation of ``range(n)`` where entry ``k`` is the original qubit index that
            becomes bit‑position ``k`` in the new ordering (LSB is ``k=0``).
        n: Number of qubits.

    Returns:
        P: A boolean permutation matrix of shape ``(2**n, 2**n)`` such that the above actions hold.

    Raises:
        QiskitError: If ``new_order`` is not a permutation of ``range(n)`` or sizes mismatch.

    Example:
        For ``n=3`` and ``new_order = [2, 0, 1]`` (LSB first):
          - New LSB (k=0) is original qubit 2,
          - New middle bit (k=1) is original qubit 0,
          - New MSB (k=2) is original qubit 1.
        If original state has bits ``(b2 b1 b0)``, the new index corresponds to bits ``(b1 b0 b2)``
        in MSB→LSB order.
    """
    # Validate
    if not isinstance(n, int) or n < 0:
        raise QiskitError("`n` must be a non‑negative integer.")
    if len(new_order) != n:
        raise QiskitError(f"`new_order` must have length n={n}.")
    if set(new_order) != set(range(n)):
        raise QiskitError("`new_order` must be a permutation of range(n).")

    dim = 2**n
    indices = np.arange(dim, dtype=np.int64)  # original indices i

    # Extract original bits b_q (q=0 is LSB) for each index.
    bits = (indices[:, None] >> np.arange(n, dtype=np.int64)) & 1  # shape (dim, n)

    # Reorder bits so that new bit‑position k gets original bit from new_order[k].
    reordered_bits = bits[:, new_order]  # shape (dim, n)

    # Convert reordered bits to new indices i'
    new_indices = np.sum(reordered_bits << np.arange(n, dtype=np.int64), axis=1)

    # Build permutation matrix with columns permuted by new_indices
    return np.eye(dim, dtype=bool)[:, new_indices]


def _check_inputs(
    op: np.ndarray, qargs: Sequence[int]
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    if not isinstance(op, np.ndarray):
        raise QiskitError("`op` must be a numpy.ndarray.")
    if op.ndim != 2 or op.shape[0] != op.shape[1]:
        raise QiskitError("`op` must be a square matrix.")
    n_float = np.log2(op.shape[0])
    if not np.isclose(n_float, int(n_float)):
        raise QiskitError("`op` dimension must be a power of 2.")
    n = int(round(n_float))
    if op.shape != (2**n, 2**n):
        raise QiskitError(f"`op` must have shape {(2**n, 2**n)}.")

    subset_a = tuple(sorted({int(q) for q in qargs}))
    if any(q < 0 or q >= n for q in subset_a):
        raise QiskitError(f"All indices in `qargs` must be in [0, {n-1}].")
    subset_b = tuple(sorted(set(range(n)) - set(subset_a)))
    if not subset_a or not subset_b:
        raise QiskitError("`qargs` must be a strict, non‑empty subset of the qubit indices.")
    return n, subset_a, subset_b


def _realign_row_major(u_perm: np.ndarray, dim_a: int, dim_b: int) -> np.ndarray:
    """Return realignment ``realigned`` of ``u_perm`` for bipartition ``A (MSB) ⊗ B (LSB)``.

    ``realigned[(iA, jA), (iB, jB)] = u_perm[(iA, iB), (jA, jB)]`` via reshape+transpose.
    """
    u4 = u_perm.reshape(dim_a, dim_b, dim_a, dim_b)  # (iA, iB, jA, jB)
    realigned = np.transpose(u4, (0, 2, 1, 3))  # (iA, jA, iB, jB)
    return realigned.reshape(dim_a * dim_a, dim_b * dim_b)


def operator_schmidt_decomposition(
    op: np.ndarray,
    qargs: Sequence[int],
    *,
    k: int | None = None,
    return_reconstruction: bool = False,
) -> dict[str, Any]:
    r"""
    Compute the operator Schmidt decomposition of ``op`` across the bipartition
    defined by ``qargs`` (subsystem :math:`A`) and its complement (subsystem :math:`B`).

    Given an operator :math:`U` acting on :math:`n` qubits, and a bipartition
    :math:`\mathcal{H} = \mathcal{H}_A \otimes \mathcal{H}_B` with
    :math:`\dim(\mathcal{H}_A) = 2^{|A|}`, :math:`\dim(\mathcal{H}_B) = 2^{|B|}`,
    the operator Schmidt decomposition is

    .. math::

        U \;=\; \sum_{r=1}^{R} s_r \, A_r \otimes B_r,

    where :math:`s_r \ge 0` are the singular values of the **realigned** matrix,
    and :math:`A_r, B_r` are matrices on :math:`\mathcal{H}_A` and
    :math:`\mathcal{H}_B`, respectively.

    **Basis and permutation.**
    The decomposition is computed in a **permuted basis** where the qubit order is
    ``[Sc, S]`` (complement first, then selected subset). In this basis we have

    .. math::

        U_{\text{perm}} \;=\; \sum_{r} A_r \otimes B_r,

    with :math:`A_r` acting on subsystem :math:`A` (MSB block) and :math:`B_r` on
    subsystem :math:`B` (LSB block). The original operator satisfies

    .. math::

        U \;=\; P^\top\, U_{\text{perm}}\, P,

    where ``P`` is the permutation matrix mapping the original qubit order to ``[Sc, S]``.

    **Truncation (top-``k`` terms).**
    If ``k`` is provided, the returned factors correspond to the best rank-``k`` approximation
    (in Frobenius norm) of the realigned matrix; i.e., only the top-``k`` singular components
    are used to construct the factors and (optionally) the reconstruction. The array
    ``singular_values`` in the return value always contains the **full** spectrum so that you
    can inspect or post-process the tail; metadata about truncation and the Frobenius error of
    the discarded part are also returned.

    Args:
        op: Complex matrix of shape ``(2**n, 2**n)`` (unitary or not).
        qargs: Qubit indices belonging to subsystem :math:`A`. Little‑endian ordering is used
            in Qiskit (qubit 0 is the least significant bit).
        k: If not ``None``, keep only the top-``k`` Schmidt terms. Must be a positive integer.
            If ``k`` exceeds the number of available singular values, it is clipped.
        return_reconstruction: If ``True``, also return the reconstruction
            (sum of kept terms) mapped back to the **original** qubit order.

    Returns:
        A dictionary with:
          * ``partition``: ``{"S": tuple, "Sc": tuple}`` with the chosen split.
          * ``permutation``: dict with:
              - ``new_order``: tuple of qubit indices in the permuted order ``[Sc, S]``.
              - ``matrix``: the permutation matrix ``P`` (shape ``(2**n, 2**n)``, real).
          * ``singular_values``: 1D ``np.ndarray`` of **all** singular values (descending).
          * ``A_factors``: list of ``np.ndarray`` of shape ``(2**|S|, 2**|S|)`` for the **kept**
            terms, in the permuted basis (A on MSB block).
          * ``B_factors``: list of ``np.ndarray`` of shape ``(2**|Sc|, 2**|Sc|)`` for the **kept**
            terms, in the permuted basis (B on LSB block).
          * ``truncation``: dict with:
              - ``kept_terms``: number of terms kept (``k`` after clipping; equals full rank if
                ``k`` is ``None``).
              - ``discarded_terms``: number of discarded terms.
              - ``frobenius_error``: Frobenius norm of the discarded tail (equal for the
                realigned matrix and the permuted operator).
              - ``relative_frobenius_error``: ``frobenius_error / np.linalg.norm(singular_values)``.
          * ``reconstruction``: optional ``np.ndarray`` of the (possibly truncated) reconstruction
            in **original** qubit order (present only when ``return_reconstruction=True``).

    Raises:
        QiskitError: If inputs are malformed (non‑power‑of‑two dimensions, invalid ``qargs``)
            or ``k`` is not a positive integer when provided.
    """
    n, subset_a, subset_b = _check_inputs(op, qargs)

    # Permute to [Sc, S] so B occupies LSB block, A occupies MSB block.
    perm = _permutation_matrix_from_qubit_order(list(subset_b) + list(subset_a), n)
    u_perm = perm @ op @ perm.T

    dim_a, dim_b = 2 ** len(subset_a), 2 ** len(subset_b)

    # Realign and SVD
    realigned = _realign_row_major(u_perm, dim_a, dim_b)
    u_left, sing_vals, vh = np.linalg.svd(realigned, full_matrices=False)
    vcols = vh.conj().T

    # Determine number of terms to keep
    total_terms = len(sing_vals)  # = min(dim_a*dim_a, dim_b*dim_b)
    if k is None:
        num = total_terms
    else:
        if not (isinstance(k, int) and k > 0):
            raise QiskitError("`k` must be a positive integer if provided.")
        num = min(k, total_terms)

    # Build factors so that sum kron(A_r, B_r) == u_perm (permuted basis), truncated if needed.
    a_factors: list[np.ndarray] = []
    b_factors: list[np.ndarray] = []
    for i in range(num):
        vec_a = u_left[:, i] * np.sqrt(sing_vals[i])
        vec_b = np.conj(vcols[:, i]) * np.sqrt(sing_vals[i])
        a_factors.append(vec_a.reshape(dim_a, dim_a))
        b_factors.append(vec_b.reshape(dim_b, dim_b))

    # Truncation metadata
    tail = sing_vals[num:]
    fro_err = float(np.sqrt(np.sum(tail**2))) if tail.size else 0.0
    denom = np.linalg.norm(sing_vals)
    rel_err = float(fro_err / denom) if denom > 0 else 0.0

    out: dict[str, Any] = {
        "partition": {"S": subset_a, "Sc": subset_b},
        "permutation": {
            "new_order": tuple(subset_b) + tuple(subset_a),
            "matrix": perm,
        },
        "singular_values": sing_vals.copy(),  # full spectrum (not truncated)
        "A_factors": a_factors,  # truncated list (num terms)
        "B_factors": b_factors,  # truncated list (num terms)
        "truncation": {
            "kept_terms": num,
            "discarded_terms": total_terms - num,
            "frobenius_error": fro_err,
            "relative_frobenius_error": rel_err,
        },
    }

    if return_reconstruction:
        u_rec = np.zeros_like(op, dtype=np.complex128)
        for i in range(num):
            u_rec += np.kron(a_factors[i], b_factors[i])
        # Map back to original qubit order
        out["reconstruction"] = perm.T @ u_rec @ perm

    return out
