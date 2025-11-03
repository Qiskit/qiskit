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

from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from qiskit.exceptions import QiskitError


def _permutation_matrix_from_qubit_order(new_order: Sequence[int], n: int) -> np.ndarray:
    """Return the ``2**n x 2**n`` permutation matrix that reorders little‑endian qubits.

    Mapping:
      * ``new_order[k]`` = which **original** qubit becomes bit-position ``k`` (``k=0`` is LSB).
      * State: ``|psi_new> = P |psi_old>``
      * Operator: ``U_new = P U_old P^T`` (``P`` is real, so ``P^T = P.conj().T``).
    """
    dim = 2**n
    indices = np.arange(dim, dtype=np.int64)
    bits = (indices[:, None] >> np.arange(n, dtype=np.int64)) & 1
    reordered_bits = bits[:, new_order]
    new_indices = np.sum(reordered_bits << np.arange(n, dtype=np.int64), axis=1)
    return np.eye(dim, dtype=np.int8)[:, new_indices]


def _check_inputs(op: np.ndarray, qargs: Sequence[int]) -> Tuple[int, Tuple[int, ...], Tuple[int, ...]]:
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
    S = tuple(sorted(set(int(q) for q in qargs)))
    if any(q < 0 or q >= n for q in S):
        raise QiskitError(f"All indices in `qargs` must be in [0, {n-1}].")
    Sc = tuple(sorted(set(range(n)) - set(S)))
    if not S or not Sc:
        raise QiskitError("`qargs` must be a strict, non‑empty subset of the qubit indices.")
    return n, S, Sc


def _realign_row_major(Up: np.ndarray, dA: int, dB: int) -> np.ndarray:
    """Return realignment ``R`` of ``Up`` for bipartition ``A (MSB) ⊗ B (LSB)``.

    ``R[(iA, jA), (iB, jB)] = Up[(iA, iB), (jA, jB)]`` via reshape+transpose.
    """
    U4 = Up.reshape(dA, dB, dA, dB)  # (iA, iB, jA, jB)
    R = np.transpose(U4, (0, 2, 1, 3))  # (iA, jA, iB, jB)
    return R.reshape(dA * dA, dB * dB)


def operator_schmidt_decomposition(
    op: np.ndarray,
    qargs: Sequence[int],
    *,
    return_reconstruction: bool = False,
) -> Dict[str, Any]:
    r"""Compute the operator Schmidt decomposition of ``op`` across the bipartition
    defined by ``qargs`` (subsystem :math:`A`) and its complement (subsystem :math:`B`).

    Given an operator :math:`U` acting on :math:`n` qubits, and a bipartition
    :math:`\mathcal{H} = \mathcal{H}_A \otimes \mathcal{H}_B` with
    :math:`\dim(\mathcal{H}_A) = 2^{|A|}`, :math:`\dim(\mathcal{H}_B) = 2^{|B|}`,
    the operator Schmidt decomposition is

    .. math::

        U \;=\; \sum_{r=1}^{R} s_r \, A_r \otimes B_r,

    where :math:`s_r \ge 0` are the singular values of the realigned matrix,
    and :math:`A_r, B_r` are matrices on :math:`\mathcal{H}_A` and :math:`\mathcal{H}_B`, respectively.

    Args:
        op: Complex matrix of shape ``(2**n, 2**n)`` (unitary or not).
        qargs: Qubit indices belonging to subsystem :math:`A`. Little‑endian ordering is used
            in Qiskit (qubit 0 is the least significant bit).
        return_reconstruction: If ``True``, also return the full reconstruction mapped
            back to the **original** qubit order.

    Returns:
        A dictionary with:
        * ``partition``: ``{"S": tuple, "Sc": tuple}`` with the chosen split.
        * ``singular_values``: 1D ``np.ndarray`` of singular values in descending order.
        * ``A_factors``: list of ``np.ndarray`` of shape ``(2**|S|, 2**|S|)``.
        * ``B_factors``: list of ``np.ndarray`` of shape ``(2**|Sc|, 2**|Sc|)``.
        * ``reconstruction``: optional ``np.ndarray`` of the full reconstruction.

    Raises:
        QiskitError: If inputs are malformed (non‑power‑of‑two dimensions, invalid ``qargs``).
    """
    n, S, Sc = _check_inputs(op, qargs)

    # Permute to [Sc, S] so B occupies LSB block, A occupies MSB block.
    P = _permutation_matrix_from_qubit_order(list(Sc) + list(S), n)
    Up = P @ op @ P.T

    dA, dB = 2 ** len(S), 2 ** len(Sc)

    # Realign and SVD
    R = _realign_row_major(Up, dA, dB)
    Uu, s, Vh = np.linalg.svd(R, full_matrices=False)
    Vcols = Vh.conj().T

    # Build factors so that sum kron(A_r, B_r) == Up (permuted basis)
    num = len(s)
    A_list: List[np.ndarray] = []
    B_list: List[np.ndarray] = []
    for i in range(num):
        vecA = Uu[:, i] * np.sqrt(s[i])
        vecB = np.conj(Vcols[:, i]) * np.sqrt(s[i]) 
        A_list.append(vecA.reshape(dA, dA))
        B_list.append(vecB.reshape(dB, dB))

    out: Dict[str, Any] = {
        "partition": {"S": S, "Sc": Sc},
        "singular_values": s.copy(),
        "A_factors": A_list,
        "B_factors": B_list,
    }

    if return_reconstruction:
        U_rec = np.zeros_like(op, dtype=np.complex128)
        for i in range(num):
            U_rec += np.kron(A_list[i], B_list[i])
        # Map back to original qubit order
        out["reconstruction"] = P.T @ U_rec @ P

    return out