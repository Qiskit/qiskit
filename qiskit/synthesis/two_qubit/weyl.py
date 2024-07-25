# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
# pylint: disable=invalid-name

"""Routines that compute  and use the Weyl chamber coordinates.
"""

from __future__ import annotations
import numpy as np

# "Magic" basis used for the Weyl decomposition. The basis and its adjoint are stored individually
# unnormalized, but such that their matrix multiplication is still the identity.  This is because
# they are only used in unitary transformations (so it's safe to do so), and `sqrt(0.5)` is not
# exactly representable in floating point.  Doing it this way means that every element of the matrix
# is stored exactly correctly, and the multiplication is _exactly_ the identity rather than
# differing by 1ULP.
_B_nonnormalized = np.array([[1, 1j, 0, 0], [0, 0, 1j, 1], [0, 0, 1j, -1], [1, -1j, 0, 0]])
_B_nonnormalized_dagger = 0.5 * _B_nonnormalized.conj().T


def transform_to_magic_basis(U: np.ndarray, reverse: bool = False) -> np.ndarray:
    """Transform the 4-by-4 matrix ``U`` into the magic basis.

    This method internally uses non-normalized versions of the basis to minimize the floating-point
    errors that arise during the transformation.

    Args:
        U (np.ndarray): 4-by-4 matrix to transform.
        reverse (bool): Whether to do the transformation forwards (``B @ U @ B.conj().T``, the
        default) or backwards (``B.conj().T @ U @ B``).

    Returns:
        np.ndarray: The transformed 4-by-4 matrix.
    """
    if reverse:
        return _B_nonnormalized_dagger @ U @ _B_nonnormalized
    return _B_nonnormalized @ U @ _B_nonnormalized_dagger


def weyl_coordinates(U: np.ndarray) -> np.ndarray:
    """Computes the Weyl coordinates for a given two-qubit unitary matrix.

    Args:
        U (np.ndarray): Input two-qubit unitary.

    Returns:
        np.ndarray: Array of the 3 Weyl coordinates.
    """
    import scipy.linalg as la

    pi2 = np.pi / 2
    pi4 = np.pi / 4

    U = U / la.det(U) ** (0.25)
    Up = transform_to_magic_basis(U, reverse=True)
    # We only need the eigenvalues of `M2 = Up.T @ Up` here, not the full diagonalization.
    D = la.eigvals(Up.T @ Up)
    d = -np.angle(D) / 2
    d[3] = -d[0] - d[1] - d[2]
    cs = np.mod((d[:3] + d[3]) / 2, 2 * np.pi)

    # Reorder the eigenvalues to get in the Weyl chamber
    cstemp = np.mod(cs, pi2)
    np.minimum(cstemp, pi2 - cstemp, cstemp)
    order = np.argsort(cstemp)[[1, 2, 0]]
    cs = cs[order]
    d[:3] = d[order]

    # Flip into Weyl chamber
    if cs[0] > pi2:
        cs[0] -= 3 * pi2
    if cs[1] > pi2:
        cs[1] -= 3 * pi2
    conjs = 0
    if cs[0] > pi4:
        cs[0] = pi2 - cs[0]
        conjs += 1
    if cs[1] > pi4:
        cs[1] = pi2 - cs[1]
        conjs += 1
    if cs[2] > pi2:
        cs[2] -= 3 * pi2
    if conjs == 1:
        cs[2] = pi2 - cs[2]
    if cs[2] > pi4:
        cs[2] -= pi2

    return cs[[1, 0, 2]]
