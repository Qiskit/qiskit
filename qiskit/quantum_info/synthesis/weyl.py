# -*- coding: utf-8 -*-

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

import numpy as np
import scipy.linalg as la
from qiskit.exceptions import QiskitError

_B = (1.0/np.sqrt(2)) * np.array([[1, 1j, 0, 0],
                                  [0, 0, 1j, 1],
                                  [0, 0, 1j, -1],
                                  [1, -1j, 0, 0]], dtype=complex)
_Bd = _B.T.conj()


def weyl_coordinates(U):
    """Computes the Weyl coordinates for
    a given two-qubit unitary matrix.

    Args:
        U (ndarray): Input two-qubit unitary.

    Returns:
        ndarray: Array of Weyl coordinates.

    Raises:
        QiskitError: Computed coordinates not in Weyl chamber.
    """
    pi2 = np.pi/2
    pi4 = np.pi/4

    U = U / la.det(U)**(0.25)
    Up = _Bd.dot(U).dot(_B)
    M2 = Up.T.dot(Up)

    # M2 is a symmetric complex matrix. We need to decompose it as M2 = P D P^T where
    # P ∈ SO(4), D is diagonal with unit-magnitude elements.
    # D, P = la.eig(M2)  # this can fail for certain kinds of degeneracy
    for _ in range(3):  # FIXME: this randomized algorithm is horrendous
        M2real = np.random.randn()*M2.real + np.random.randn()*M2.imag
        _, P = la.eigh(M2real)
        D = P.T.dot(M2).dot(P).diagonal()
        if np.allclose(P.dot(np.diag(D)).dot(P.T), M2, rtol=1.0e-13, atol=1.0e-13):
            break
    else:
        raise QiskitError("TwoQubitWeylDecomposition: failed to diagonalize M2")

    d = -np.angle(D)/2
    d[3] = -d[0]-d[1]-d[2]
    cs = np.mod((d[:3]+d[3])/2, 2*np.pi)

    # Reorder the eigenvalues to get in the Weyl chamber
    cstemp = np.mod(cs, pi2)
    np.minimum(cstemp, pi2-cstemp, cstemp)
    order = np.argsort(cstemp)[[1, 2, 0]]
    cs = cs[order]
    d[:3] = d[order]

    # Flip into Weyl chamber
    if cs[0] > pi2:
        cs[0] -= 3*pi2
    if cs[1] > pi2:
        cs[1] -= 3*pi2
    conjs = 0
    if cs[0] > pi4:
        cs[0] = pi2-cs[0]
        conjs += 1
    if cs[1] > pi4:
        cs[1] = pi2-cs[1]
        conjs += 1
    if cs[2] > pi2:
        cs[2] -= 3*pi2
    if conjs == 1:
        cs[2] = pi2-cs[2]
    if cs[2] > pi4:
        cs[2] -= pi2

    return cs[[1, 0, 2]]
