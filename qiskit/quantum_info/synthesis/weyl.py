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

Y = np.array([[0, -1],
              [1, 0]], dtype=complex)

YY = np.kron(Y, Y)


def weyl_coordinates(U):
    """Computes the Weyl coordinates for
    a given two qubit unitary matrix.

    Args:
        U (ndarray): Input two qubit unitary.

    Returns:
        ndarray: Array of Weyl coordinates.

    Raises:
        ValueError: Computed coordinates not in Weyl chamber.

    Notes:
        Based entirely on A. M. Childs et al.,
        Phys. Rev. A 68, 052311 (2003).

    """
    U = np.asarray(U)
    if U.shape != (4, 4):
        raise ValueError('Unitary must correspond to a two qubit gate.')

    # Eq 13
    U_tilde = YY.dot(U.T.dot(YY))
    det_u = la.det(U)

    U2 = U.dot(U_tilde)
    evals = la.eigvals(U2/np.sqrt(det_u))

    # Eq 39
    two_S = np.angle(evals) / np.pi

    # Eq 40
    for kk in range(4):
        if two_S[kk] <= -0.5:
            two_S[kk] += 2.0

    # Sort like Eq 37
    # Sort increasing then flip to decreasing with slice
    S = np.sort(two_S/2.0)[::-1]

    # Eq. 42
    N = int(round(sum(S)))

    # Select values and use to get weyl coordinates
    # per Eqs 9-12
    S -= np.array([1]*N+[0]*(4-N))
    S = np.roll(S, -N)
    M = np.array([[1, 1, 0],
                  [1, 0, 1],
                  [0, 1, 1]])

    weyl = np.dot(M, S[:3])

    if weyl[2] < 0:
        weyl[0] = 1 - weyl[0]
        weyl[2] *= -1

    # Verify that they are in Weyl chamber
    test1 = (weyl[0] < 0.5 and weyl[1] <= weyl[0]) and (weyl[2] <= weyl[1])
    test2 = (weyl[0] >= 0.5 and weyl[1] <= 1 - weyl[0]) and weyl[2] <= weyl[1]
    if not (test1 or test2):
        raise ValueError('Point ({}, {}, {}) not in Weyl chamber.'
                         .format(np.pi*weyl[0], np.pi*weyl[1], np.pi*weyl[2]))

    return (np.round(weyl, 12) + 0.0)*np.pi


def local_equivalence(weyl):
    """Computes the equivalent local invariants from the
    Weyl coordinates.

    Args:
        weyl (ndarray): Weyl coordinates.

    Returns:
        ndarray: Local equivalent coordinates [g0, g1, g3].

    Notes:
        This uses Eq. 30 from Zhang et al, PRA 67, 042313 (2003).
    """
    g0_equiv = np.prod(np.cos(weyl)**2)-np.prod(np.sin(weyl)**2)
    g1_equiv = np.prod(np.sin(2*weyl))/4
    g2_equiv = 4*np.prod(np.cos(weyl)**2)-4*np.prod(np.sin(weyl)**2)-np.prod(np.cos(2*weyl))
    return np.array([g0_equiv, g1_equiv, g2_equiv])
