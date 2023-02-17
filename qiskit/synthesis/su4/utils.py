# This code is part of Qiskit.
#
# (C) Copyright IBM 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Utilities for SU(4) synthesis: Weyl chamber, polytopes, distances, etc."""

import numpy as np


_EPS = 1e-12


def find_min_point(P):
    """Find the closest point on convex polytope P to the origin (in Euclidean distance).

    P is given as a list of points forming its vertices.

    Function from
    https://scipy-cookbook.readthedocs.io/items/Finding_Convex_Hull_Minimum_Point.html
    """
    if len(P) == 1:
        return P[0]

    P = [np.array(p) for p in P]

    # Step 0. Choose a point from C(P)
    x = P[np.array([np.dot(p, p) for p in P]).argmin()]

    while True:
        # Step 1. \alpha_k := min{x_{k-1}^T p | p \in P}
        p_alpha = P[np.array([np.dot(x, p) for p in P]).argmin()]

        if np.dot(x, x - p_alpha) < _EPS:
            return np.array([float(i) for i in x])

        Pk = [p for p in P if abs(np.dot(x, p - p_alpha)) < _EPS]

        # Step 2. P_k := { p | p \in P and x_{k-1}^T p = \alpha_k}
        P_Pk = [p for p in P if not np.array([(p == q).all() for q in Pk]).any()]

        if len(Pk) == len(P):
            return np.array([float(i) for i in x])

        y = find_min_point(Pk)

        p_beta = P_Pk[np.array([np.dot(y, p) for p in P_Pk]).argmin()]

        if np.dot(y, y - p_beta) < _EPS:
            return np.array([float(i) for i in y])

        # Step 4.
        P_aux = [p for p in P_Pk if (np.dot(y - x, y - p) > _EPS) and (np.dot(x, y - p) != 0)]
        p_lambda = P_aux[np.array([np.dot(y, y - p) / np.dot(x, y - p) for p in P_aux]).argmin()]
        lam = np.dot(x, p_lambda - y) / np.dot(y - x, y - p_lambda)

        x += lam * (y - x)


def average_infidelity(p, q):
    """Computes the infidelity distance between two points p, q expressed in
    positive canonical coordinates.

    Average gate fidelity is :math:`Fbar = (d + |Tr (Utarget \\cdot U^dag)|^2) / d(d+1)`

    [1] M. Horodecki, P. Horodecki and R. Horodecki, PRA 60, 1888 (1999)
    """

    a0, b0, c0 = p
    a1, b1, c1 = q

    return 1 - 1 / 20 * (
        4
        + 16
        * (
            np.cos(a0 - a1) ** 2 * np.cos(b0 - b1) ** 2 * np.cos(c0 - c1) ** 2
            + np.sin(a0 - a1) ** 2 * np.sin(b0 - b1) ** 2 * np.sin(c0 - c1) ** 2
        )
    )
