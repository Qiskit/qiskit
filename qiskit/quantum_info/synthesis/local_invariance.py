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

"""Routines that use local invariants to compute properties
of two-qubit unitary operators.
"""
from math import sqrt, sin, cos
import numpy as np

INVARIANT_TOL = 1e-12

# Bell "Magic" basis
MAGIC = 1.0/sqrt(2)*np.array([
    [1, 0, 0, 1j],
    [0, 1j, -1, 0],
    [0, 1j, 1, 0],
    [1, 0, 0, -1j]], dtype=complex)


def two_qubit_local_invariants(U):
    """Computes the local invarants for a two qubit unitary.

    Args:
        U (ndarray): Input two qubit unitary.

    Returns:
        ndarray: NumPy array of local invariants [g0, g1, g2].

    Raises:
        ValueError: Input not a 2q unitary.

    Notes:
        Y. Makhlin, Quant. Info. Proc. 1, 243-252 (2002).
        Zhang et al., Phys Rev A. 67, 042313 (2003).
    """
    U = np.asarray(U)
    if U.shape != (4, 4):
        raise ValueError('Unitary must correspond to a two qubit gate.')

    # Transform to bell basis
    Um = MAGIC.conj().T.dot(U.dot(MAGIC))
    # Get determinate since +- one is allowed.
    det_um = np.linalg.det(Um)
    M = Um.T.dot(Um)
    # trace(M)**2
    m_tr2 = M.trace()
    m_tr2 *= m_tr2

    # Table II of Ref. 1 or Eq. 28 of Ref. 2.
    G1 = m_tr2/(16*det_um)
    G2 = (m_tr2 - np.trace(M.dot(M)))/(4*det_um)

    # Here we split the real and imag pieces of G1 into two so as
    # to better equate to the Weyl chamber coordinates (c0,c1,c2)
    # and explore the parameter space.
    # Also do a FP trick -0.0 + 0.0 = 0.0
    return np.round([G1.real, G1.imag, G2.real], 12) + 0.0


def cx_equivalence(U):
    """Returns the number of cx gates in the
    local equivilent set of U: 0, 1, 2, or 3.

    Args:
         U (ndarray): Input two qubit unitary.

    Returns:
        int: Number of cx gates in local equivalence set.

    Notes:
        The trivial zero and one cx cases are in the
        literature.  The space of two cx invariant
        sets seems not to have been explored.
    """
    vec = two_qubit_local_invariants(U)

    # We default to 3 here since KAK decomposition
    # tells us that any 2q gate can be represented
    # by at most 3 CNOT gates.
    cx_equiv = 3

    # Locally equiv to a identity
    if all(np.abs(vec - np.array([1, 0, 3])) < INVARIANT_TOL):
        cx_equiv = 0
    # Locally equiv to a cx gate
    elif all(np.abs(vec - np.array([0, 0, 1])) < INVARIANT_TOL):
        cx_equiv = 1
    # Locally equiv to some 2q circuit with 2 cx
    # This is simply all points with g0 >= 0 and g1 == 0
    # since we already checked the I and cx points above.
    elif vec[0] >= 0:
        # g1 should be zero
        if abs(vec[1]) < INVARIANT_TOL:
            cx_equiv = 2
    return cx_equiv


def maximally_entangling(U):
    """Computes whether a two qubit unitary is a maximally
    entangling gate; It can generate a Bell state with a single
    application.

    Args:
        U (ndarray): Input two qubit unitary.

    Returns:
        bool: True if gate is maximally entangling.

    Notes:
        From Y. Makhlin, Quant. Info. Proc. 1, 243-252 (2002).
    """
    vec = two_qubit_local_invariants(U)
    angle = np.arctan2(vec[1], vec[0])
    mod = sqrt(vec[0]**2 + vec[1]**2)
    max_entangling = False
    if sin(angle)**2 <= 4*mod:
        if cos(angle)*(cos(angle)-vec[2]) >= 0:
            max_entangling = True
    return max_entangling
