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
from math import sqrt
import numpy as np

INVARIANT_TOL = 1e-12

# Bell "Magic" basis
MAGIC = 1.0/sqrt(2)*np.array([
    [1, 0, 0, 1j],
    [0, 1j, 1, 0],
    [0, 1j, -1, 0],
    [1, 0, 0, -1j]], dtype=complex)


def two_qubit_local_invariants(U):
    """Computes the local invariants for a two-qubit unitary.

    Args:
        U (ndarray): Input two-qubit unitary.

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
        raise ValueError('Unitary must correspond to a two-qubit gate.')

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


def local_equivalence(weyl):
    """Computes the equivalent local invariants from the
    Weyl coordinates.

    Args:
        weyl (ndarray): Weyl coordinates.

    Returns:
        ndarray: Local equivalent coordinates [g0, g1, g3].

    Notes:
        This uses Eq. 30 from Zhang et al, PRA 67, 042313 (2003),
        but we multiply weyl coordinates by 2 since we are
        working in the reduced chamber.
    """
    g0_equiv = np.prod(np.cos(2*weyl)**2)-np.prod(np.sin(2*weyl)**2)
    g1_equiv = np.prod(np.sin(4*weyl))/4
    g2_equiv = 4*np.prod(np.cos(2*weyl)**2)-4*np.prod(np.sin(2*weyl)**2)-np.prod(np.cos(4*weyl))
    return np.round([g0_equiv, g1_equiv, g2_equiv], 12) + 0.0
