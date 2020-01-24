# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.
"""
Quantum information measures, metrics, and related functions for states.
"""

import numpy as np
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.states.utils import _format_state, _funm_svd


def state_fidelity(state1, state2, validate=True):
    r"""Return the state fidelity between two quantum states.

    The state fidelity :math:`F` for density matrix input states
    :math:`\rho_1, \rho_2` is given by

    .. math::
        F(\rho_1, \rho_2) = Tr[\sqrt{\sqrt{\rho_1}\rho_2\sqrt{\rho_1}}]^2.

    If one of the states is a pure state this simplies to
    :math:`F(\rho_1, \rho_2) = \langle\psi_1|\rho_2|\psi_1\rangle`, where
    :math:`\rho_1 = |\psi_1\rangle\!\langle\psi_1|`.

    Args:
        state1 (Statevector or DensityMatrix): the first quantum state.
        state2 (Statevector or DensityMatrix): the second quantum state.
        validate (bool): check if the inputs are valid quantum states
                         [Default: True]

    Returns:
        float: The state fidelity :math:`F(\rho_1, \rho_2)`.

    Raises:
        QiskitError: if ``validate=True`` and the inputs are invalid quantum states.
    """
    # convert input to numpy arrays
    state1 = _format_state(state1, validate=validate)
    state2 = _format_state(state2, validate=validate)

    # Get underlying numpy arrays
    arr1 = state1.data
    arr2 = state2.data
    if isinstance(state1, Statevector):
        if isinstance(state2, Statevector):
            # Fidelity of two Statevectors
            fid = np.abs(arr2.conj().dot(arr1))**2
        else:
            # Fidelity of Statevector(1) and DensityMatrix(2)
            fid = arr1.conj().dot(arr2).dot(arr1)
    elif isinstance(state2, Statevector):
        # Fidelity of Statevector(2) and DensityMatrix(1)
        fid = arr2.conj().dot(arr1).dot(arr2)
    else:
        # Fidelity of two DensityMatrices
        s1sq = _funm_svd(arr1, np.sqrt)
        s2sq = _funm_svd(arr2, np.sqrt)
        fid = np.linalg.norm(s1sq.dot(s2sq), ord='nuc')**2
    # Convert to py float rather than return np.float
    return float(np.real(fid))
