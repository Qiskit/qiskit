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
This module contains utility functions for circuits.
"""

import numpy
from qiskit.exceptions import QiskitError


def _compute_control_matrix(base_mat, num_ctrl_qubits, ctrl_state=None):
    r"""
    Compute the controlled version of the input matrix with qiskit ordering.
    This function computes the controlled unitary with :math:`n` control qubits
    and :math:`m` target qubits,

    .. math::

        V_n^j(U_{2^m}) = (U_{2^m} \otimes |j\rangle\!\langle j|) +
                         (I_{2^m} \otimes (I_{2^n} - |j\rangle\!\langle j|)).

    where :math:`|j\rangle \in \mathcal{H}^{2^n}` is the control state.

    Args:
        base_mat (ndarray): unitary to be controlled
        num_ctrl_qubits (int): number of controls for new unitary
        ctrl_state (int or str or None): The control state in decimal or as
            a bitstring (e.g. '111'). If None, use 2**num_ctrl_qubits-1.

    Returns:
        ndarray: controlled version of base matrix.

    Raises:
        QiskitError: unrecognized mode or invalid ctrl_state
    """
    num_target = int(numpy.log2(base_mat.shape[0]))
    ctrl_dim = 2**num_ctrl_qubits
    ctrl_grnd = numpy.repeat([[1], [0]], [1, ctrl_dim-1])
    if ctrl_state is None:
        ctrl_state = ctrl_dim - 1
    elif isinstance(ctrl_state, str):
        ctrl_state = int(ctrl_state, 2)
    if isinstance(ctrl_state, int):
        if not 0 <= ctrl_state < ctrl_dim:
            raise QiskitError('Invalid control state value specified.')
    else:
        raise QiskitError('Invalid control state type specified.')
    full_mat_dim = ctrl_dim * base_mat.shape[0]
    full_mat = numpy.zeros((full_mat_dim, full_mat_dim), dtype=base_mat.dtype)
    ctrl_proj = numpy.diag(numpy.roll(ctrl_grnd, ctrl_state))
    full_mat = (numpy.kron(numpy.eye(2**num_target),
                           numpy.eye(ctrl_dim) - ctrl_proj)
                + numpy.kron(base_mat, ctrl_proj))
    return full_mat
