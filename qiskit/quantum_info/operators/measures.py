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

# pylint: disable=invalid-name
"""
A collection of useful quantum information functions for operators.
"""

import warnings
import numpy as np

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.pauli import Pauli
from qiskit.quantum_info.operators.channel import SuperOp


def process_fidelity(channel,
                     target=None,
                     require_cp=True,
                     require_tp=False,
                     require_cptp=False):
    r"""Return the process fidelity of a noisy quantum channel.

    This process fidelity :math:`F_{\text{pro}}` is given by

    .. math::
        F_{\text{pro}}(\mathcal{E}, U)
            = \frac{Tr[S_U^\dagger S_{\mathcal{E}}]}{d^2}

    where :math:`S_{\mathcal{E}}, S_{U}` are the
    :class:`~qiskit.quantum_info.SuperOp` matrices for the input quantum
    *channel* :math:`\cal{E}` and *target* unitary :math:`U` respectively,
    and :math:`d` is the dimension of the *channel*.

    Args:
        channel (QuantumChannel): noisy quantum channel.
        target (Operator or None): target unitary operator.
            If `None` target is the identity operator [Default: None].
        require_cp (bool): require channel to be completely-positive
            [Default: True].
        require_tp (bool): require channel to be trace-preserving
            [Default: False].
        require_cptp (bool): (DEPRECATED) require input channels to be
            CPTP [Default: False].

    Returns:
        float: The process fidelity :math:`F_{\text{pro}}`.

    Raises:
        QiskitError: if the channel and target do not have the same
        dimensions, or have different input and output dimensions, or are
        not completely-positive (with ``require_cp=True``) or not
        trace-preserving (with ``require_tp=True``).
    """
    # Format inputs
    if isinstance(channel, (list, np.ndarray, Operator, Pauli)):
        channel = Operator(channel)
    else:
        channel = SuperOp(channel)
    input_dim, output_dim = channel.dim
    if input_dim != output_dim:
        raise QiskitError(
            'Quantum channel must have equal input and output dimensions.')

    if target is not None:
        # Multiple channel by adjoint of target
        target = Operator(target)
        if (input_dim, output_dim) != target.dim:
            raise QiskitError(
                'Quantum channel and target must have the same dimensions.')
        channel = channel @ target.adjoint()

    # Validate complete-positivity and trace-preserving
    if require_cptp:
        # require_cptp kwarg is DEPRECATED
        # Remove in future qiskit version
        warnings.warn(
            "Please use `require_cp=True, require_tp=True` "
            "instead of `require_cptp=True`.", DeprecationWarning)
        require_cp = True
        require_tp = True
    if isinstance(channel, Operator) and (require_cp or require_tp):
        is_unitary = channel.is_unitary()
        # Validate as unitary
        if require_cp and not is_unitary:
            raise QiskitError('channel is not completely-positive')
        if require_tp and not is_unitary:
            raise QiskitError('channel is not trace-preserving')
    else:
        # Validate as QuantumChannel
        if require_cp and not channel.is_cp():
            raise QiskitError('channel is not completely-positive')
        if require_tp and not channel.is_tp():
            raise QiskitError('channel is not trace-preserving')

    # Compute process fidelity with identity channel
    if isinstance(channel, Operator):
        # |Tr[U]/dim| ** 2
        fid = np.abs(np.trace(channel.data) / input_dim)**2
    else:
        # Tr[S] / (dim ** 2)
        fid = np.trace(channel.data) / (input_dim**2)
    return float(np.real(fid))


def average_gate_fidelity(channel,
                          target=None,
                          require_cp=True,
                          require_tp=False):
    r"""Return the average gate fidelity of a noisy quantum channel.

    The average gate fidelity :math:`F_{\text{ave}}` is given by

    .. math::
        F_{\text{ave}}(\mathcal{E}, U)
            &= \int d\psi \langle\psi|U^\dagger
                \mathcal{E}(|\psi\rangle\!\langle\psi|)U|\psi\rangle \\
            &= \frac{d F_{\text{pro}}(\mathcal{E}, U) + 1}{d + 1}

    where :math:`F_{\text{pro}}(\mathcal{E}, U)` is the
    :meth:`~qiskit.quantum_info.process_fidelity` of the input quantum
    *channel* :math:`\mathcal{E}` with a *target* unitary :math:`U`, and
    :math:`d` is the dimension of the *channel*.

    Args:
        channel (QuantumChannel): noisy quantum channel.
        target (Operator or None): target unitary operator.
            If `None` target is the identity operator [Default: None].
        require_cp (bool): require channel to be completely-positive
            [Default: True].
        require_tp (bool): require channel to be trace-preserving
            [Default: False].

    Returns:
        float: The average gate fidelity :math:`F_{\text{ave}}`.

    Raises:
        QiskitError: if the channel and target do not have the same
        dimensions, or have different input and output dimensions, or are
        not completely-positive (with ``require_cp=True``) or not
        trace-preserving (with ``require_tp=True``).
    """
    if isinstance(channel, (list, np.ndarray, Operator, Pauli)):
        channel = Operator(channel)
    else:
        channel = SuperOp(channel)
    dim, _ = channel.dim
    f_pro = process_fidelity(channel,
                             target=target,
                             require_cp=require_cp,
                             require_tp=require_tp)
    return (dim * f_pro + 1) / (dim + 1)


def gate_error(channel, target=None, require_cp=True, require_tp=False):
    r"""Return the gate error of a noisy quantum channel.

    The gate error :math:`E` is given by the average gate infidelity

    .. math::
        E(\mathcal{E}, U) = 1 - F_{\text{ave}}(\mathcal{E}, U)

    where :math:`F_{\text{ave}}(\mathcal{E}, U)` is the
    :meth:`~qiskit.quantum_info.average_gate_fidelity` of the input
    quantum *channel* :math:`\mathcal{E}` with a *target* unitary
    :math:`U`.

    Args:
        channel (QuantumChannel): noisy quantum channel.
        target (Operator or None): target unitary operator.
            If `None` target is the identity operator [Default: None].
        require_cp (bool): require channel to be completely-positive
            [Default: True].
        require_tp (bool): require channel to be trace-preserving
            [Default: False].

    Returns:
        float: The average gate error :math:`E`.

    Raises:
        QiskitError: if the channel and target do not have the same
        dimensions, or have different input and output dimensions, or are
        not completely-positive (with ``require_cp=True``) or not
        trace-preserving (with ``require_tp=True``).
    """
    return 1 - average_gate_fidelity(
        channel, target=target, require_cp=require_cp, require_tp=require_tp)
