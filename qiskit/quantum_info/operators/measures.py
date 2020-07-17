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
from scipy import sparse

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.pauli import Pauli
from qiskit.quantum_info.operators.channel import SuperOp, Choi

try:
    import cvxpy
    _HAS_CVX = True
except ImportError:
    _HAS_CVX = False


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
        QiskitError: if the channel and target do not have the same dimensions,
                     or have different input and output dimensions.
        QiskitError: if the channel and target or are not completely-positive
                     (with ``require_cp=True``) or not trace-preserving
                     (with ``require_tp=True``).
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
        QiskitError: if the channel and target do not have the same dimensions,
                     or have different input and output dimensions.
        QiskitError: if the channel and target or are not completely-positive
                     (with ``require_cp=True``) or not trace-preserving
                     (with ``require_tp=True``).
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
        QiskitError: if the channel and target do not have the same dimensions,
                     or have different input and output dimensions.
        QiskitError: if the channel and target or are not completely-positive
                     (with ``require_cp=True``) or not trace-preserving
                     (with ``require_tp=True``).
    """
    return 1 - average_gate_fidelity(
        channel, target=target, require_cp=require_cp, require_tp=require_tp)


def diamond_norm(choi, **kwargs):
    r"""Return the diamond norm of the input quantum channel object.

    This function computes the completely-bounded trace-norm (often
    referred to as the diamond-norm) of the input quantum channel object
    using the semidefinite-program from reference [1].

    Args:
        choi(Choi or QuantumChannel): a quantum channel object or
                                      Choi-matrix array.
        kwargs: optional arguments to pass to CVXPY solver.

    Returns:
        float: The completely-bounded trace norm
               :math:`\|\mathcal{E}\|_{\diamond}`.

    Raises:
        QiskitError: if CVXPY package cannot be found.

    Additional Information:
        The input to this function is typically *not* a CPTP quantum
        channel, but rather the *difference* between two quantum channels
        :math:`\|\Delta\mathcal{E}\|_\diamond` where
        :math:`\Delta\mathcal{E} = \mathcal{E}_1 - \mathcal{E}_2`.

    Reference:
        J. Watrous. "Simpler semidefinite programs for completely bounded
        norms", arXiv:1207.5726 [quant-ph] (2012).

    .. note::

        This function requires the optional CVXPY package to be installed.
        Any additional kwargs will be passed to the ``cvxpy.solve``
        function. See the CVXPY documentation for information on available
        SDP solvers.
    """
    _cvxpy_check('`diamond_norm`')  # Check CVXPY is installed

    if not isinstance(choi, Choi):
        choi = Choi(choi)

    def cvx_bmat(mat_r, mat_i):
        """Block matrix for embedding complex matrix in reals"""
        return cvxpy.bmat([[mat_r, -mat_i], [mat_i, mat_r]])

    # Dimension of input and output spaces
    dim_in = choi._input_dim
    dim_out = choi._output_dim
    size = dim_in * dim_out

    # SDP Variables to convert to real valued problem
    r0_r = cvxpy.Variable((dim_in, dim_in))
    r0_i = cvxpy.Variable((dim_in, dim_in))
    r0 = cvx_bmat(r0_r, r0_i)

    r1_r = cvxpy.Variable((dim_in, dim_in))
    r1_i = cvxpy.Variable((dim_in, dim_in))
    r1 = cvx_bmat(r1_r, r1_i)

    x_r = cvxpy.Variable((size, size))
    x_i = cvxpy.Variable((size, size))
    iden = sparse.eye(dim_out)

    # Watrous uses row-vec convention for his Choi matrix while we use
    # col-vec. It turns out row-vec convention is requried for CVXPY too
    # since the cvxpy.kron function must have a constant as its first argument.
    c_r = cvxpy.bmat([[cvxpy.kron(iden, r0_r), x_r], [x_r.T, cvxpy.kron(iden, r1_r)]])
    c_i = cvxpy.bmat([[cvxpy.kron(iden, r0_i), x_i], [-x_i.T, cvxpy.kron(iden, r1_i)]])
    c = cvx_bmat(c_r, c_i)

    # Transpose out Choi-matrix to row-vec convention and vectorize.
    choi_vec = np.transpose(
        np.reshape(choi.data, (dim_in, dim_out, dim_in, dim_out)),
        (1, 0, 3, 2)).ravel(order='F')
    choi_vec_r = choi_vec.real
    choi_vec_i = choi_vec.imag

    # Constraints
    cons = [
        r0 >> 0, r0_r == r0_r.T, r0_i == - r0_i.T, cvxpy.trace(r0_r) == 1,
        r1 >> 0, r1_r == r1_r.T, r1_i == - r1_i.T, cvxpy.trace(r1_r) == 1,
        c >> 0
    ]

    # Objective function
    obj = cvxpy.Maximize(choi_vec_r * cvxpy.vec(x_r) - choi_vec_i * cvxpy.vec(x_i))
    prob = cvxpy.Problem(obj, cons)
    sol = prob.solve(**kwargs)
    return sol


def _cvxpy_check(name):
    """Check that a supported CVXPY version is installed"""
    # Check if CVXPY package is installed
    if not _HAS_CVX:
        raise QiskitError(
            'CVXPY backage is requried for {}. Install'
            ' with `pip install cvxpy` to use.'.format(name))
    # Check CVXPY version
    version = cvxpy.__version__
    if version[0] != '1':
        raise ImportError(
            'Incompatible CVXPY version {} found.'
            ' Install version >=1.0.'.format(version))
