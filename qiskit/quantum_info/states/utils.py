# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Quantum information utility functions for states.
"""

import numpy as np
import scipy.linalg as la

from qiskit.exceptions import QiskitError
from qiskit.quantum_info.states.statevector import Statevector
from qiskit.quantum_info.states.densitymatrix import DensityMatrix
from qiskit.quantum_info.operators.channel import SuperOp


def partial_trace(state, qargs):
    """Return reduced density matrix by tracing out part of quantum state.

    If all subsystems are traced over this returns the
    :meth:`~qiskit.quantum_info.DensityMatrix.trace` of the
    input state.

    Args:
        state (Statevector or DensityMatrix): the input state.
        qargs (list): The subsystems to trace over.

    Returns:
        DensityMatrix: The reduced density matrix.

    Raises:
        QiskitError: if input state is invalid.
    """
    state = _format_state(state, validate=False)

    # If we are tracing over all subsystems we return the trace
    if sorted(qargs) == list(range(len(state.dims()))):
        # Should this raise an exception instead?
        # Or return a 1x1 density matrix?
        return state.trace()

    # Statevector case
    if isinstance(state, Statevector):
        trace_systems = len(state._dims) - 1 - np.array(qargs)
        new_dims = tuple(np.delete(np.array(state._dims), qargs))
        new_dim = np.product(new_dims)
        arr = state._data.reshape(state._shape)
        rho = np.tensordot(arr, arr.conj(),
                           axes=(trace_systems, trace_systems))
        rho = np.reshape(rho, (new_dim, new_dim))
        return DensityMatrix(rho, dims=new_dims)

    # Density matrix case
    # Trace first subsystem to avoid coping whole density matrix
    dims = state.dims(qargs)
    tr_op = SuperOp(np.eye(dims[0]).reshape(1, dims[0] ** 2),
                    input_dims=[dims[0]], output_dims=[1])
    ret = state.evolve(tr_op, [qargs[0]])
    # Trace over remaining subsystems
    for qarg, dim in zip(qargs[1:], dims[1:]):
        tr_op = SuperOp(np.eye(dim).reshape(1, dim ** 2),
                        input_dims=[dim], output_dims=[1])
        ret = ret.evolve(tr_op, [qarg])
    # Remove traced over subsystems which are listed as dimension 1
    ret._reshape(tuple(np.delete(np.array(ret._dims), qargs)))
    return ret


def shannon_entropy(pvec, base=2):
    r"""Compute the Shannon entropy of a probability vector.

    The shannon entropy of a probability vector
    :math:`\vec{p} = [p_0, ..., p_{n-1}]` is defined as

    .. math:

        H(\vec{p}) = \sum_{i=0}^{n-1} p_i \log_b(p_i)

    where :math:`b` is the log base and (default 2), and
    :math:`0 \log_b(0) \equiv 0`.

    Args:
        pvec (array_like): a probability vector.
        base (int): the base of the logarithm [Default: 2].

    Returns:
        float: The Shannon entropy H(pvec).
    """
    if base == 2:
        def logfn(x):
            return - x * np.log2(x)
    elif base == np.e:
        def logfn(x):
            return - x * np.log(x)
    else:
        def logfn(x):
            return -x * np.log(x) / np.log(base)

    h_val = 0.
    for x in pvec:
        if 0 < x < 1:
            h_val += logfn(x)
    return h_val


def _format_state(state, validate=True):
    """Format input state into class object"""
    if isinstance(state, list):
        state = np.array(state, dtype=complex)
    if isinstance(state, np.ndarray):
        ndim = state.ndim
        if ndim == 1:
            state = Statevector(state)
        elif ndim == 2:
            dim1, dim2 = state.shape
            if dim2 == 1:
                state = Statevector(state)
            elif dim1 == dim2:
                state = DensityMatrix(state)
    if not isinstance(state, (Statevector, DensityMatrix)):
        raise QiskitError("Input is not a quantum state")
    if validate and not state.is_valid():
        raise QiskitError("Input quantum state is not a valid")
    return state


def _funm_svd(matrix, func):
    """Apply real scalar function to singular values of a matrix.

    Args:
        matrix (array_like): (N, N) Matrix at which to evaluate the function.
        func (callable): Callable object that evaluates a scalar function f.

    Returns:
        ndarray: funm (N, N) Value of the matrix function specified by func
        evaluated at `A`.
    """
    unitary1, singular_values, unitary2 = la.svd(matrix, lapack_driver='gesvd')
    diag_func_singular = np.diag(func(singular_values))
    return unitary1.dot(diag_func_singular).dot(unitary2)
