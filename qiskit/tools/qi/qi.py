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

# pylint: disable=invalid-name,unsupported-assignment-operation
"""
A collection of DEPRECATED quantum information functions.

Please see the `qiskit.quantum_info` module for replacements.
"""

import warnings
import math
import numpy as np
import scipy.linalg as la

from qiskit.quantum_info import pauli_group

###############################################################
# circuit manipulation.
###############################################################


# Define methods for making QFT circuits
def qft(circ, q, n):
    """n-qubit QFT on q in circ."""
    warnings.warn(
        'This function is deprecated and will be removed in a future release.',
        DeprecationWarning)
    for j in range(n):
        for k in range(j):
            circ.cu1(math.pi / float(2**(j - k)), q[j], q[k])
        circ.h(q[j])


###############################################################
# State manipulation.
###############################################################


def partial_trace(state, trace_systems, dimensions=None, reverse=True):
    """
    Partial trace over subsystems of multi-partite matrix.

    Note that subsystems are ordered as rho012 = rho0(x)rho1(x)rho2.

    Args:
        state (matrix_like): a matrix NxN
        trace_systems (list(int)): a list of subsystems (starting from 0) to
                                  trace over.
        dimensions (list(int)): a list of the dimensions of the subsystems.
                                If this is not set it will assume all
                                subsystems are qubits.
        reverse (bool): ordering of systems in operator.
            If True system-0 is the right most system in tensor product.
            If False system-0 is the left most system in tensor product.

    Returns:
        matrix_like: A density matrix with the appropriate subsystems traced
            over.
    Raises:
        Exception: if input is not a multi-qubit state.
    """
    # DEPRECATED
    warnings.warn(
        'This function is deprecated and will be removed in the future.'
        ' It is superseded by the `quantum_info.partial_trace` function',
        DeprecationWarning)

    if dimensions is None:  # compute dims if not specified
        num_qubits = int(np.log2(len(state)))
        dimensions = [2 for _ in range(num_qubits)]
        if len(state) != 2**num_qubits:
            raise Exception("Input is not a multi-qubit state, "
                            "specify input state dims")
    else:
        dimensions = list(dimensions)

    if isinstance(trace_systems, int):
        trace_systems = [trace_systems]
    else:  # reverse sort trace sys
        trace_systems = sorted(trace_systems, reverse=True)

    # trace out subsystems
    if state.ndim == 1:
        # optimized partial trace for input state vector
        return __partial_trace_vec(state, trace_systems, dimensions, reverse)
    # standard partial trace for input density matrix
    return __partial_trace_mat(state, trace_systems, dimensions, reverse)


def __partial_trace_vec(vec, trace_systems, dimensions, reverse=True):
    """
    Partial trace over subsystems of multi-partite vector.

    Args:
        vec (vector_like): complex vector N
        trace_systems (list(int)): a list of subsystems (starting from 0) to
                                  trace over.
        dimensions (list(int)): a list of the dimensions of the subsystems.
                                If this is not set it will assume all
                                subsystems are qubits.
        reverse (bool): ordering of systems in operator.
            If True system-0 is the right most system in tensor product.
            If False system-0 is the left most system in tensor product.

    Returns:
        ndarray: A density matrix with the appropriate subsystems traced over.
    """

    # trace sys positions
    if reverse:
        dimensions = dimensions[::-1]
        trace_systems = len(dimensions) - 1 - np.array(trace_systems)

    rho = vec.reshape(dimensions)
    rho = np.tensordot(rho, rho.conj(), axes=(trace_systems, trace_systems))
    d = int(np.sqrt(np.product(rho.shape)))

    return rho.reshape(d, d)


def __partial_trace_mat(mat, trace_systems, dimensions, reverse=True):
    """
    Partial trace over subsystems of multi-partite matrix.

    Note that subsystems are ordered as rho012 = rho0(x)rho1(x)rho2.

    Args:
        mat (matrix_like): a matrix NxN.
        trace_systems (list(int)): a list of subsystems (starting from 0) to
                                  trace over.
        dimensions (list(int)): a list of the dimensions of the subsystems.
                                If this is not set it will assume all
                                subsystems are qubits.
        reverse (bool): ordering of systems in operator.
            If True system-0 is the right most system in tensor product.
            If False system-0 is the left most system in tensor product.

    Returns:
        ndarray: A density matrix with the appropriate subsystems traced over.
    """

    trace_systems = sorted(trace_systems, reverse=True)
    for j in trace_systems:
        # Partition subsystem dimensions
        dimension_trace = int(dimensions[j])  # traced out system
        if reverse:
            left_dimensions = dimensions[j + 1:]
            right_dimensions = dimensions[:j]
            dimensions = right_dimensions + left_dimensions
        else:
            left_dimensions = dimensions[:j]
            right_dimensions = dimensions[j + 1:]
            dimensions = left_dimensions + right_dimensions
        # Contract remaining dimensions
        dimension_left = int(np.prod(left_dimensions))
        dimension_right = int(np.prod(right_dimensions))

        # Reshape input array into tri-partite system with system to be
        # traced as the middle index
        mat = mat.reshape([
            dimension_left, dimension_trace, dimension_right, dimension_left,
            dimension_trace, dimension_right
        ])
        # trace out the middle system and reshape back to a matrix
        mat = mat.trace(axis1=1,
                        axis2=4).reshape(dimension_left * dimension_right,
                                         dimension_left * dimension_right)
    return mat


def vectorize(density_matrix, method='col'):
    """Flatten an operator to a vector in a specified basis.

    Args:
        density_matrix (ndarray): a density matrix.
        method (str): the method of vectorization. Allowed values are
            - 'col' (default) flattens to column-major vector.
            - 'row' flattens to row-major vector.
            - 'pauli' flattens in the n-qubit Pauli basis.
            - 'pauli-weights': flattens in the n-qubit Pauli basis ordered by
               weight.

    Returns:
        ndarray: the resulting vector.
    Raises:
        Exception: if input state is not a n-qubit state
    """
    warnings.warn(
        'This function is deprecated and will be removed in a future release.',
        DeprecationWarning)
    density_matrix = np.array(density_matrix)
    if method == 'col':
        return density_matrix.flatten(order='F')
    elif method == 'row':
        return density_matrix.flatten(order='C')
    elif method in ['pauli', 'pauli_weights']:
        num = int(np.log2(len(density_matrix)))  # number of qubits
        if len(density_matrix) != 2**num:
            raise Exception('Input state must be n-qubit state')
        if method == 'pauli_weights':
            pgroup = pauli_group(num, case='weight')
        else:
            pgroup = pauli_group(num, case='tensor')
        vals = [
            np.trace(np.dot(p.to_matrix(), density_matrix)) for p in pgroup
        ]
        return np.array(vals)
    return None


def devectorize(vectorized_mat, method='col'):
    """Devectorize a vectorized square matrix.

    Args:
        vectorized_mat (ndarray): a vectorized density matrix.
        method (str): the method of devectorization. Allowed values are
            - 'col' (default): flattens to column-major vector.
            - 'row': flattens to row-major vector.
            - 'pauli': flattens in the n-qubit Pauli basis.
            - 'pauli-weights': flattens in the n-qubit Pauli basis ordered by
               weight.

    Returns:
        ndarray: the resulting matrix.
    Raises:
        Exception: if input state is not a n-qubit state
    """
    warnings.warn(
        'This function is deprecated and will be removed in a future release.',
        DeprecationWarning)
    vectorized_mat = np.array(vectorized_mat)
    dimension = int(np.sqrt(vectorized_mat.size))
    if len(vectorized_mat) != dimension * dimension:
        raise Exception('Input is not a vectorized square matrix')

    if method == 'col':
        return vectorized_mat.reshape(dimension, dimension, order='F')
    elif method == 'row':
        return vectorized_mat.reshape(dimension, dimension, order='C')
    elif method in ['pauli', 'pauli_weights']:
        num_qubits = int(np.log2(dimension))  # number of qubits
        if dimension != 2**num_qubits:
            raise Exception('Input state must be n-qubit state')
        if method == 'pauli_weights':
            pgroup = pauli_group(num_qubits, case='weight')
        else:
            pgroup = pauli_group(num_qubits, case='tensor')
        pbasis = np.array([p.to_matrix() for p in pgroup]) / 2**num_qubits
        return np.tensordot(vectorized_mat, pbasis, axes=1)
    return None


def choi_to_pauli(choi, order=1):
    """
    Convert a Choi-matrix to a Pauli-basis superoperator.

    Note that this function assumes that the Choi-matrix
    is defined in the standard column-stacking convention
    and is normalized to have trace 1. For a channel E this
    is defined as: choi = (I \\otimes E)(bell_state).

    The resulting 'rauli' R acts on input states as
    |rho_out>_p = R.|rho_in>_p
    where |rho> = vectorize(rho, method='pauli') for order=1
    and |rho> = vectorize(rho, method='pauli_weights') for order=0.

    Args:
        choi (matrix): the input Choi-matrix.
        order (int): ordering of the Pauli group vector.
            order=1 (default) is standard lexicographic ordering.
                Eg: [II, IX, IY, IZ, XI, XX, XY,...]
            order=0 is ordered by weights.
                Eg. [II, IX, IY, IZ, XI, XY, XZ, XX, XY,...]

    Returns:
        np.array: A superoperator in the Pauli basis.
    """
    warnings.warn(
        'This function is deprecated and will be removed in a future release.'
        ' Use the `quantum_info.Choi` and `quantum_info.PTM` quantum channel'
        ' classes for similar functionality', DeprecationWarning)
    if order == 0:
        order = 'weight'
    elif order == 1:
        order = 'tensor'

    # get number of qubits'
    num_qubits = int(np.log2(np.sqrt(len(choi))))
    pgp = pauli_group(num_qubits, case=order)
    rauli = []
    for i in pgp:
        for j in pgp:
            pauliop = np.kron(j.to_matrix().T, i.to_matrix())
            rauli += [np.trace(np.dot(choi, pauliop))]
    return np.array(rauli).reshape(4**num_qubits, 4**num_qubits)


def chop(array, epsilon=1e-10):
    """
    Truncate small values of a complex array.

    Args:
        array (array_like): array to truncate small values.
        epsilon (float): threshold.

    Returns:
        np.array: A new operator with small values set to zero.
    """
    warnings.warn(
        'This function is deprecated and will be removed in a future release.'
        ' Use `numpy.round` for similar functionality.', DeprecationWarning)
    ret = np.array(array)

    if np.isrealobj(ret):
        ret[abs(ret) < epsilon] = 0.0
    else:
        ret.real[abs(ret.real) < epsilon] = 0.0
        ret.imag[abs(ret.imag) < epsilon] = 0.0
    return ret


def outer(vector1, vector2=None):
    """
    Construct the outer product of two vectors.

    The second vector argument is optional, if absent the projector
    of the first vector will be returned.

    Args:
        vector1 (ndarray): the first vector.
        vector2 (ndarray): the (optional) second vector.

    Returns:
        np.array: The matrix |v1><v2|.

    """
    warnings.warn(
        'This function is deprecated and will be removed in a future release.'
        ' Please use `numpy.outer` function for similar functionality.',
        DeprecationWarning)
    if vector2 is None:
        vector2 = np.array(vector1).conj()
    else:
        vector2 = np.array(vector2).conj()
    return np.outer(vector1, vector2)


###############################################################
# Measures.
###############################################################


def concurrence(state):
    """Calculate the concurrence.

    Args:
        state (np.array): a quantum state (1x4 array) or a density matrix (4x4
                          array)
    Returns:
        float: concurrence.
    Raises:
        Exception: if attempted on more than two qubits.
    """
    warnings.warn(
        'This function is deprecated and will be removed in a future release.'
        ' It is superseded by the `quantum_info.concurrence` function.',
        DeprecationWarning)
    rho = np.array(state)
    if rho.ndim == 1:
        rho = outer(state)
    if len(state) != 4:
        raise Exception("Concurrence is only defined for more than two qubits")

    YY = np.fliplr(np.diag([-1, 1, 1, -1]))
    A = rho.dot(YY).dot(rho.conj()).dot(YY)
    w = np.sort(np.real(la.eigvals(A)))
    w = np.sqrt(np.maximum(w, 0.))
    return max(0.0, w[-1] - np.sum(w[0:-1]))


def shannon_entropy(pvec, base=2):
    """
    Compute the Shannon entropy of a probability vector.

    The shannon entropy of a probability vector pv is defined as
    $H(pv) = - \\sum_j pv[j] log_b (pv[j])$ where $0 log_b 0 = 0$.

    Args:
        pvec (array_like): a probability vector.
        base (int): the base of the logarithm

    Returns:
        float: The Shannon entropy H(pvec).
    """
    # DEPRECATED
    warnings.warn(
        'This function is deprecated and will be removed in the future.'
        ' It is superseded by the `quantum_info.shannon_entropy` function',
        DeprecationWarning)
    if base == 2:

        def logfn(x):
            return -x * np.log2(x)
    elif base == np.e:

        def logfn(x):
            return -x * np.log(x)
    else:

        def logfn(x):
            return -x * np.log(x) / np.log(base)

    h = 0.
    for x in pvec:
        if 0 < x < 1:
            h += logfn(x)
    return h


def entropy(state):
    """
    Compute the von-Neumann entropy of a quantum state.

    Args:
        state (array_like): a density matrix or state vector.

    Returns:
        float: The von-Neumann entropy S(rho).
    """
    # DEPRECATED
    warnings.warn(
        'This function is deprecated and will be removed in the future.'
        ' It is superseded by the `quantum_info.entropy` function.',
        DeprecationWarning)
    # pylint: disable=assignment-from-no-return
    rho = np.array(state)
    if rho.ndim == 1:
        return 0
    evals = np.maximum(np.linalg.eigvalsh(state), 0.)
    return shannon_entropy(evals, base=np.e)


def mutual_information(state, d0, d1=None):
    """
    Compute the mutual information of a bipartite state.

    Args:
        state (array_like): a bipartite state-vector or density-matrix.
        d0 (int): dimension of the first subsystem.
        d1 (int or None): dimension of the second subsystem.

    Returns:
        float: The mutual information S(rho_A) + S(rho_B) - S(rho_AB).
    """
    # DEPRECATED
    warnings.warn(
        'This function is deprecated and will be removed in the future.'
        ' It is superseded by the `quantum_info.mutual_information` function',
        DeprecationWarning)
    if d1 is None:
        d1 = int(len(state) / d0)
    mi = entropy(partial_trace(state, [0], dimensions=[d0, d1]))
    mi += entropy(partial_trace(state, [1], dimensions=[d0, d1]))
    mi -= entropy(state)
    return mi


def entanglement_of_formation(state, d0, d1=None):
    """
    Compute the entanglement of formation of quantum state.

    The input quantum state must be either a bipartite state vector, or a
    2-qubit density matrix.

    Args:
        state (array_like): (N) array_like or (4,4) array_like, a
            bipartite quantum state.
        d0 (int): the dimension of the first subsystem.
        d1 (int or None): the dimension of the second subsystem.

    Returns:
        float: The entanglement of formation.
    """
    # DEPRECATED
    warnings.warn(
        'This function is deprecated and will be removed in the future.'
        ' It is superseded by the `quantum_info.entanglement_of_formation` function.',
        DeprecationWarning)
    state = np.array(state)

    if d1 is None:
        d1 = int(len(state) / d0)

    if state.ndim == 2 and len(state) == 4 and d0 == 2 and d1 == 2:
        return __eof_qubit(state)
    elif state.ndim == 1:
        # trace out largest dimension
        if d0 < d1:
            tr = [1]
        else:
            tr = [0]
        state = partial_trace(state, tr, dimensions=[d0, d1])
        return entropy(state)
    else:
        print('Input must be a state-vector or 2-qubit density matrix.')
    return None


def __eof_qubit(rho):
    """
    Compute the Entanglement of Formation of a 2-qubit density matrix.

    Args:
        rho ((array_like): (4,4) array_like, input density matrix.

    Returns:
        float: The entanglement of formation.
    """
    c = concurrence(rho)
    c = 0.5 + 0.5 * np.sqrt(1 - c * c)
    return shannon_entropy([c, 1 - c])


###############################################################
# Other.
###############################################################


def is_pos_def(x):
    """Return is_pos_def."""
    warnings.warn(
        'This function is deprecated and will be removed in the future.'
        ' It is superseded by the '
        '`quantum_info.operators.predicates.is_positive_semidefinite_matrix`'
        ' function.', DeprecationWarning)
    return np.all(np.linalg.eigvals(x) > 0)
