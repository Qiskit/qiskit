# -*- coding: utf-8 -*-

# Copyright 2019, IBM.
#
# This source code is licensed under the Apache License, Version 2.0 found in
# the LICENSE.txt file in the root directory of this source tree.

import logging

import numpy as np
import scipy.linalg as la

from qiskit.qiskiterror import QiskitError
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from qiskit.quantum_info.operators.predicates import ATOL_DEFAULT

logger = logging.getLogger(__name__)


def _to_choi(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the Choi representation."""
    if rep == 'Choi':
        return data
    if rep == 'UnitaryChannel':
        return _from_unitary('Choi', data, input_dim, output_dim)
    if rep == 'SuperOp':
        return _superop_to_choi(data, input_dim, output_dim)
    if rep == 'Kraus':
        return _kraus_to_choi(data, input_dim, output_dim)
    if rep == 'Chi':
        return _chi_to_choi(data, input_dim, output_dim)
    if rep == 'PTM':
        data = _ptm_to_superop(data, input_dim, output_dim)
        return _superop_to_choi(data, input_dim, output_dim)
    if rep == 'Stinespring':
        data = _stinespring_to_kraus(data, input_dim, output_dim)
        return _kraus_to_choi(data, input_dim, output_dim)
    raise QiskitError('Invalid QuantumChannel {}'.format(rep))


def _to_superop(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the SuperOp representation."""
    if rep == 'SuperOp':
        return data
    if rep == 'UnitaryChannel':
        return _from_unitary('SuperOp', data, input_dim, output_dim)
    if rep == 'Choi':
        return _choi_to_superop(data, input_dim, output_dim)
    if rep == 'Kraus':
        return _kraus_to_superop(data, input_dim, output_dim)
    if rep == 'Chi':
        data = _chi_to_choi(data, input_dim, output_dim)
        return _choi_to_superop(data, input_dim, output_dim)
    if rep == 'PTM':
        return _ptm_to_superop(data, input_dim, output_dim)
    if rep == 'Stinespring':
        data = _stinespring_to_kraus(data, input_dim, output_dim)
        return _kraus_to_superop(data, input_dim, output_dim)
    raise QiskitError('Invalid QuantumChannel {}'.format(rep))


def _to_kraus(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the Kraus representation."""
    if rep == 'Kraus':
        return data
    if rep == 'Stinespring':
        return _stinespring_to_kraus(data, input_dim, output_dim)
    if rep == 'UnitaryChannel':
        return _from_unitary('Kraus', data, input_dim, output_dim)
    # Convert via Choi and Kraus
    if rep != 'Choi':
        data = _to_choi(rep, data, input_dim, output_dim)
    return _choi_to_kraus(data, input_dim, output_dim)


def _to_chi(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the Chi representation."""
    if rep == 'Chi':
        return data
    # Check valid n-qubit input
    _check_nqubit_dim(input_dim, output_dim)
    if rep == 'UnitaryChannel':
        return _from_unitary('Chi', data, input_dim, output_dim)
    # Convert via Choi representation
    if rep != 'Choi':
        data = _to_choi(rep, data, input_dim, output_dim)
    return _choi_to_chi(data, input_dim, output_dim)


def _to_ptm(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the PTM representation."""
    if rep == 'PTM':
        return data
    # Check valid n-qubit input
    _check_nqubit_dim(input_dim, output_dim)
    if rep == 'UnitaryChannel':
        return _from_unitary('PTM', data, input_dim, output_dim)
    # Convert via Superoperator representation
    if rep != 'SuperOp':
        data = _to_superop(rep, data, input_dim, output_dim)
    return _superop_to_ptm(data, input_dim, output_dim)


def _to_stinespring(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the Stinespring representation."""
    if rep == 'Stinespring':
        return data
    if rep == 'UnitaryChannel':
        return _from_unitary('Stinespring', data, input_dim, output_dim)
    # Convert via Superoperator representation
    if rep != 'Kraus':
        data = _to_kraus(rep, data, input_dim, output_dim)
    return _kraus_to_stinespring(data, input_dim, output_dim)


def _to_unitary(rep, data, input_dim, output_dim):
    """Transform a QuantumChannel to the UnitaryChannel representation."""
    if rep == 'UnitaryChannel':
        return data
    # Convert via Kraus representation
    if rep != 'Kraus':
        data = _to_kraus(rep, data, input_dim, output_dim)
    return _kraus_to_unitary(data, input_dim, output_dim)


def _from_unitary(rep, data, input_dim, output_dim):
    """Transform UnitaryChannel representation to other representation."""
    if rep == 'UnitaryChannel':
        return data
    if rep == 'SuperOp':
        return np.kron(np.conj(data), data)
    if rep == 'Choi':
        vec = np.ravel(data, order='F')
        return np.outer(vec, np.conj(vec))
    if rep == 'Kraus':
        return ([data], None)
    if rep == 'Stinespring':
        return (data, None)
    if rep == 'Chi':
        _check_nqubit_dim(input_dim, output_dim)
        data = _from_unitary('Choi', data, input_dim, output_dim)
        return _choi_to_chi(data, input_dim, output_dim)
    if rep == 'PTM':
        _check_nqubit_dim(input_dim, output_dim)
        data = _from_unitary('SuperOp', data, input_dim, output_dim)
        return _superop_to_ptm(data, input_dim, output_dim)
    raise QiskitError('Invalid QuantumChannel {}'.format(rep))


def _kraus_to_unitary(data, input_dim, output_dim):
    """Transform Kraus representation to UnitaryChannel representation."""
    kraus_l, kraus_r = data
    if input_dim != output_dim or kraus_r is not None or len(kraus_l) > 1:
        logger.error('Channel cannot be converted to UnitaryChannel representation')
        raise QiskitError('Channel cannot be converted to UnitaryChannel representation')
    return kraus_l[0]


def _superop_to_choi(data, input_dim, output_dim):
    """Transform SuperOp representation to Choi representation."""
    shape = (output_dim, output_dim, input_dim, input_dim)
    return _reshuffle(data, shape)


def _choi_to_superop(data, input_dim, output_dim):
    """Transform Choi to SuperOp representation."""
    shape = (input_dim, output_dim, input_dim, output_dim)
    return _reshuffle(data, shape)


def _kraus_to_choi(data, input_dim, output_dim):
    """Transform Kraus representation to Choi representation."""
    choi = 0
    kraus_l, kraus_r = data
    if kraus_r is None:
        for a in kraus_l:
            u = a.ravel(order='F')
            choi += np.dot(u[:, None], u.conj()[None, :])
    else:
        for a, b in zip(kraus_l, kraus_r):
            choi += np.dot(a.ravel(order='F')[:, None],
                           b.ravel(order='F').conj()[None, :])
    return choi


def _choi_to_kraus(data, input_dim, output_dim, atol=ATOL_DEFAULT):
    """Transform Choi representation to Kraus representation."""
    # Check if hermitian matrix
    if is_hermitian_matrix(data, atol=atol):
        # Get eigen-decomposition of Choi-matrix
        w, v = la.eigh(data)
        # Check eigenvaleus are non-negative
        if len(w[w < -atol]) == 0:
            # CP-map Kraus representation
            kraus = []
            for val, vec in zip(w, v.T):
                if abs(val) > atol:
                    k = np.sqrt(val) * vec.reshape((output_dim, input_dim), order='F')
                    kraus.append(k)
            # If we are converting a zero matrix, we need to return a Kraus set
            # with a single zero-element Kraus matrix
            if not kraus:
                kraus.append(np.zeros((output_dim, input_dim), dtype=complex))
            return (kraus, None)
    # Non-CP-map generalized Kraus representation
    U, s, Vh = la.svd(data)
    kraus_l = []
    kraus_r = []
    for val, vecL, vecR in zip(s, U.T, Vh.conj()):
        kraus_l.append(np.sqrt(val) * vecL.reshape((output_dim, input_dim), order='F'))
        kraus_r.append(np.sqrt(val) * vecR.reshape((output_dim, input_dim), order='F'))
    return (kraus_l, kraus_r)


def _stinespring_to_kraus(data, input_dim, output_dim):
    """Transform Stinespring representation to Kraus representation."""
    kraus_pair = []
    for stine in data:
        if stine is None:
            kraus_pair.append(None)
        else:
            trace_dim = stine.shape[0] // output_dim
            iden = np.eye(output_dim)
            kraus = []
            for j in range(trace_dim):
                v = np.zeros(trace_dim)
                v[j] = 1
                kraus.append(np.kron(iden, v[None, :]).dot(stine))
            kraus_pair.append(kraus)
    return tuple(kraus_pair)


def _kraus_to_stinespring(data, input_dim, output_dim):
    """Transform Kraus representation to Stinespring representation."""
    stine_pair = [None, None]
    for i, kraus in enumerate(data):
        if kraus is not None:
            num_kraus = len(kraus)
            stine = np.zeros((output_dim * num_kraus, input_dim), dtype=complex)
            for j, k in enumerate(kraus):
                v = np.zeros(num_kraus)
                v[j] = 1
                stine += np.kron(k, v[:, None])
            stine_pair[i] = stine
    return tuple(stine_pair)


def _kraus_to_superop(data, input_dim, output_dim):
    """Transform Kraus representation to SuperOp representation."""
    kraus_l, kraus_r = data
    superop = 0
    if kraus_r is None:
        for a in kraus_l:
            superop += np.kron(np.conj(a), a)
    else:
        for a, b in zip(kraus_l, kraus_r):
            superop += np.kron(np.conj(b), a)
    return superop


def _chi_to_choi(data, input_dim, output_dim):
    """Transform Chi representation to a Choi representation."""
    num_qubits = int(np.log2(input_dim))
    return _transform_basis(data, num_qubits, _pauli2col(), 2)


def _choi_to_chi(data, input_dim, output_dim):
    """Transform Choi representation to the Chi representation."""
    num_qubits = int(np.log2(input_dim))
    return _transform_basis(data, num_qubits, _col2pauli(), 2)


def _ptm_to_superop(data, input_dim, output_dim):
    """Transform PTM representation to SuperOp representation."""
    num_qubits = int(np.log2(input_dim))
    return _transform_basis(data, num_qubits, _pauli2col(), 2)


def _superop_to_ptm(data, input_dim, output_dim):
    """Transform SuperOp representation to PTM representation."""
    num_qubits = int(np.log2(input_dim))
    return _transform_basis(data, num_qubits, _col2pauli(), 2)


def _bipartite_tensor(a, b, front=False, shape_a=None, shape_b=None):
    """Tensor product to bipartite matrices and reravel indicies.

    This is used for tensor product of superoperators and Choi matrices.

    Args:
        a (matrix like): a bipartite matrix
        b (matrix like): a bipartite matrix
        front (bool): If False return (b ⊗ a) if True return (a ⊗ b) [Default: False]
        shape_a (tuple): bipartite-shape for matrix a (a0, a1, a2, a3)
        shape_b (tuple): bipartite-shape for matrix b (b0, b1, b2, b3)

    Returns:
        a bipartite matrix for reravel(a ⊗ b) or  reravel(b ⊗ a).
    """
    # Convert inputs to numpy arrays
    a = np.array(a)
    b = np.array(b)

    # Determine bipartite dimensions if not provided
    dim_a0, dim_a1 = a.shape
    dim_b0, dim_b1 = b.shape
    if shape_a is None:
        a0 = int(np.sqrt(dim_a0))
        a1 = int(np.sqrt(dim_a1))
        shape_a = (a0, a0, a1, a1)
    if shape_b is None:
        b0 = int(np.sqrt(dim_b0))
        b1 = int(np.sqrt(dim_b1))
        shape_b = (b0, b0, b1, b1)
    # Check dimensions
    if len(shape_a) != 4 or shape_a[0] * shape_a[1] != dim_a0 or \
            shape_a[2] * shape_a[3] != dim_a1:
        raise QiskitError("Invalid shape_a")
    if len(shape_b) != 4 or shape_b[0] * shape_b[1] != dim_b0 or \
            shape_b[2] * shape_b[3] != dim_b1:
        raise QiskitError("Invalid shape_b")

    if front:
        # Return A ⊗ B
        return _reravel(a, b, shape_a, shape_b)
    # Return B ⊗ A
    return _reravel(b, a, shape_b, shape_a)


def _reravel(a, b, shape_a, shape_b):
    """
    shape_a and shape_b length-4 lists.
    """
    data = np.kron(a, b)
    # Reshuffle indicies
    left_dims = shape_a[:2] + shape_b[:2]
    right_dims = shape_a[2:] + shape_b[2:]
    tensor_shape = left_dims + right_dims
    final_shape = (np.product(left_dims), np.product(right_dims))

    # Tensor product matrices
    data = np.kron(a, b)
    data = np.reshape(np.transpose(np.reshape(data, tensor_shape),
                                   (0, 2, 1, 3, 4, 6, 5, 7)),
                      final_shape)
    return data


def _transform_basis(data, num_qubits, basis_mat, factor=1):
    """Change of basis of bipartite matrix represenation."""
    # Change basis
    # Note that we manually renormalized after change of basis
    # to avoid rounding errors from square-roots of 2.
    cob = basis_mat
    for j in range(num_qubits - 1):
        dim = int(np.sqrt(len(cob)))
        cob = _bipartite_tensor(basis_mat, cob, (2, 2, 2, 2), (dim, dim, dim, dim))
    return np.dot(np.dot(cob, data), cob.conj().T) / factor ** num_qubits


def _pauli2col():
    """Change of basis matrix from col-vec to Pauli."""
    # This matrix is given by:
    # sum_{i=0}^3 =|\sigma_i>><i|=
    return np.array([[1, 0, 0, 1],
                     [0, 1, 1j, 0],
                     [0, 1, -1j, 0],
                     [1, 0j, 0, -1]], dtype=complex)


def _col2pauli():
    """Change of basis matrix from Pauli col-vec."""
    # This matrix is given by:
    # sum_{i=0}^3 |i>><\sigma_i|
    return np.array([[1, 0, 0, 1],
                     [0, 1, 1, 0],
                     [0, -1j, 1j, 0],
                     [1, 0j, 0, -1]], dtype=complex)


def _reshuffle(a, shape):
    """Reshuffle the indicies of a bipartite matrix A[ij,kl] -> A[lj,ki]."""
    return np.reshape(np.transpose(np.reshape(a, shape), (3, 1, 2, 0)),
                      (shape[3] * shape[1], shape[0] * shape[2]))


def _check_nqubit_dim(input_dim, output_dim):
    """Return true if dims correspond to an n-qubit channel."""
    if input_dim != output_dim:
        raise QiskitError('Not an n-qubit channel: input_dim' +
                          ' ({}) != output_dim ({})'.format(input_dim, output_dim))
    num_qubits = int(np.log2(input_dim))
    if 2 ** num_qubits != input_dim:
        raise QiskitError('Not an n-qubit channel: input_dim != 2 ** n')
