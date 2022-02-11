# This code is part of Qiskit.
#
# (C) Copyright IBM 2021.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Matrix designed for fast multiplication by permutation and block-diagonal ones.
"""

from typing import Optional
import numpy as np
# from .fast_grad_utils import get_max_num_bits
from .layer import Layer1Q, Layer2Q


class PMatrix:
    """
    Wrapper around a matrix that enables fast multiplication by permutation
    matrices and block-diagonal ones.
    """

    def __init__(self, mat: Optional[np.ndarray] = None):
        """
        Constructor.
        N O T E: the external matrix will be modified, i.e. the class keeps
        matrix reference and alters its content in-place. This is done on
        purpose, because the matrix can be huge.

        Args:
            mat: matrix we want to multiply on the left and on the right by
                 layer matrices.
        """
        if mat is not None:
            # assert isinstance(mat, np.ndarray)
            dim = mat.shape[0]
            # assert mat.shape == (dim, dim) and mat.dtype == np.cfloat
            self._dim = dim
            self._mat = mat
            self._temp_g2x2 = np.zeros((2, 2), dtype=np.cfloat)
            self._temp_g4x4 = np.zeros((4, 4), dtype=np.cfloat)
            self._temp_2x2 = self._temp_g2x2.copy()
            self._temp_4x4 = self._temp_g4x4.copy()
            self._identity_perm = np.arange(dim, dtype=np.int64)
            self._left_perm = self._identity_perm.copy()
            self._right_perm = self._identity_perm.copy()
            self._temp_perm = self._identity_perm.copy()
            self._temp_slice_dim_x_2 = np.zeros((dim, 2), dtype=np.cfloat)
            self._temp_slice_dim_x_4 = np.zeros((dim, 4), dtype=np.cfloat)
            self._idx_mat = self._init_index_matrix(dim)
            self._temp_block_diag = np.zeros(self._idx_mat.shape, dtype=np.cfloat)
        else:
            self._dim = int(0)
            self._mat = np.empty(0)
            self._temp_g2x2 = np.empty(0)
            self._temp_g4x4 = np.empty(0)
            self._temp_2x2 = np.empty(0)
            self._temp_4x4 = np.empty(0)
            self._identity_perm = np.empty(0)
            self._left_perm = np.empty(0)
            self._right_perm = np.empty(0)
            self._temp_perm = np.empty(0)
            self._temp_slice_dim_x_2 = np.empty(0)
            self._temp_slice_dim_x_4 = np.empty(0)
            self._idx_mat = np.empty(0)
            self._temp_block_diag = np.empty(0)

    def initialize(self, num_qubits: int):
        """
        Initializes the internal structures of this object but does not set
        the matrix yet.

        Args:
            num_qubits: number of qubits.
        """
        # assert isinstance(num_qubits, int) and 2 <= num_qubits <= get_max_num_bits()
        dim = 2**num_qubits
        self._dim = dim
        self._temp_g2x2 = np.zeros((2, 2), dtype=np.cfloat)
        self._temp_g4x4 = np.zeros((4, 4), dtype=np.cfloat)
        self._temp_2x2 = self._temp_g2x2.copy()
        self._temp_4x4 = self._temp_g4x4.copy()
        self._identity_perm = np.arange(dim, dtype=np.int64)
        self._left_perm = self._identity_perm.copy()
        self._right_perm = self._identity_perm.copy()
        self._temp_perm = self._identity_perm.copy()
        self._temp_slice_dim_x_2 = np.zeros((dim, 2), dtype=np.cfloat)
        self._temp_slice_dim_x_4 = np.zeros((dim, 4), dtype=np.cfloat)
        self._idx_mat = self._init_index_matrix(dim)
        self._temp_block_diag = np.zeros(self._idx_mat.shape, dtype=np.cfloat)

    def set_matrix(self, mat: np.ndarray):
        """
        Copies specified matrix to internal storage. The call to this function
        must be preceded by the call to self.initialize() one. Once the matrix
        is set, the object is ready for use.
        N O T E, the matrix will be copied, mind the size issues.

        Args:
            mat: matrix we want to multiply on the left and on the right by
                 layer matrices.
        """
        # assert isinstance(mat, np.ndarray)
        # assert mat.shape == (self._dim, self._dim) and mat.dtype == np.cfloat
        if self._mat.size == 0:
            self._mat = mat.copy()
        else:
            np.copyto(self._mat, mat)

    @staticmethod
    def _init_index_matrix(dim: int) -> np.ndarray:
        """
        Fast multiplication can be implemented by picking up a subset of
        entries in a sparse matrix.

        Args:
            dim: problem dimensionality.

        Returns:
            2d-array of indices for the fast multiplication.
        """
        # assert isinstance(dim, int) and dim >= 2
        all_idx = np.arange(dim * dim, dtype=np.int64).reshape(dim, dim)
        idx = np.full((dim // 4, 4 * 4), fill_value=0, dtype=np.int64)
        b = np.full((4, 4), fill_value=0, dtype=np.int64)
        for i in range(0, dim, 4):
            b[:, :] = all_idx[i: i + 4, i: i + 4]
            idx[i // 4, :] = b.T.ravel()
        return idx

    def mul_right_q1(self, layer: Layer1Q, temp_mat: np.ndarray, dagger: bool):
        """
        Multiplies NxN matrix, wrapped by this object, by a 1-qubit layer
        matrix of the right, where N is the actual size of matrices involved,
        N = 2^{num. of qubits}.

        Args:
            layer: 1-qubit layer, i.e. the layer with just one non-trivial
                   1-qubit gate and other gates are just identity operators.
            temp_mat: a temporary NxN matrix used as a workspace.
            dagger: if true, the right-hand side matrix will be taken as
                    conjugate transposed.
        """
        # assert isinstance(layer, Layer1Q) and isinstance(temp_mat, np.ndarray)
        # assert isinstance(dagger, bool)

        gmat, perm, inv_perm = layer.get_attr()
        mat = self._mat
        dim = perm.size
        # assert mat.shape == temp_mat.shape == (dim, dim) and temp_mat.dtype == mat.dtype

        # temp_mat <-- mat[:, right_perm[perm]] = mat[:, right_perm][:, perm]:
        np.take(mat, np.take(self._right_perm, perm, out=self._temp_perm), axis=1, out=temp_mat)

        # mat <-- mat[:, right_perm][:, perm] @ kron(I(N/4), gmat)^dagger, where
        # conjugate-transposition might be or might be not applied:
        gmat_right = np.conj(gmat, out=self._temp_g2x2).T if dagger else gmat
        for i in range(0, dim, 2):
            mat[:, i: i + 2] = np.dot(
                temp_mat[:, i: i + 2], gmat_right, out=self._temp_slice_dim_x_2
            )

        # Update right permutation:
        self._right_perm[:] = inv_perm

    def mul_right_q2(self, layer: Layer2Q, temp_mat: np.ndarray, dagger: bool = True):
        """
        Multiplies NxN matrix, wrapped by this object, by a 2-qubit layer
        matrix on the right, where N is the actual size of matrices involved,
        N = 2^{num. of qubits}.

        Args:
            layer: 2-qubit layer, i.e. the layer with just one non-trivial
                   2-qubit gate and other gates are just identity operators.
            temp_mat: a temporary NxN matrix used as a workspace.
            dagger: if true, the right-hand side matrix will be taken as
                    conjugate transposed.
        """
        # assert isinstance(layer, Layer2Q) and isinstance(temp_mat, np.ndarray)
        # assert isinstance(dagger, bool)

        gmat, perm, inv_perm = layer.get_attr()
        mat = self._mat
        dim = perm.size
        # assert mat.shape == temp_mat.shape == (dim, dim) and temp_mat.dtype == mat.dtype

        # temp_mat <-- mat[:, right_perm[perm]] = mat[:, right_perm][:, perm]:
        np.take(mat, np.take(self._right_perm, perm, out=self._temp_perm), axis=1, out=temp_mat)

        # mat <-- mat[:, right_perm][:, perm] @ kron(I(N/4), gmat)^dagger, where
        # conjugate-transposition might be or might be not applied:
        gmat_right = np.conj(gmat, out=self._temp_g4x4).T if dagger else gmat
        for i in range(0, dim, 4):
            mat[:, i: i + 4] = np.dot(
                temp_mat[:, i: i + 4], gmat_right, out=self._temp_slice_dim_x_4
            )

        # Update right permutation:
        self._right_perm[:] = inv_perm

    def mul_left_q1(self, layer: Layer1Q, temp_mat: np.ndarray):
        """
        Multiplies NxN matrix, wrapped by this object, by a 1-qubit layer
        matrix of the left, where dim is the actual size of matrices involved,
        dim = 2^{num. of qubits}.

        Args:
            layer: 1-qubit layer, i.e. the layer with just one non-trivial
                   1-qubit gate and other gates are just identity operators.
            temp_mat: a temporary NxN matrix used as a workspace.
        """
        mat = self._mat
        # assert isinstance(layer, Layer1Q)
        # assert isinstance(temp_mat, np.ndarray) and id(temp_mat) != id(mat)
        # assert temp_mat.shape == mat.shape and temp_mat.dtype == mat.dtype
        gmat, perm, inv_perm = layer.get_attr()
        dim = perm.size
        # assert mat.shape == (dim, dim)

        # temp_mat <-- mat[left_perm[perm]] = mat[left_perm][perm]:
        np.take(mat, np.take(self._left_perm, perm, out=self._temp_perm), axis=0, out=temp_mat)

        # mat <-- kron(I(dim/4), gmat) @ mat[left_perm][perm]:
        if dim > 512:
            # Faster for large matrices.
            for i in range(0, dim, 2):
                np.dot(gmat, temp_mat[i: i + 2, :], out=mat[i: i + 2, :])
        else:
            # Faster for small matrices.
            half = dim // 2
            np.copyto(
                mat.reshape((2, half, dim)), np.swapaxes(temp_mat.reshape((half, 2, dim)), 0, 1)
            )
            np.dot(gmat, mat.reshape(2, -1), out=temp_mat.reshape(2, -1))
            np.copyto(
                mat.reshape((half, 2, dim)), np.swapaxes(temp_mat.reshape((2, half, dim)), 0, 1)
            )

        # Update left permutation:
        self._left_perm[:] = inv_perm

    def mul_left_q2(self, layer: Layer2Q, temp_mat: np.ndarray):
        """
        Multiplies NxN matrix, wrapped by this object, by a 2-qubit layer
        matrix on the left, where dim is the actual size of matrices involved,
        dim = 2^{num. of qubits}.

        Args:
            layer: 2-qubit layer, i.e. the layer with just one non-trivial
                   2-qubit gate and other gates are just identity operators.
            temp_mat: a temporary NxN matrix used as a workspace.
        """
        mat = self._mat
        # assert isinstance(layer, Layer2Q)
        # assert isinstance(temp_mat, np.ndarray) and id(temp_mat) != id(mat)
        # assert temp_mat.shape == mat.shape and temp_mat.dtype == mat.dtype
        gmat, perm, inv_perm = layer.get_attr()
        dim = perm.size
        # assert mat.shape == (dim, dim)

        # temp_mat <-- mat[left_perm[perm]] = mat[left_perm][perm]:
        np.take(mat, np.take(self._left_perm, perm, out=self._temp_perm), axis=0, out=temp_mat)

        # mat <-- kron(I(dim/4), gmat) @ mat[left_perm][perm]:
        if dim > 512:
            # Faster for large matrices.
            for i in range(0, dim, 4):
                np.dot(gmat, temp_mat[i: i + 4, :], out=mat[i: i + 4, :])
        else:
            # Faster for small matrices.
            half = dim // 4
            np.copyto(
                mat.reshape((4, half, dim)), np.swapaxes(temp_mat.reshape((half, 4, dim)), 0, 1)
            )
            np.dot(gmat, mat.reshape(4, -1), out=temp_mat.reshape(4, -1))
            np.copyto(
                mat.reshape((half, 4, dim)), np.swapaxes(temp_mat.reshape((4, half, dim)), 0, 1)
            )

        # Update left permutation:
        self._left_perm[:] = inv_perm

    def product_q1(self, layer: Layer1Q, tmp1: np.ndarray, tmp2: np.ndarray) -> np.cfloat:
        """
        Computes and returns: Trace(mat @ C) = Trace(mat @ P^T @ gmat @ P) =
        Trace((P @ mat @ P^T) @ gmat) = Trace(C @ (P @ mat @ P^T)) =
        vec(gmat^T)^T @ vec(P @ mat @ P^T), where mat is NxN matrix wrapped
        by this object, C is matrix representation of the layer L, and gmat
        is 2x2 matrix of underlying 1-qubit gate.

        N O T E: matrix of this class must be finalized beforehand.

        Args:
            layer: 1-qubit layer.
            tmp1: temporary, external matrix used as a workspace.
            tmp2: temporary, external matrix used as a workspace.

        Returns:
            trace of the matrix product.
        """
        mat = self._mat
        # assert isinstance(layer, Layer1Q)
        # assert id(tmp1) != id(mat) != id(tmp2) and id(tmp1) != id(tmp2)
        # assert mat.dtype == tmp1.dtype == tmp2.dtype
        gmat, perm, _ = layer.get_attr()

        # tmp2 = P @ mat @ P^T:
        np.take(np.take(mat, perm, axis=0, out=tmp1), perm, axis=1, out=tmp2)

        # matrix dot product = Tr(transposed(kron(I(dim/4), gmat)), (P @ mat @ P^T)):
        # TODO: why do we have compact_version = False, is it slower?
        # TODO: Clean-up experimental stuff.
        _compact_version = False
        if _compact_version:
            bldia = self._temp_block_diag
            np.take(tmp2.ravel(), self._idx_mat.ravel(), axis=0, out=bldia.ravel())
            bldia *= gmat.reshape(-1, gmat.size)
            return np.cfloat(np.sum(bldia))
        else:
            gmat_t, tmp3 = self._temp_g2x2, self._temp_2x2
            np.copyto(gmat_t, gmat.T)
            _sum = 0.0
            for i in range(0, mat.shape[0], 2):
                tmp3[:, :] = tmp2[i: i + 2, i: i + 2]
                _sum += np.dot(gmat_t.ravel(), tmp3.ravel())

        return np.cfloat(_sum)

    def product_q2(self, layer: Layer2Q, tmp1: np.ndarray, tmp2: np.ndarray) -> np.cfloat:
        """
        Computes and returns: Trace(mat @ C) = Trace(mat @ P^T @ gmat @ P) =
        Trace((P @ mat @ P^T) @ gmat) = Trace(C @ (P @ mat @ P^T)) =
        vec(gmat^T)^T @ vec(P @ mat @ P^T), where mat is NxN matrix wrapped
        by this object, C is matrix representation of the layer L, and gmat
        is 4x4 matrix of underlying 2-qubit gate.

        N O T E: matrix of this class must be finalized beforehand.

        Args:
            layer: 2-qubit layer.
            tmp1: temporary, external matrix used as a workspace.
            tmp2: temporary, external matrix used as a workspace.

        Returns:
            trace of the matrix product.
        """
        mat = self._mat
        # assert isinstance(layer, Layer2Q)
        # assert id(tmp1) != id(mat) != id(tmp2) and id(tmp1) != id(tmp2)
        # assert mat.dtype == tmp1.dtype == tmp2.dtype
        gmat, perm, _ = layer.get_attr()

        # matrix dot product = Tr(transposed(kron(I(dim/4), gmat)), (P @ mat @ P^T)):
        # TODO: clean up experimental code.
        # TODO: retain the fastest version only.
        _version = 1
        if _version == 1:
            # The fastest version so far, but requires two NxN temp. matrices.
            # tmp2 = P @ mat @ P^T:
            np.take(np.take(mat, perm, axis=0, out=tmp1), perm, axis=1, out=tmp2)

            bldia = self._temp_block_diag
            np.take(tmp2.ravel(), self._idx_mat.ravel(), axis=0, out=bldia.ravel())
            bldia *= gmat.reshape(-1, gmat.size)
            return np.cfloat(np.sum(bldia))
        elif _version == 2:
            # The second fastest, but very close to the first one. Requires
            # two NxN temporary matrices.
            # tmp2 = P @ mat @ P^T:
            np.take(np.take(mat, perm, axis=0, out=tmp1), perm, axis=1, out=tmp2)

            gmat_t, tmp3 = self._temp_g4x4, self._temp_4x4
            np.copyto(gmat_t, gmat.T)
            _sum = 0.0
            for i in range(0, mat.shape[0], 4):
                tmp3[:, :] = tmp2[i: i + 4, i: i + 4]
                _sum += np.dot(gmat_t.ravel(), tmp3.ravel())
            return np.cfloat(_sum)
        else:  # _version == 3
            # This is the least memory consuming version, but the slowest one.
            # Possibly, np.ix_() is slow (a new object is created on each step).
            # However, C++ implementation should be fast (the fastest?).
            dim = self._dim
            gmat_t, tmp3 = self._temp_g4x4, self._temp_4x4
            np.copyto(gmat_t, gmat.T)
            _sum = 0.0
            for i in range(0, dim, 4):
                idx = perm[i: i + 4]
                tmp3[:, :] = mat[np.ix_(idx, idx)]
                _sum += np.dot(gmat_t.ravel(), tmp3.ravel())
            return np.cfloat(_sum)

    def finalize(self, temp_mat: np.ndarray) -> np.ndarray:
        """
        Applies the left (row) and right (column) permutations to the matrix.
        at the end of computation process.

        Args:
            temp_mat: temporary, external matrix.

        Returns:
            finalized matrix with all transformations applied.
        """
        mat = self._mat
        # assert isinstance(temp_mat, np.ndarray) and id(temp_mat) != id(mat)
        # assert temp_mat.shape == mat.shape and temp_mat.dtype == mat.dtype

        # mat <-- mat[left_perm][:, right_perm] = P_left @ mat @ transposed(P_right)
        np.take(mat, self._left_perm, axis=0, out=temp_mat)
        np.take(temp_mat, self._right_perm, axis=1, out=mat)

        # Set both permutations to identity once they have been applied.
        self._left_perm[:] = self._identity_perm
        self._right_perm[:] = self._identity_perm
        return self._mat

    # def product2(self, L: Layer2Q, tmp1: np.ndarray, tmp2: np.ndarray) -> float:
    #     """
    #     N O T E: matrix of this class must be finalized beforehand.
    #     :param L:
    #     :param tmp1:
    #     :param tmp2:
    #     :return:
    #     """
    #     mat = self._mat
    #     # assert id(tmp1) != id(mat) != id(tmp2) and id(tmp1) != id(tmp2)
    #     # assert mat.dtype == tmp1.dtype == tmp2.dtype
    #     gmat, perm, _ = L.get_attr()
    #
    #     # tmp2 = P @ mat @ P^T:
    #     np.take(np.take(mat, perm, axis=0, out=tmp1), perm, axis=1, out=tmp2)
    #
    #     # matrix dot product = Tr(transposed(kron(I(dim/4), gmat)), (P @ mat @ P^T)):
    #     Gt, tmp3 = self._temp_g4x4, self._temp_4x4
    #     np.copyto(Gt, gmat.T)
    #     _sum = 0.0
    #     for i in range(0, mat.shape[0], 4):
    #         tmp3[:, :] = tmp2[i : i + 4, i : i + 4]
    #         _sum += np.dot(Gt.ravel(), tmp3.ravel())
    #     return _sum
    #
