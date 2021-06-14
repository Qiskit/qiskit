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
"""Fast multiplication algorithm."""

from typing import Optional

import numpy as np

from .fast_grad_utils import get_max_num_bits
from .layer import Layer1Q, Layer2Q


class PMatrix:
    """
    Wrapper around a matrix that enables fast multiplication by permutation
    matrices and block-diagonal ones.
    """

    def __init__(self, M: Optional[np.ndarray] = None):
        """
        Args:
            M: matrix we want to multiply on the left and on the right by layer matrices.

                N O T E: the external matrix will be modified, i.e. the class keeps matrix
                reference and alters its content in-place. This is done on purpose, because
                the matrix can be huge.
        """
        if M is not None:
            assert isinstance(M, np.ndarray)
            N = M.shape[0]
            assert M.shape == (N, N) and M.dtype == np.cfloat
            self._N = N
            self._M = M
            self._temp_G2x2 = np.full((2, 2), fill_value=0, dtype=np.cfloat)
            self._temp_G4x4 = np.full((4, 4), fill_value=0, dtype=np.cfloat)
            self._temp_2x2 = self._temp_G2x2.copy()
            self._temp_4x4 = self._temp_G4x4.copy()
            self._identity_perm = np.arange(N, dtype=np.int64)
            self._left_perm = self._identity_perm.copy()
            self._right_perm = self._identity_perm.copy()
            self._temp_perm = self._identity_perm.copy()
            self._temp_slice_Nx2 = np.full((N, 2), fill_value=0, dtype=np.cfloat)
            self._temp_slice_Nx4 = np.full((N, 4), fill_value=0, dtype=np.cfloat)
            self._I = self._init_index_matrix(N=N)
            self._temp_block_diag = np.full(shape=self._I.shape, fill_value=0, dtype=np.cfloat)
        else:
            self._N = int(0)
            self._M = np.empty(0)
            self._temp_G2x2 = np.empty(0)
            self._temp_G4x4 = np.empty(0)
            self._temp_2x2 = np.empty(0)
            self._temp_4x4 = np.empty(0)
            self._identity_perm = np.empty(0)
            self._left_perm = np.empty(0)
            self._right_perm = np.empty(0)
            self._temp_perm = np.empty(0)
            self._temp_slice_Nx2 = np.empty(0)
            self._temp_slice_Nx4 = np.empty(0)
            self._I = np.empty(0)
            self._temp_block_diag = np.empty(0)

    def initialize(self, num_qubits: int):
        """
        Initializes the internal structures of this object but does not set the matrix yet.

        Args:
            num_qubits: number of qubits.
        """
        assert isinstance(num_qubits, int) and 2 <= num_qubits <= get_max_num_bits()
        N = 2 ** num_qubits
        self._N = N
        self._temp_G2x2 = np.full((2, 2), fill_value=0, dtype=np.cfloat)
        self._temp_G4x4 = np.full((4, 4), fill_value=0, dtype=np.cfloat)
        self._temp_2x2 = self._temp_G2x2.copy()
        self._temp_4x4 = self._temp_G4x4.copy()
        self._identity_perm = np.arange(N, dtype=np.int64)
        self._left_perm = self._identity_perm.copy()
        self._right_perm = self._identity_perm.copy()
        self._temp_perm = self._identity_perm.copy()
        self._temp_slice_Nx2 = np.full((N, 2), fill_value=0, dtype=np.cfloat)
        self._temp_slice_Nx4 = np.full((N, 4), fill_value=0, dtype=np.cfloat)
        self._I = self._init_index_matrix(N=N)
        self._temp_block_diag = np.full(shape=self._I.shape, fill_value=0, dtype=np.cfloat)

    def set_matrix(self, M: np.ndarray):
        """
        Copies specified matrix to internal storage. The call to this function must be preceded by
        the call to self.initialize() one. Once the matrix is set, the object is ready for use.
        Note, the matrix will be copied, mind the size issues.
        """
        assert isinstance(M, np.ndarray)
        assert M.shape == (self._N, self._N) and M.dtype == np.cfloat
        if self._M.size == 0:
            self._M = M.copy()
        else:
            np.copyto(self._M, M)

    @staticmethod
    def _init_index_matrix(N: int) -> np.ndarray:
        assert isinstance(N, int) and N >= 2
        all_idx = np.arange(N * N, dtype=np.int64).reshape(N, N)
        I = np.full((N // 4, 4 * 4), fill_value=0, dtype=np.int64)
        b = np.full((4, 4), fill_value=0, dtype=np.int64)
        for i in range(0, N, 4):
            b[:, :] = all_idx[i : i + 4, i : i + 4]
            I[i // 4, :] = b.T.ravel()
        return I

    def mul_right_q1(self, L: Layer1Q, temp_mat: np.ndarray, dagger: bool):
        """
        Multiplies NxN matrix, wrapped by this object, by a 1-qubit layer matrix of the right,
        where N is the actual size of matrices involved, N = 2^{num. of qubits}.

        Args:
            L: 1-qubit layer, i.e. the layer with just one non-trivial 1-qubit gate and other gates
                are just identity operators.
            temp_mat: a temporary NxN matrix used as a workspace.
            dagger: if true, the right-hand side matrix will be taken as conjugate transposed.
        """
        assert isinstance(L, Layer1Q) and isinstance(temp_mat, np.ndarray)
        assert isinstance(dagger, bool)

        G, perm, inv_perm = L.get_attr()
        M = self._M
        N = perm.size
        assert M.shape == temp_mat.shape == (N, N) and temp_mat.dtype == M.dtype

        # temp_mat <-- M[:, right_perm[perm]] = M[:, right_perm][:, perm]:
        np.take(M, np.take(self._right_perm, perm, out=self._temp_perm), axis=1, out=temp_mat)

        # M <-- M[:, right_perm][:, perm] @ kron(I(N/4), G)^dagger, where
        # conjugate-transposition might be or might be not applied:
        G_right = np.conj(G, out=self._temp_G2x2).T if dagger else G
        for i in range(0, N, 2):
            M[:, i : i + 2] = np.dot(temp_mat[:, i : i + 2], G_right, out=self._temp_slice_Nx2)

        # Update right permutation:
        self._right_perm[:] = inv_perm

    def mul_right_q2(self, L: Layer2Q, temp_mat: np.ndarray, dagger: bool = True):
        """
        Multiplies NxN matrix, wrapped by this object, by a 2-qubit layer matrix on the right,
        where N is the actual size of matrices involved, N = 2^{num. of qubits}.

        Args:
            L: 2-qubit layer, i.e. the layer with just one non-trivial 2-qubit gate and other
                gates are just identity operators.
            temp_mat: a temporary NxN matrix used as a workspace.
            dagger: if true, the right-hand side matrix will be taken as conjugate transposed.
        """
        assert isinstance(L, Layer2Q) and isinstance(temp_mat, np.ndarray)
        assert isinstance(dagger, bool)

        G, perm, inv_perm = L.get_attr()
        M = self._M
        N = perm.size
        assert M.shape == temp_mat.shape == (N, N) and temp_mat.dtype == M.dtype

        # temp_mat <-- M[:, right_perm[perm]] = M[:, right_perm][:, perm]:
        np.take(M, np.take(self._right_perm, perm, out=self._temp_perm), axis=1, out=temp_mat)

        # M <-- M[:, right_perm][:, perm] @ kron(I(N/4), G)^dagger, where
        # conjugate-transposition might be or might be not applied:
        G_right = np.conj(G, out=self._temp_G4x4).T if dagger else G
        for i in range(0, N, 4):
            M[:, i : i + 4] = np.dot(temp_mat[:, i : i + 4], G_right, out=self._temp_slice_Nx4)

        # Update right permutation:
        self._right_perm[:] = inv_perm

    def mul_left_q1(self, L: Layer1Q, temp_mat: np.ndarray):
        """
        Multiplies NxN matrix, wrapped by this object, by a 1-qubit layer matrix of the left,
        where N is the actual size of matrices involved, N = 2^{num. of qubits}.

        Args:
            L: 1-qubit layer, i.e. the layer with just one non-trivial 1-qubit gate and other gates
                are just identity operators.
            temp_mat: a temporary NxN matrix used as a workspace.
        """
        M = self._M
        assert isinstance(L, Layer1Q)
        assert isinstance(temp_mat, np.ndarray) and id(temp_mat) != id(M)
        assert temp_mat.shape == M.shape and temp_mat.dtype == M.dtype
        G, perm, inv_perm = L.get_attr()
        N = perm.size
        assert M.shape == (N, N)

        # temp_mat <-- M[left_perm[perm]] = M[left_perm][perm]:
        np.take(M, np.take(self._left_perm, perm, out=self._temp_perm), axis=0, out=temp_mat)

        # M <-- kron(I(N/4), G) @ M[left_perm][perm]:
        if N > 512:
            # Faster for large matrices.
            for i in range(0, N, 2):
                np.dot(G, temp_mat[i : i + 2, :], out=M[i : i + 2, :])
        else:
            # Faster for small matrices.
            K = N // 2
            np.copyto(M.reshape((2, K, N)), np.swapaxes(temp_mat.reshape((K, 2, N)), 0, 1))
            np.dot(G, M.reshape(2, -1), out=temp_mat.reshape(2, -1))
            np.copyto(M.reshape((K, 2, N)), np.swapaxes(temp_mat.reshape((2, K, N)), 0, 1))

        # Update left permutation:
        self._left_perm[:] = inv_perm

    def mul_left_q2(self, L: Layer2Q, temp_mat: np.ndarray):
        """
        Multiplies NxN matrix, wrapped by this object, by a 2-qubit layer matrix on the left,
        where N is the actual size of matrices involved, N = 2^{num. of qubits}.

        Args:
            L: 2-qubit layer, i.e. the layer with just one non-trivial 2-qubit gate and other gates
                are just identity operators.
            temp_mat: a temporary NxN matrix used as a workspace.
        """
        M = self._M
        assert isinstance(L, Layer2Q)
        assert isinstance(temp_mat, np.ndarray) and id(temp_mat) != id(M)
        assert temp_mat.shape == M.shape and temp_mat.dtype == M.dtype
        G, perm, inv_perm = L.get_attr()
        N = perm.size
        assert M.shape == (N, N)

        # temp_mat <-- M[left_perm[perm]] = M[left_perm][perm]:
        np.take(M, np.take(self._left_perm, perm, out=self._temp_perm), axis=0, out=temp_mat)

        # M <-- kron(I(N/4), G) @ M[left_perm][perm]:
        if N > 512:
            # Faster for large matrices.
            for i in range(0, N, 4):
                np.dot(G, temp_mat[i : i + 4, :], out=M[i : i + 4, :])
        else:
            # Faster for small matrices.
            K = N // 4
            np.copyto(M.reshape((4, K, N)), np.swapaxes(temp_mat.reshape((K, 4, N)), 0, 1))
            np.dot(G, M.reshape(4, -1), out=temp_mat.reshape(4, -1))
            np.copyto(M.reshape((K, 4, N)), np.swapaxes(temp_mat.reshape((4, K, N)), 0, 1))

        # Update left permutation:
        self._left_perm[:] = inv_perm

    def product_q1(self, L: Layer1Q, tmpA: np.ndarray, tmpB: np.ndarray) -> np.cfloat:
        """
        Computes and returns: Trace(M @ C) = Trace(M @ P^T @ G @ P) = Trace((P @ M @ P^T) @ G) =
        Trace(C @ (P @ M @ P^T)) = vec(G^T)^T @ vec(P @ M @ P^T), where M is NxN matrix wrapped
        by this object, C is matrix representation of the layer L, and G is 2x2 matrix of
        underlying 1-qubit gate.

        N O T E: matrix of this class must be finalized beforehand.

        Args:
            L: 1-qubit layer.
            tmpA: temporary, external matrix used as a workspace.
            tmpB: temporary, external matrix used as a workspace.

        Returns:
            TODO: what is the return value?
        """
        M = self._M
        assert isinstance(L, Layer1Q)
        assert id(tmpA) != id(M) != id(tmpB) and id(tmpA) != id(tmpB)
        assert M.dtype == tmpA.dtype == tmpB.dtype
        G, perm, _ = L.get_attr()

        # tmpB = P @ M @ P^T:
        np.take(np.take(M, perm, axis=0, out=tmpA), perm, axis=1, out=tmpB)

        # matrix dot product = Tr(transposed(kron(I(N/4), G)), (P @ M @ P^T)):
        # TODO: why do we have compact_version = False ?
        # TODO: NotImplementedError
        compact_version = False
        if compact_version:
            # bldia = self._temp_block_diag
            # np.take(tmpB.ravel(), self._I.ravel(), axis=0, out=bldia.ravel())
            # bldia *= G.reshape(-1, G.size)
            # return np.cfloat(np.sum(bldia))
            raise NotImplementedError("in product_q1()")
        # else:
        Gt, T = self._temp_G2x2, self._temp_2x2
        np.copyto(Gt, G.T)
        _sum = 0.0
        for i in range(0, M.shape[0], 2):
            T[:, :] = tmpB[i : i + 2, i : i + 2]
            _sum += np.dot(Gt.ravel(), T.ravel())
        return np.cfloat(_sum)

    def product_q2(self, L: Layer2Q, tmpA: np.ndarray, tmpB: np.ndarray) -> np.cfloat:
        """
        Computes and returns: Trace(M @ C) = Trace(M @ P^T @ G @ P) = Trace((P @ M @ P^T) @ G) =
        Trace(C @ (P @ M @ P^T)) = vec(G^T)^T @ vec(P @ M @ P^T), where M is NxN matrix wrapped
        by this object, C is matrix representation of the layer L, and G is 4x4 matrix of
        underlying 2-qubit gate.

        N O T E: matrix of this class must be finalized beforehand.

        Args:
            L: 2-qubit layer.
            tmpA: temporary, external matrix used as a workspace.
            tmpB: temporary, external matrix used as a workspace.

        Returns:
            TODO: what is the return value?
        """
        M = self._M
        assert isinstance(L, Layer2Q)
        assert id(tmpA) != id(M) != id(tmpB) and id(tmpA) != id(tmpB)
        assert M.dtype == tmpA.dtype == tmpB.dtype
        G, perm, _ = L.get_attr()

        # matrix dot product = Tr(transposed(kron(I(N/4), G)), (P @ M @ P^T)):
        _version = 1
        if _version == 1:
            # The fastest version so far, but requires two NxN temp. matrices.
            # tmpB = P @ M @ P^T:
            np.take(np.take(M, perm, axis=0, out=tmpA), perm, axis=1, out=tmpB)

            bldia = self._temp_block_diag
            np.take(tmpB.ravel(), self._I.ravel(), axis=0, out=bldia.ravel())
            bldia *= G.reshape(-1, G.size)
            return np.cfloat(np.sum(bldia))
        elif _version == 2:
            # The second fastest, but very close to the first one. Requires
            # two NxN temporary matrices.
            # tmpB = P @ M @ P^T:
            np.take(np.take(M, perm, axis=0, out=tmpA), perm, axis=1, out=tmpB)

            Gt, T = self._temp_G4x4, self._temp_4x4
            np.copyto(Gt, G.T)
            _sum = 0.0
            for i in range(0, M.shape[0], 4):
                T[:, :] = tmpB[i : i + 4, i : i + 4]
                _sum += np.dot(Gt.ravel(), T.ravel())
            return np.cfloat(_sum)
        else:  # _version == 3
            # This is the least memory consuming version, but the slowest one.
            # Possibly, np.ix_() is slow (a new object is created on each step).
            # However, C++ implementation should be fast (the fastest?).
            N = self._N
            Gt, T = self._temp_G4x4, self._temp_4x4
            np.copyto(Gt, G.T)
            _sum = 0.0
            for i in range(0, N, 4):
                idx = perm[i : i + 4]
                T[:, :] = M[np.ix_(idx, idx)]
                _sum += np.dot(Gt.ravel(), T.ravel())
            return np.cfloat(_sum)

    def finalize(self, temp_mat: np.ndarray) -> np.ndarray:
        """
        Applies the left (row) and right (column) permutations to the matrix. at the end of
        computation process.

        Args:
            temp_mat: temporary, external matrix.

        Returns:
            finalized matrix with all transformations applied.
        """
        M = self._M
        assert isinstance(temp_mat, np.ndarray) and id(temp_mat) != id(M)
        assert temp_mat.shape == M.shape and temp_mat.dtype == M.dtype

        # M <-- M[left_perm][:, right_perm] = P_left @ M @ transposed(P_right)
        np.take(M, self._left_perm, axis=0, out=temp_mat)
        np.take(temp_mat, self._right_perm, axis=1, out=M)

        # Set both permutations to identity once they have been applied.
        self._left_perm[:] = self._identity_perm
        self._right_perm[:] = self._identity_perm
        return self._M

    # def product2(self, L: Layer2Q, tmpA: np.ndarray, tmpB: np.ndarray) -> float:
    #     """
    #     N O T E: matrix of this class must be finalized beforehand.
    #     :param L:
    #     :param tmpA:
    #     :param tmpB:
    #     :return:
    #     """
    #     M = self._M
    #     assert id(tmpA) != id(M) != id(tmpB) and id(tmpA) != id(tmpB)
    #     assert M.dtype == tmpA.dtype == tmpB.dtype
    #     G, perm, _ = L.get_attr()
    #
    #     # tmpB = P @ M @ P^T:
    #     np.take(np.take(M, perm, axis=0, out=tmpA), perm, axis=1, out=tmpB)
    #
    #     # matrix dot product = Tr(transposed(kron(I(N/4), G)), (P @ M @ P^T)):
    #     Gt, T = self._temp_G4x4, self._temp_4x4
    #     np.copyto(Gt, G.T)
    #     _sum = 0.0
    #     for i in range(0, M.shape[0], 4):
    #         T[:, :] = tmpB[i : i + 4, i : i + 4]
    #         _sum += np.dot(Gt.ravel(), T.ravel())
    #     return _sum
    #
