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
"""Implementation of the fast gradient computation."""

import numpy as np

from .fast_grad_utils import get_max_num_bits, temporary_code
from .layer import (
    LayerBase,
    Layer2Q,
    Layer1Q,
    init_layer2q_matrices,
    init_layer2q_deriv_matrices,
    init_layer1q_matrices,
    init_layer1q_deriv_matrices,
)
from .pmatrix import PMatrix
from ..gradient import GradientBase


class FastGradient(GradientBase):
    """
    Fifth version of the fast algorithm for gradient computation that uses
    permutation of layer matrices.
    """

    def __init__(self, num_qubits: int, cnots: np.ndarray, verbose: int = 0):
        """Constructor."""
        super().__init__()
        assert isinstance(verbose, int)
        if verbose >= 1:
            print("Gradient:", self.__class__.__name__)

        n = num_qubits  # short-hand alias
        assert isinstance(n, int) and 2 <= n <= get_max_num_bits()
        assert isinstance(cnots, np.ndarray) and cnots.ndim == 2
        assert cnots.dtype == np.int64 and cnots.shape[0] == 2
        # pylint: disable=misplaced-comparison-constant
        assert np.all(1 <= cnots) and np.all(cnots <= n)

        self.num_qubits = n  # number of qubits
        self.cnots = cnots  # CNOT structure
        N = 2 ** n  # actual problem dimension
        L = cnots.shape[1]
        self._L = L  # number of C-layers

        self._U = np.empty(0)  # reference to external target matrix U
        self._UCF = PMatrix(M=None)  # U^dagger @ C @ F
        self._FUC = PMatrix(M=None)  # F @ U^dagger @ C
        self._UCF.initialize(n)
        self._FUC.initialize(n)

        # array of C-layers:
        self._C_layers = np.array([LayerBase()] * L, dtype=LayerBase)
        # array of F-layers:
        self._F_layers = np.array([LayerBase()] * n, dtype=LayerBase)
        # 4x4 C-gate matrices:
        self._C_gates = np.full((L, 4, 4), fill_value=0, dtype=np.cfloat)
        # derivatives of 4x4 C-gate matrices:
        self._C_dervs = np.full((L, 4, 4, 4), fill_value=0, dtype=np.cfloat)
        # 4x4 F-gate matrices:
        self._F_gates = np.full((n, 2, 2), fill_value=0, dtype=np.cfloat)
        # derivatives of 4x4 F-gate matrices:
        self._F_dervs = np.full((n, 3, 2, 2), fill_value=0, dtype=np.cfloat)
        # temporary NxN matrices:
        self._tmpA = np.full((N, N), fill_value=0, dtype=np.cfloat)
        self._tmpB = np.full((N, N), fill_value=0, dtype=np.cfloat)

        # Create layers of 2-qubit gates.
        for l in range(L):
            j = int(cnots[0, l]) - 1  # make 0-based index
            k = int(cnots[1, l]) - 1  # make 0-based index
            self._C_layers[l] = Layer2Q(nbits=n, j=j, k=k)

        # Create layers of 1-qubit gates.
        for k in range(n):
            self._F_layers[k] = Layer1Q(nbits=n, k=k)

        self._verbose = verbose
        self._debug = False
        if self._debug:
            temporary_code("Debugging mode is on")

    def get_gradient(self, thetas: np.ndarray, target_matrix: np.ndarray) -> (float, np.ndarray):
        """
        Computes the gradient of objective function.
        See description of the base class method.
        """
        L, n = self._L, self.num_qubits
        assert L >= 2
        assert isinstance(target_matrix, np.ndarray)
        assert target_matrix.shape == (2 ** n, 2 ** n)
        assert target_matrix.dtype == np.cfloat
        assert isinstance(thetas, np.ndarray) and thetas.ndim == 1
        assert thetas.size == 4 * L + 3 * n and thetas.dtype == np.float64
        self._U = target_matrix  # reference to external target matrix U

        thetas4L = thetas[: 4 * L].reshape(L, 4)
        thetas3n = thetas[4 * L :].reshape(n, 3)

        grad = np.full((thetas.size,), fill_value=0, dtype=np.float64)
        grad4L = grad[: 4 * L].reshape(L, 4)
        grad3n = grad[4 * L :].reshape(n, 3)

        init_layer2q_matrices(thetas=thetas4L, dst=self._C_gates)
        init_layer2q_deriv_matrices(thetas=thetas4L, dst=self._C_dervs)
        init_layer1q_matrices(thetas=thetas3n, dst=self._F_gates)
        init_layer1q_deriv_matrices(thetas=thetas3n, dst=self._F_dervs)

        self._init_layers()
        self._calc_ucf_fuc()
        objective_value = self._calc_objective_function()
        self._calc_gradient4L(grad4L)
        self._calc_gradient3n(grad3n)

        assert np.isreal(objective_value) and grad.dtype == np.float64
        return objective_value, grad

    def __copy__(self):
        raise NotImplementedError("non-copyable")

    def _init_layers(self):
        """
        Initializes C-layers and F-layers by corresponding gate matrices.
        """
        C_gates = self._C_gates
        C_layers = self._C_layers
        for l in range(self._L):
            C_layers[l].set_from_matrix(g4x4=C_gates[l])

        F_gates = self._F_gates
        F_layers = self._F_layers
        for l in range(self.num_qubits):
            F_layers[l].set_from_matrix(g2x2=F_gates[l])

    def _calc_ucf_fuc(self):
        """
        Computes matrices UCF and FUC. Both remain non-finalized.
        """
        UCF = self._UCF
        FUC = self._FUC
        tmpA = self._tmpA
        C_layers = self._C_layers
        F_layers = self._F_layers
        L, n = self._L, self.num_qubits

        # tmpA = U^dagger.
        np.conj(self._U.T, out=tmpA)

        # UCF = FUC = U^dagger @ C = U^dagger @ C_{L-1} @ ... @ C_{0}.
        self._UCF.set_matrix(tmpA)
        for l in range(L - 1, -1, -1):
            UCF.mul_right_q2(C_layers[l], temp_mat=tmpA, dagger=False)
        FUC.set_matrix(UCF.finalize(temp_mat=tmpA))

        # FUC = F @ U^dagger @ C = F_{n-1} @ ... @ F_{0} @ U^dagger @ C.
        for l in range(n):
            FUC.mul_left_q1(F_layers[l], temp_mat=tmpA)

        # UCF = U^dagger @ C @ F = U^dagger @ C @ F_{n-1} @ ... @ F_{0}.
        for l in range(n - 1, -1, -1):
            UCF.mul_right_q1(F_layers[l], temp_mat=tmpA, dagger=False)

    def _calc_objective_function(self) -> float:
        """
        Computes the value of objective function.
        """
        ucf = self._UCF.finalize(temp_mat=self._tmpA)
        trace_ucf = np.trace(ucf)
        fobj = abs((2 ** self.num_qubits) - float(np.real(trace_ucf)))

        # No need to finalize both matrices UCF and FUC, just for debugging.
        if self._debug:
            fuc = self._FUC.finalize(temp_mat=self._tmpA)
            trace_fuc = np.trace(fuc)
            print(
                "trace relative residual: {:0.16f},  "
                "trace(UCF): {:f},  trace(FUC): {:f}".format(
                    abs(trace_ucf - trace_fuc) / max(abs(trace_ucf), abs(trace_fuc)),
                    trace_ucf,
                    trace_fuc,
                )
            )
            _EPS = float(np.sqrt(np.finfo(np.float64).eps))
            assert abs(trace_ucf - trace_fuc) <= _EPS + _EPS * abs(trace_fuc)

        return fobj

    def _calc_gradient4L(self, grad4L: np.ndarray):
        T = self._FUC
        tmpA, tmpB = self._tmpA, self._tmpB
        C_gates = self._C_gates
        C_dervs = self._C_dervs
        C_layers = self._C_layers
        for l in range(self._L):
            # T[l] <-- C[l-1] @ T[l-1] @ C[l].conj.T. Note, C_layers[l] has
            # been initialized in _init_layers(), however, C_layers[l-1] was
            # reused on the previous step, see below, so we need to restore it.
            if l > 0:
                C_layers[l - 1].set_from_matrix(g4x4=C_gates[l - 1])
                T.mul_left_q2(C_layers[l - 1], temp_mat=tmpA)
            T.mul_right_q2(C_layers[l], temp_mat=tmpA, dagger=True)
            T.finalize(temp_mat=tmpA)
            # Compute gradient components. We reuse C_layers[l] several times.
            for i in range(4):
                C_layers[l].set_from_matrix(g4x4=C_dervs[l, i])
                grad4L[l, i] = (-1) * np.real(T.product_q2(L=C_layers[l], tmpA=tmpA, tmpB=tmpB))

    def _calc_gradient3n(self, grad3n: np.ndarray):
        W = self._UCF
        tmpA, tmpB = self._tmpA, self._tmpB
        F_gates = self._F_gates
        F_dervs = self._F_dervs
        F_layers = self._F_layers
        for l in range(self.num_qubits):
            # W[l] <-- F[l-1] @ W[l-1] @ F[l].conj.T. Note, F_layers[l] has
            # been initialized in _init_layers(), however, F_layers[l-1] was
            # reused on the previous step, see below, so we need to restore it.
            if l > 0:
                F_layers[l - 1].set_from_matrix(g2x2=F_gates[l - 1])
                W.mul_left_q1(F_layers[l - 1], temp_mat=tmpA)
            W.mul_right_q1(F_layers[l], temp_mat=tmpA, dagger=True)
            W.finalize(temp_mat=tmpA)
            # Compute gradient components. We reuse F_layers[l] several times.
            for i in range(3):
                F_layers[l].set_from_matrix(g2x2=F_dervs[l, i])
                grad3n[l, i] = (-1) * np.real(W.product_q1(L=F_layers[l], tmpA=tmpA, tmpB=tmpB))
