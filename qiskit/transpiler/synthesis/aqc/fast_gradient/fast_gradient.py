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
Implementation of the fast objective function class.
"""

import warnings
import numpy as np

# from .fast_grad_utils import get_max_num_bits
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
from ..approximate import ApproximatingObjective


class FastCNOTUnitObjective(ApproximatingObjective):
    """
    Implementation of objective function and gradient calculator, which is
    similar to
    :class:`~qiskit.transpiler.aqc.DefaultCNOTUnitObjective`
    but several times faster.
    """

    def __init__(self, num_qubits: int, cnots: np.ndarray):
        """Constructor."""
        super().__init__()

        self._verbose = 0
        self._debug = False
        if self._debug:
            print("Debugging mode is on")
        if self._verbose >= 1:
            print("Gradient:", self.__class__.__name__)

        # assert isinstance(num_qubits, int) and 2 <= num_qubits <= get_max_num_bits()
        # assert isinstance(cnots, np.ndarray) and cnots.ndim == 2
        # assert cnots.dtype == np.int64 and cnots.shape[0] == 2
        # pylint: disable=misplaced-comparison-constant
        # assert np.all(0 <= cnots) and np.all(cnots < num_qubits)

        self.num_qubits = num_qubits  # number of qubits
        dim = 2**num_qubits  # actual problem dimension
        depth = cnots.shape[1]
        self._depth = depth  # number of C-layers (circuit depth)

        self._u_mat = np.empty(0)  # reference to external target matrix U
        self._ucf_mat = PMatrix(mat=None)  # U^dagger @ C @ F
        self._fuc_mat = PMatrix(mat=None)  # F @ U^dagger @ C
        self._ucf_mat.initialize(num_qubits)
        self._fuc_mat.initialize(num_qubits)
        self._circ_thetas = np.empty(0)  # last thetas used in objective

        # array of C-layers:
        self._c_layers = np.array([object()] * depth, dtype=LayerBase)
        # array of F-layers:
        self._f_layers = np.array([object()] * num_qubits, dtype=LayerBase)
        # 4x4 C-gate matrices:
        self._c_gates = np.full((depth, 4, 4), fill_value=0, dtype=np.cfloat)
        # derivatives of 4x4 C-gate matrices:
        self._c_dervs = np.full((depth, 4, 4, 4), fill_value=0, dtype=np.cfloat)
        # 4x4 F-gate matrices:
        self._f_gates = np.full((num_qubits, 2, 2), fill_value=0, dtype=np.cfloat)
        # derivatives of 4x4 F-gate matrices:
        self._f_dervs = np.full((num_qubits, 3, 2, 2), fill_value=0, dtype=np.cfloat)
        # temporary NxN matrices:
        self._tmp1 = np.full((dim, dim), fill_value=0, dtype=np.cfloat)
        self._tmp2 = np.full((dim, dim), fill_value=0, dtype=np.cfloat)

        # Create layers of 2-qubit gates.
        for q in range(depth):
            j = int(cnots[0, q])
            k = int(cnots[1, q])
            self._c_layers[q] = Layer2Q(nbits=num_qubits, j=j, k=k)

        # Create layers of 1-qubit gates.
        for k in range(num_qubits):
            self._f_layers[k] = Layer1Q(nbits=num_qubits, k=k)

    @property
    def num_thetas(self) -> int:
        """
        Returns:
            the number of parameters in this optimization problem.
        """
        return 4 * self._depth + 3 * self.num_qubits

    def objective(self, param_values: np.ndarray) -> float:
        """
        Computes the objective function and some intermediate data for
        the subsequent gradient computation.
        See description of the base class method.
        """
        depth, n = self._depth, self.num_qubits
        # assert depth >= 2
        # assert isinstance(self.target_matrix, np.ndarray)
        # assert self.target_matrix.shape == (2**n, 2**n)
        # assert self.target_matrix.dtype == np.cfloat
        # assert isinstance(param_values, np.ndarray) and param_values.ndim == 1
        # assert param_values.size == self.num_thetas and param_values.dtype == np.float64
        self._u_mat = self.target_matrix  # reference to external target matrix U

        # Memorize the last angular parameters used to compute the objective.
        if self._circ_thetas.size == 0:
            self._circ_thetas = np.zeros((self.num_thetas,))
        np.copyto(self._circ_thetas, param_values)

        thetas4d = param_values[: 4 * depth].reshape(depth, 4)
        thetas3n = param_values[4 * depth :].reshape(n, 3)

        init_layer2q_matrices(thetas=thetas4d, dst=self._c_gates)
        init_layer2q_deriv_matrices(thetas=thetas4d, dst=self._c_dervs)
        init_layer1q_matrices(thetas=thetas3n, dst=self._f_gates)
        init_layer1q_deriv_matrices(thetas=thetas3n, dst=self._f_dervs)

        self._init_layers()
        self._calc_ucf_fuc()
        objective_value = self._calc_objective_function()
        # assert np.isreal(objective_value)
        return objective_value

    def gradient(self, param_values: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of objective function.
        See description of the base class method.
        """
        # assert isinstance(param_values, np.ndarray) and param_values.ndim == 1
        # assert param_values.size == self.num_thetas and param_values.dtype == np.float64

        # If thetas are the same as used for objective value calculation
        # before calling this function, then we re-use the computations,
        # otherwise we have to re-compute the objective.
        tol = float(np.sqrt(np.finfo(float).eps))
        if not np.allclose(param_values, self._circ_thetas, atol=tol, rtol=tol):
            self.objective(param_values)
            warnings.warn("gradient is computed before the objective")

        grad = np.full((param_values.size,), fill_value=0, dtype=np.float64)
        grad4d = grad[: 4 * self._depth].reshape(self._depth, 4)
        grad3n = grad[4 * self._depth :].reshape(self.num_qubits, 3)
        self._calc_gradient4d(grad4d)
        self._calc_gradient3n(grad3n)
        # assert grad.dtype == np.float64
        return grad

    def __copy__(self):
        """
        Prevents this object from being copied.
        """
        raise NotImplementedError("non-copyable")

    def _init_layers(self):
        """
        Initializes C-layers and F-layers by corresponding gate matrices.
        """
        c_gates = self._c_gates
        c_layers = self._c_layers
        for q in range(self._depth):
            c_layers[q].set_from_matrix(mat=c_gates[q])

        f_gates = self._f_gates
        f_layers = self._f_layers
        for q in range(self.num_qubits):
            f_layers[q].set_from_matrix(mat=f_gates[q])

    def _calc_ucf_fuc(self):
        """
        Computes matrices ucf_mat and fuc_mat. Both remain non-finalized.
        """
        ucf_mat = self._ucf_mat
        fuc_mat = self._fuc_mat
        tmp1 = self._tmp1
        c_layers = self._c_layers
        f_layers = self._f_layers
        depth, n = self._depth, self.num_qubits

        # tmp1 = U^dagger.
        np.conj(self._u_mat.T, out=tmp1)

        # ucf_mat = fuc_mat = U^dagger @ C = U^dagger @ C_{depth-1} @ ... @ C_{0}.
        self._ucf_mat.set_matrix(tmp1)
        for q in range(depth - 1, -1, -1):
            ucf_mat.mul_right_q2(c_layers[q], temp_mat=tmp1, dagger=False)
        fuc_mat.set_matrix(ucf_mat.finalize(temp_mat=tmp1))

        # fuc_mat = F @ U^dagger @ C = F_{n-1} @ ... @ F_{0} @ U^dagger @ C.
        for q in range(n):
            fuc_mat.mul_left_q1(f_layers[q], temp_mat=tmp1)

        # ucf_mat = U^dagger @ C @ F = U^dagger @ C @ F_{n-1} @ ... @ F_{0}.
        for q in range(n - 1, -1, -1):
            ucf_mat.mul_right_q1(f_layers[q], temp_mat=tmp1, dagger=False)

    def _calc_objective_function(self) -> float:
        """
        Computes the value of objective function.
        """
        ucf = self._ucf_mat.finalize(temp_mat=self._tmp1)
        trace_ucf = np.trace(ucf)
        fobj = abs((2**self.num_qubits) - float(np.real(trace_ucf)))

        # No need to finalize both matrices ucf_mat and fuc_mat, just for debugging.
        if self._debug:
            fuc = self._fuc_mat.finalize(temp_mat=self._tmp1)
            trace_fuc = np.trace(fuc)
            print(
                "trace relative residual: {:0.16f},  "
                "trace(ucf_mat): {:f},  trace(fuc_mat): {:f}".format(
                    abs(trace_ucf - trace_fuc) / max(abs(trace_ucf), abs(trace_fuc)),
                    trace_ucf,
                    trace_fuc,
                )
            )
            _eps = float(np.sqrt(np.finfo(np.float64).eps))
            # assert abs(trace_ucf - trace_fuc) <= _eps + _eps * abs(trace_fuc)

        return fobj

    def _calc_gradient4d(self, grad4d: np.ndarray):
        """
        Calculates a part gradient contributed by 2-qubit gates.
        """
        fuc = self._fuc_mat
        tmp1, tmp2 = self._tmp1, self._tmp2
        c_gates = self._c_gates
        c_dervs = self._c_dervs
        c_layers = self._c_layers
        for q in range(self._depth):
            # fuc[q] <-- C[q-1] @ fuc[q-1] @ C[q].conj.T. Note, c_layers[q] has
            # been initialized in _init_layers(), however, c_layers[q-1] was
            # reused on the previous step, see below, so we need to restore it.
            if q > 0:
                c_layers[q - 1].set_from_matrix(mat=c_gates[q - 1])
                fuc.mul_left_q2(c_layers[q - 1], temp_mat=tmp1)
            fuc.mul_right_q2(c_layers[q], temp_mat=tmp1, dagger=True)
            fuc.finalize(temp_mat=tmp1)
            # Compute gradient components. We reuse c_layers[q] several times.
            for i in range(4):
                c_layers[q].set_from_matrix(mat=c_dervs[q, i])
                grad4d[q, i] = (-1) * np.real(
                    fuc.product_q2(layer=c_layers[q], tmp1=tmp1, tmp2=tmp2)
                )

    def _calc_gradient3n(self, grad3n: np.ndarray):
        """
        Calculates a part gradient contributed by 1-qubit gates.
        """
        ucf = self._ucf_mat
        tmp1, tmp2 = self._tmp1, self._tmp2
        f_gates = self._f_gates
        f_dervs = self._f_dervs
        f_layers = self._f_layers
        for q in range(self.num_qubits):
            # ucf[q] <-- F[q-1] @ ucf[q-1] @ F[q].conj.T. Note, f_layers[q] has
            # been initialized in _init_layers(), however, f_layers[q-1] was
            # reused on the previous step, see below, so we need to restore it.
            if q > 0:
                f_layers[q - 1].set_from_matrix(mat=f_gates[q - 1])
                ucf.mul_left_q1(f_layers[q - 1], temp_mat=tmp1)
            ucf.mul_right_q1(f_layers[q], temp_mat=tmp1, dagger=True)
            ucf.finalize(temp_mat=tmp1)
            # Compute gradient components. We reuse f_layers[q] several times.
            for i in range(3):
                f_layers[q].set_from_matrix(mat=f_dervs[q, i])
                grad3n[q, i] = (-1) * np.real(
                    ucf.product_q1(layer=f_layers[q], tmp1=tmp1, tmp2=tmp2)
                )
