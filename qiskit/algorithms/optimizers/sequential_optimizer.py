# This code is part of Qiskit.
#
# (C) Copyright IBM 2023.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Sequential optimizer."""

from abc import abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from scipy.optimize import OptimizeResult

from qiskit.circuit.library import UGate
from qiskit.quantum_info import OneQubitEulerDecomposer, Pauli

from .scipy_optimizer import SciPyOptimizer

# Constants
_DECOMPOSER = OneQubitEulerDecomposer()
_I = Pauli("I").to_matrix()
_X = Pauli("X").to_matrix()
_Y = Pauli("Y").to_matrix()
_Z = Pauli("Z").to_matrix()
_IX = (_I + 1j * _X) / np.sqrt(2)
_IY = (_I + 1j * _Y) / np.sqrt(2)
_IZ = (_I + 1j * _Z) / np.sqrt(2)
_XY = (_X + _Y) / np.sqrt(2)
_YZ = (_Y + _Z) / np.sqrt(2)
_ZX = (_Z + _X) / np.sqrt(2)


class SequentialOptimizer(SciPyOptimizer):
    """
    Base class for sequential optimizer.

    This family of optimizers optimizes parameterized U gates one by one in the order they are included in the ansatz.
    Once the last parameterized U gate is optimized, it returns to the first one.
    It may evaluate energy function multiple times to optimize a parameterized U gate.

    .. note::

        This optimizer only works with U gates as parameterized gates.
    """

    _OPTIONS = ["maxiter", "xtol"]

    # pylint: disable=unused-argument
    def __init__(
        self,
        maxiter: Optional[int] = None,
        xtol: Optional[float] = None,
        options: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            maxiter: Maximum number of iterations to perform. Will default to None.
                If None, it is interpreted as N*2, where N is the number of parameters
                in the input circuit.
            xtol: If the norm of the parameter update is smaller than this threshold,
                the optimizer is considered to have converged.
                This check is invoked at every first parameterized U gate.
                Formally, the convergence is determined if ``|x0 - x0_prev| < xtol * |x0_prev|``,
                where ``x0_prev`` is ``x0`` value at the first parameterized U gate in the last loop.
                Will default to None. If None, no convergence check is invoked.
            options: A dictionary of solver options.
            kwargs: additional kwargs for scipy.optimize.minimize.
        """
        if options is None:
            options = {}
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                options[k] = v
        super().__init__(method=self._minimize, options=options, **kwargs)

    def _default_maxiter(self, x0: np.ndarray) -> int:
        return x0.size * 2

    def _initialize(self, x0: np.ndarray) -> np.ndarray:
        return x0

    def _vec2angles(self, vec: np.ndarray) -> Tuple[float, float, float]:
        r_d = _I * vec[0] + 1j * (_X * vec[1] + _Y * vec[2] + _Z * vec[3])
        return _DECOMPOSER.angles(r_d)

    @property
    @abstractmethod
    def _angles(self) -> List[Tuple[float, float, float]]:
        raise NotImplementedError("_angles method is not implemented")

    @abstractmethod
    def _cost_matrix(self, vals: List[float]) -> np.ndarray:
        raise NotImplementedError("_cost_matrix method is not implemented")

    # pylint: disable=invalid-name
    def _minimize(self, fun, x0, args=(), maxiter=None, xtol=None, callback=None, **_):
        """
        Find the global minimum of a function.

        Args:
            fun (callable): ``f(x, *args)``
                Function to be optimized.  ``args`` can be passed as an optional item
                in the dict ``minimizer_kwargs``.
                This function must satisfy the three condition written in Ref. [1].
            x0 (ndarray): shape (n,)
                Initial guess. Array of real elements of size (n,),
                where 'n' is the number of independent variables.
            args (tuple, optional):
                Extra arguments passed to the objective function.
            maxiter (int, optional):
                Maximum number of iterations to perform. Will default to None.
                If None, it is interpreted as N*2, where N is the number of parameters in the input circuit.
            xtol (float, optional):
                If the norm of the parameter update is smaller than this threshold,
                the optimizer is considered to have converged.
                This check is invoked at every first parameterized U gate.
                Formally, the convergence is determined if ``|x0 - x0_prev| < xtol * |x0_prev|``,
                where ``x0_prev`` is ``x0`` value at the first parameterized U gate in the last loop.
                Will default to None. If None, no convergence check is invoked.
            **_ : additional options
            callback (callable, optional):
                Called after each iteration.
        Returns:
            OptimizeResult:
                The optimization result represented as a ``OptimizeResult`` object.
                Important attributes are: ``x`` the solution array. See
                `OptimizeResult` for a description of other attributes.
        Raises:
            ValueError: if the size of ``x0`` is not multiple of 3.
        """
        x0 = np.array(x0)  # copy x0 not to modify the original x0
        if x0.size % 3 != 0:
            raise ValueError(
                f"The size of x0 should be multiple of 3. Actual size: {x0.size}. "
                "Note that SequentialOptimizer works with only U gates as parameterized gates."
            )
        if maxiter is None:
            maxiter = self._default_maxiter(x0)

        x0 = self._initialize(x0)
        niter = 0
        funcalls = 0
        x0_prev = x0.copy()

        while True:
            idx = (niter * 3) % x0.size

            if xtol is not None:
                # check convergence at every first parameterized U gate
                if niter > 0 and idx == 0:
                    norm_x = np.linalg.norm(x0_prev)
                    norm_dx = np.linalg.norm(x0 - x0_prev)
                    if norm_dx < xtol * norm_x:
                        break
                    x0_prev = x0.copy()

            xs = []
            for angles in self._angles:
                p = x0.copy()
                p[idx : idx + 3] = angles
                xs.append(p)

            vals = fun(xs, *args)
            funcalls += len(xs)
            mat = self._cost_matrix(vals)
            eigvals, eigvecs = np.linalg.eigh(mat)

            # use the eigenvector `eigvecs[:, 0]` with the minimum eigenvalue
            x0[idx : idx + 3] = self._vec2angles(eigvecs[:, 0])

            niter += 1

            if callback is not None:
                # pass x0 values and the estimated energy value fun(x0) to the callback
                state = OptimizeResult(
                    fun=eigvals[0], x=x0, nit=niter, nfev=funcalls, success=(niter > 1)
                )
                terminate = callback(x0, state)
                if terminate:
                    break

            if niter >= maxiter:
                break

        return OptimizeResult(
            fun=fun(x0, *args), x=x0, nit=niter, nfev=funcalls, success=(niter > 1)
        )


class FQS(SequentialOptimizer):
    """
    Free Quaternion Selection (FQS) algorithm [1].

    This optimizer optimizes parameterized U gates one by one in the order they are included in the ansatz.
    Once the last parameterized U gate is optimized, it returns to the first one.

    .. note::

        This optimizer only works with U gates as parameterized gates.

    .. note::

        This optimizer evaluates the energy function 10 times for each U gate.

    References:
      [1] "Sequential optimal selection of a single-qubit gate and its relation to barren plateau in parameterized
          quantum circuits,"
          K. Wada, R. Raymond, Y. Sato, H.C. Watanabe,
          `arXiv:2209.08535 <https://arxiv.org/abs/2209.08535>`__
    """

    _MATRICES = [_I, _X, _Y, _Z, _IX, _IY, _IZ, _XY, _YZ, _ZX]
    _ANGLES = [_DECOMPOSER.angles(mat) for mat in _MATRICES]

    @property
    def _angles(self) -> List[Tuple[float, float, float]]:
        return self._ANGLES

    def _cost_matrix(self, vals: list[float]) -> np.ndarray:
        r_id, r_x, r_y, r_z, r_ix, r_iy, r_iz, r_xy, r_yz, r_zx = vals
        mat = np.array(
            [
                [
                    r_id / 2,
                    r_ix - r_x / 2 - r_id / 2,
                    r_iy - r_y / 2 - r_id / 2,
                    r_iz - r_z / 2 - r_id / 2,
                ],
                [0, r_x / 2, r_xy - r_x / 2 - r_y / 2, r_zx - r_x / 2 - r_z / 2],
                [0, 0, r_y / 2, r_yz - r_y / 2 - r_z / 2],
                [0, 0, 0, r_z / 2],
            ]
        )
        return mat + mat.T


class Fraxis(SequentialOptimizer):
    """
    Free-Axis Selection (Fraxis) algorithm [1].

    More precisely, this class implements π-Fraxis algorithm in Algorithm 1 of [1].
    This optimizer optimizes parameterized U gates one by one in the order they are included in the ansatz.
    Once the last parameterized U gate is optimized, it returns to the first one.

    .. note::

        This optimizer only works with U gates as parameterized gates.

    .. note::

        This optimizer evaluates the energy function 6 times for each U gate.

    References:
      [1] "Optimizing Parameterized Quantum Circuits with Free-Axis Selection,"
          H.C. Watanabe, R. Raymond, Y. Ohnishi, E. Kaminishi, M. Sugawara
          `arXiv:2104.14875 <https://arxiv.org/abs/2104.14875>`__
    """

    _MATRICES = [_X, _Y, _Z, _XY, _YZ, _ZX]
    _ANGLES = [_DECOMPOSER.angles(mat) for mat in _MATRICES]

    def _initialize(self, x0: np.ndarray):
        for idx in range(0, x0.size, 3):
            # Fraxis cannot handle some U3 rotations such as identity(=U3(0,0,0)).
            # The following converts such rotations into ones that Fraxis can handle.
            mat = UGate(*x0[idx : idx + 3]).to_matrix()
            n_x = mat[1, 0].real
            n_y = mat[1, 0].imag
            n_z = mat[0, 0]
            vec = np.array([0, n_x, n_y, n_z])
            if np.allclose(vec, 0):
                vec[0] = 1
            vec /= np.linalg.norm(vec)
            x0[idx : idx + 3] = self._vec2angles(vec)
        return x0

    @property
    def _angles(self) -> List[Tuple[float, float, float]]:
        return self._ANGLES

    def _cost_matrix(self, vals: list[float]) -> np.ndarray:
        r_x, r_y, r_z, r_xy, r_yz, r_zx = vals
        mat = np.array(
            [
                [0, 0, 0, 0],
                [0, r_x / 2, r_xy - r_x / 2 - r_y / 2, r_zx - r_x / 2 - r_z / 2],
                [0, 0, r_y / 2, r_yz - r_y / 2 - r_z / 2],
                [0, 0, 0, r_z / 2],
            ]
        )
        return mat + mat.T
