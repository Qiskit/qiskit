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

"""Free-Axis Selection (Fraxis) algorithm."""

from typing import Optional, Tuple

import numpy as np
from scipy.optimize import OptimizeResult

from qiskit.circuit.library import UGate
from qiskit.quantum_info import OneQubitEulerDecomposer, Pauli

from .scipy_optimizer import SciPyOptimizer


class FQS(SciPyOptimizer):
    """
    Free-Axis Selection (Fraxis) algorithm [1].

    More precisely, this class implements π-Fraxis algorithm in Algorithm 1 of [1].

    .. note::

        This optimizer only works with U gates as parameterized gates.

    References:
      [1] "Optimizing Parameterized Quantum Circuits with Free-Axis Selection,"
          HC. Watanabe, R. Raymond, Y. Ohnishi, E. Kaminishi, M. Sugawara
          `arXiv:2104.14875 <https://arxiv.org/abs/2104.14875>`__
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
        super().__init__(method=fqs, options=options, **kwargs)


I_mat = Pauli("I").to_matrix()
X_mat = Pauli("X").to_matrix()
Y_mat = Pauli("Y").to_matrix()
Z_mat = Pauli("Z").to_matrix()
iX_mat = Pauli("iX").to_matrix()
iY_mat = Pauli("iY").to_matrix()
iZ_mat = Pauli("iZ").to_matrix()
Xp_mat = (I_mat + iX_mat) / np.sqrt(2)
Xm_mat = (I_mat - iX_mat) / np.sqrt(2)
Yp_mat = (I_mat + iY_mat) / np.sqrt(2)
Ym_mat = (I_mat - iY_mat) / np.sqrt(2)
Zp_mat = (I_mat + iZ_mat) / np.sqrt(2)
Zm_mat = (I_mat - iZ_mat) / np.sqrt(2)
XY_mat = (iX_mat + iY_mat) / np.sqrt(2)
YZ_mat = (iY_mat + iZ_mat) / np.sqrt(2)
ZX_mat = (iZ_mat + iX_mat) / np.sqrt(2)

MATRICES = [I_mat, X_mat, Y_mat, Z_mat, Xp_mat, Yp_mat, Zp_mat, XY_mat, YZ_mat, ZX_mat]
DECOMPOSER = OneQubitEulerDecomposer()
ANGLES = [DECOMPOSER.angles(mat) for mat in MATRICES]


def _vec2angles(vec: np.ndarray) -> Tuple[float, float, float]:
    r_d = I_mat * vec[0] + iX_mat * vec[1] + iY_mat * vec[2] + iZ_mat * vec[3]
    return DECOMPOSER.angles(r_d)


# pylint: disable=invalid-name
def fqs(fun, x0, args=(), maxiter=None, xtol=None, callback=None, **_):
    """
    Find the global minimum of a function using Fraxis algorithm.

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
    x0 = np.asarray(x0)
    if x0.size % 3 != 0:
        raise ValueError(
            f"The size of x0 should be multiple of 3. Actual size: {x0.size}. "
            "Note that Fraxis works with only U gates as parameterized gates."
        )
    if maxiter is None:
        maxiter = x0.size * 2

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
        for angles in ANGLES:
            p = np.copy(x0)
            p[idx : idx + 3] = angles
            xs.append(p)

        r_id, r_x, r_y, r_z, r_xp, r_yp, r_zp, r_xy, r_yz, r_zx = fun(xs, *args)
        funcalls += len(xs)

        mat = np.array(
            [
                [
                    r_id / 2,
                    r_xp - r_x / 2 - r_id / 2,
                    r_yp - r_y / 2 - r_id / 2,
                    r_zp - r_z / 2 - r_id / 2,
                ],
                [0, r_x / 2, r_xy - r_x / 2 - r_y / 2, r_zx - r_x / 2 - r_z / 2],
                [0, 0, r_y / 2, r_yz - r_y / 2 - r_z / 2],
                [0, 0, 0, r_z / 2],
            ]
        )
        mat += mat.T
        eigvals, eigvecs = np.linalg.eigh(mat)

        # use the eigenvector `eigvecs[:, 0]` with the minimum eigenvalue
        x0[idx : idx + 3] = _vec2angles(eigvecs[:, 0])

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

    return OptimizeResult(fun=fun(x0, *args), x=x0, nit=niter, nfev=funcalls, success=(niter > 1))
