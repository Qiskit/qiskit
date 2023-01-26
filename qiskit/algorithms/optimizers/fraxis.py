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

from typing import Optional

import numpy as np
from scipy.optimize import OptimizeResult

from qiskit.quantum_info import OneQubitEulerDecomposer, Pauli

from .scipy_optimizer import SciPyOptimizer


class FraxisOptimizer(SciPyOptimizer):
    """
    Free-Axis Selection (Fraxis) algorithm.

    Reference
      [1] "Optimizing Parameterized Quantum Circuits with Free-Axis Selection,"
          HC. Watanabe, R. Raymond, Y. Ohnishi, E. Kaminishi, M. Sugawara
          https://arxiv.org/abs/2104.14875
    """

    _OPTIONS = ["maxiter", "disp"]

    # pylint: disable=unused-argument
    def __init__(
        self,
        maxiter: Optional[int] = None,
        disp: bool = False,
        options: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            maxiter: Maximum number of iterations to perform.
            disp: Set to True to print convergence messages.
            options: A dictionary of solver options.
            kwargs: additional kwargs for scipy.optimize.minimize.
        """
        if options is None:
            options = {}
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                options[k] = v
        super().__init__(method=fraxis, options=options, **kwargs)


X_mat = Pauli("X").to_matrix()
Y_mat = Pauli("Y").to_matrix()
Z_mat = Pauli("Z").to_matrix()
XY_mat = (X_mat + Y_mat) / np.sqrt(2)
YZ_mat = (Y_mat + Z_mat) / np.sqrt(2)
ZX_mat = (Z_mat + X_mat) / np.sqrt(2)

MATRICES = [X_mat, Y_mat, Z_mat, XY_mat, YZ_mat, ZX_mat]
DECOMPOSER = OneQubitEulerDecomposer()
ANGLES = [DECOMPOSER.angles(mat) for mat in MATRICES]


# pylint: disable=invalid-name
def fraxis(fun, x0, args=(), maxiter=None, callback=None, **_):
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
        maxiter (int):
            Maximum number of iterations to perform.
            Default: None.
        eps (float): eps
        **_ : additional options
        callback (callable, optional):
            Called after each iteration.
    Returns:
        OptimizeResult:
            The optimization result represented as a ``OptimizeResult`` object.
            Important attributes are: ``x`` the solution array. See
            `OptimizeResult` for a description of other attributes.
    Notes:
        In this optimization method, the optimization function have to satisfy
        three conditions written in [1].
    """

    x0 = np.asarray(x0)
    size = x0.shape[0]
    assert size % 3 == 0
    niter = 0
    funcalls = 0
    f_last = None

    while True:
        idx = (niter * 3) % size

        vals = []
        for angles in ANGLES:
            p = np.copy(x0)
            p[idx : idx + 3] = angles
            vals.append(fun(p, *args))
            funcalls += 1

        r_x = vals[0]
        r_y = vals[1]
        r_z = vals[2]
        r_xy = vals[3]
        r_yz = vals[4]
        r_zx = vals[5]

        mat = np.array(
            [
                [r_x, 2 * r_xy - r_x - r_y, 2 * r_zx - r_x - r_z],
                [0, r_y, 2 * r_yz - r_y - r_z],
                [0, 0, r_z],
            ]
        )
        mat += mat.T
        _, eigvecs = np.linalg.eigh(mat)

        vals = []
        for vec in eigvecs.T:
            r_d = X_mat * vec[0] + Y_mat * vec[1] + Z_mat * vec[2]
            angles = DECOMPOSER.angles(r_d)
            p = np.copy(x0)
            p[idx : idx + 3] = angles
            vals.append([fun(p, *args), *angles])
            funcalls += 1

        f_min, *angles = min(vals)
        if f_last is not None:
            print(f_last, f_min, f_last > f_min)
        if f_last is None or f_last > f_min:
            x0[idx : idx + 3] = angles
            f_last = f_min

        niter += 1

        if callback is not None:
            callback(np.copy(x0))

        if maxiter is not None:
            if niter >= maxiter:
                break

    return OptimizeResult(
        fun=fun(np.copy(x0), *args), x=x0, nit=niter, nfev=funcalls, success=(niter > 1)
    )
