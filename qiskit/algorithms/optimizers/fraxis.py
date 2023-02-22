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

from qiskit.quantum_info import OneQubitEulerDecomposer, Pauli
from qiskit.circuit.library import UGate

from .scipy_optimizer import SciPyOptimizer


class Fraxis(SciPyOptimizer):
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

    _OPTIONS = ["maxiter"]

    # pylint: disable=unused-argument
    def __init__(
        self,
        maxiter: Optional[int] = None,
        options: Optional[dict] = None,
        **kwargs,
    ) -> None:
        """
        Args:
            maxiter: Maximum number of iterations to perform. Will default to None.
                If None, it is interpreted as N*3, where N is the number of parameters
                in the input circuit.
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


def _vec2angles(vec: np.ndarray) -> Tuple[float, float, float]:
    r_d = X_mat * vec[0] + Y_mat * vec[1] + Z_mat * vec[2]
    return DECOMPOSER.angles(r_d)


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
            Maximum number of iterations to perform. Will default to None.
            If None, it is interpreted as N*3, where N is the number of parameters in the input circuit.
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
        maxiter = x0.size * 3

    niter = 0
    funcalls = 0

    for idx in range(0, x0.size, 3):
        # Fraxis cannot handle some U3 rotations such as identity(=U3(0,0,0)).
        # The following converts such rotations into ones that Fraxis can handle.
        mat = UGate(*x0[idx : idx + 3]).to_matrix()
        n_x = mat[1, 0].real
        n_y = mat[1, 0].imag
        n_z = mat[0, 0]
        vec = np.array([n_x, n_y, n_z])
        if np.allclose(vec, 0):
            vec[0] = 1
        vec /= np.linalg.norm(vec)
        x0[idx : idx + 3] = _vec2angles(vec)

    while True:
        idx = (niter * 3) % x0.size

        xs = []
        for angles in ANGLES:
            p = np.copy(x0)
            p[idx : idx + 3] = angles
            xs.append(p)

        r_x, r_y, r_z, r_xy, r_yz, r_zx = fun(xs, *args)
        funcalls += len(xs)

        mat = np.array(
            [
                [r_x, 2 * r_xy - r_x - r_y, 2 * r_zx - r_x - r_z],
                [0, r_y, 2 * r_yz - r_y - r_z],
                [0, 0, r_z],
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
                fun=eigvals[0] / 2, x=x0, nit=niter, nfev=funcalls, success=(niter > 1)
            )
            terminate = callback(x0, state)
            if terminate:
                break

        if niter >= maxiter:
            break

    return OptimizeResult(fun=fun(x0, *args), x=x0, nit=niter, nfev=funcalls, success=(niter > 1))
