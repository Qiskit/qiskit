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

from qiskit.quantum_info import OneQubitEulerDecomposer, Pauli

from .scipy_optimizer import SciPyOptimizer


class _Paulis:
    DECOMPOSER = OneQubitEulerDecomposer()
    I = Pauli("I").to_matrix()
    X = Pauli("X").to_matrix()
    Y = Pauli("Y").to_matrix()
    Z = Pauli("Z").to_matrix()
    IX = (I + 1j * X) / np.sqrt(2)
    IY = (I + 1j * Y) / np.sqrt(2)
    IZ = (I + 1j * Z) / np.sqrt(2)
    XY = (X + Y) / np.sqrt(2)
    YZ = (Y + Z) / np.sqrt(2)
    ZX = (Z + X) / np.sqrt(2)

    @classmethod
    def _vec2angles(cls, vec: np.ndarray) -> Tuple[float, float, float]:
        r_d = cls.I * vec[0] + 1j * (cls.X * vec[1] + cls.Y * vec[2] + cls.Z * vec[3])
        return cls.DECOMPOSER.angles(r_d)


class SequentialOptimizer(SciPyOptimizer):
    """
    Base class for sequential optimizer.

    This family of optimizers optimizes parameterized U gates one by one in the order they appear
    in the ansatz. Once the last parameterized U gate is optimized, it returns to the first one.
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

    def _default_maxiter(self, x_0: np.ndarray) -> int:
        return x_0.size * 2

    def _initialize(self, x_0: np.ndarray) -> np.ndarray:
        return x_0

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
                If None, it is interpreted as N*2, where N is the number of parameters in the input
                circuit.
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
            x0[idx : idx + 3] = _Paulis._vec2angles(eigvecs[:, 0])

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
