# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Nelder-Mead algorithm."""

from typing import Optional
import logging

from scipy.optimize import minimize
from .optimizer import Optimizer

logger = logging.getLogger(__name__)

# pylint: disable=invalid-name


class NELDER_MEAD(Optimizer):
    """Nelder-Mead algorithm.

    Uses scipy.optimize.minimize Nelder-Mead
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    _OPTIONS = ['maxiter', 'maxfev', 'disp', 'xatol', 'adaptive']

    # pylint: disable=unused-argument
    def __init__(self,
                 maxiter: Optional[int] = None,
                 maxfev: int = 1000,
                 disp: bool = False,
                 xatol: float = 0.0001,
                 tol: Optional[float] = None,
                 adaptive: bool = False) -> None:
        """
        Constructor.

        For details, please refer to
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html.

        Args:
            maxiter: Maximum allowed number of iterations. If both maxiter and maxfev are set,
                           minimization will stop at the first reached.
            maxfev: Maximum allowed number of function evaluations. If both maxiter and
                          maxfev are set, minimization will stop at the first reached.
            disp: Set to True to print convergence messages.
            xatol: Absolute error in xopt between iterations
                            that is acceptable for convergence.
            tol: Tolerance for termination.
            adaptive: Adapt algorithm parameters to dimensionality of problem.
        """
        super().__init__()
        for k, v in locals().items():
            if k in self._OPTIONS:
                self._options[k] = v
        self._tol = tol

    def get_support_level(self):
        """ return support level dictionary """
        return {
            'gradient': Optimizer.SupportLevel.ignored,
            'bounds': Optimizer.SupportLevel.ignored,
            'initial_point': Optimizer.SupportLevel.required
        }

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)

        res = minimize(objective_function, initial_point, tol=self._tol,
                       method="Nelder-Mead", options=self._options)
        return res.x, res.fun, res.nfev
