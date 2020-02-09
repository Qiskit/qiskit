# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2018, 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Powell optimizer."""

from typing import Optional
import logging

from scipy.optimize import minimize
from .optimizer import Optimizer

logger = logging.getLogger(__name__)


class POWELL(Optimizer):
    """
    Powell optimizer.

    The Powell algorithm performs unconstrained optimization; it ignores bounds or
    constraints. Powell is a *conjugate direction method*: it performs sequential one-dimensional
    minimization along each directional vector, which is updated at
    each iteration of the main minimization loop. The function being minimized need not be
    differentiable, and no derivatives are taken.

    Uses scipy.optimize.minimize Powell.
    For further detail, please refer to
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    _OPTIONS = ['maxiter', 'maxfev', 'disp', 'xtol']

    # pylint: disable=unused-argument
    def __init__(self,
                 maxiter: Optional[int] = None,
                 maxfev: int = 1000,
                 disp: bool = False,
                 xtol: float = 0.0001,
                 tol: Optional[float] = None) -> None:
        """
        Args:
            maxiter: Maximum allowed number of iterations. If both maxiter and maxfev
                are set, minimization will stop at the first reached.
            maxfev: Maximum allowed number of function evaluations. If both maxiter and
                maxfev are set, minimization will stop at the first reached.
            disp: Set to True to print convergence messages.
            xtol: Relative error in solution xopt acceptable for convergence.
            tol: Tolerance for termination.
        """
        super().__init__()
        for k, v in locals().items():
            if k in self._OPTIONS:
                self._options[k] = v
        self._tol = tol

    def get_support_level(self):
        """ Return support level dictionary """
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
                       method="Powell", options=self._options)
        return res.x, res.fun, res.nfev
