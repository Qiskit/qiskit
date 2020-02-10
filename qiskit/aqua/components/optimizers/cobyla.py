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

"""Constrained Optimization By Linear Approximation optimizer."""

from typing import Optional
import logging

from scipy.optimize import minimize
from .optimizer import Optimizer

logger = logging.getLogger(__name__)


class COBYLA(Optimizer):
    """
    Constrained Optimization By Linear Approximation optimizer.

    COBYLA is a numerical optimization method for constrained problems
    where the derivative of the objective function is not known.

    Uses scipy.optimize.minimize COBYLA.
    For further detail, please refer to
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    _OPTIONS = ['maxiter', 'disp', 'rhobeg']

    # pylint: disable=unused-argument
    def __init__(self,
                 maxiter: int = 1000,
                 disp: bool = False,
                 rhobeg: float = 1.0,
                 tol: Optional[float] = None) -> None:
        """
        Args:
            maxiter: Maximum number of function evaluations.
            disp: Set to True to print convergence messages.
            rhobeg: Reasonable initial changes to the variables.
            tol: Final accuracy in the optimization (not precisely guaranteed).
                 This is a lower bound on the size of the trust region.
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
                       method="COBYLA", options=self._options)
        return res.x, res.fun, res.nfev
