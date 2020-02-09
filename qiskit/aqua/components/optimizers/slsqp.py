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

"""Sequential Least SQuares Programming optimizer"""

from typing import Optional
import logging

from scipy.optimize import minimize
from .optimizer import Optimizer

logger = logging.getLogger(__name__)


class SLSQP(Optimizer):
    """
    Sequential Least SQuares Programming optimizer.

    SLSQP minimizes a function of several variables with any combination of bounds, equality
    and inequality constraints. The method wraps the SLSQP Optimization subroutine originally
    implemented by Dieter Kraft.

    SLSQP is ideal for mathematical problems for which the objective function and the constraints
    are twice continuously differentiable. Note that the wrapper handles infinite values in bounds
    by converting them into large floating values.

    Uses scipy.optimize.minimize SLSQP.
    For further detail, please refer to
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    _OPTIONS = ['maxiter', 'disp', 'ftol', 'eps']

    # pylint: disable=unused-argument
    def __init__(self,
                 maxiter: int = 100,
                 disp: bool = False,
                 ftol: float = 1e-06,
                 tol: Optional[float] = None,
                 eps: float = 1.4901161193847656e-08) -> None:
        """
        Args:
            maxiter: Maximum number of iterations.
            disp: Set to True to print convergence messages.
            ftol: Precision goal for the value of f in the stopping criterion.
            tol: Tolerance for termination.
            eps: Step size used for numerical approximation of the Jacobian.
        """
        super().__init__()
        for k, v in locals().items():
            if k in self._OPTIONS:
                self._options[k] = v
        self._tol = tol

    def get_support_level(self):
        """ Return support level dictionary """
        return {
            'gradient': Optimizer.SupportLevel.supported,
            'bounds': Optimizer.SupportLevel.supported,
            'initial_point': Optimizer.SupportLevel.required
        }

    def optimize(self, num_vars, objective_function, gradient_function=None,
                 variable_bounds=None, initial_point=None):
        super().optimize(num_vars, objective_function,
                         gradient_function, variable_bounds, initial_point)

        if gradient_function is None and self._max_evals_grouped > 1:
            epsilon = self._options['eps']
            gradient_function = Optimizer.wrap_function(Optimizer.gradient_num_diff,
                                                        (objective_function, epsilon,
                                                         self._max_evals_grouped))

        res = minimize(objective_function, initial_point, jac=gradient_function,
                       tol=self._tol, bounds=variable_bounds, method="SLSQP",
                       options=self._options)
        return res.x, res.fun, res.nfev
