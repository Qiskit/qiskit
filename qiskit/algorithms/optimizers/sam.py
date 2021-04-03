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

"""Sharpness-Aware Minimization (SAM) optimizer."""

import logging
from typing import Callable, Tuple, List, Dict, Optional
import numpy as np

from .optimizer import Optimizer, OptimizerSupportLevel
from ...utils import algorithm_globals

logger = logging.getLogger(__name__)


class SAM(Optimizer):
    """Sharpness-Aware Minimization for Efficiently Improving Generalization.
     An effective procedure for simultaneously minimizing loss value and loss sharpness.
     In particular, SAM seeks parameters that lie in neighborhoods having uniformly low loss;
     this formulation results in a minmax optimization problem on which gradient descent can be
     performed efficiently.

    * Pierre Foret, Ariel Kleiner, Hossein Mobahi, Behnam Neyshabur. (2020)
      Sharpness-Aware Minimization for Efficiently Improving Generalization
      https://arxiv.org/pdf/2010.01412v2.pdf

    """
    _OPTIONS = ['maxiter', 'eta', 'tol', 'param_tol', 'rho', 'second_order']

    def __init__(self,
                 maxiter: int = 10000,
                 eta: float = 0.1,
                 eps: float = 1e-10,
                 tol: float = 1e-6,
                 param_tol: float = 1e-6,
                 rho: float = 0.001,
                 second_order: bool = False) -> None:

        """
        Args:
            maxiter: Maximum number of iterations (full gradient steps)
            eta: The coefficient of the gradient update. Increasing this value
                results in larger step sizes: params_new = params - eta * grad
            eps : Value >=0, Epsilon to be used for finite differences if no analytic
                gradient method is given.
            tol: Tolerance for change in windowed average of objective values.
                Convergence occurs when either objective tolerance is met OR parameter
                tolerance is met.
            param_tol: Tolerance for change in norm of parameters.
            rho: Neighborhood size.
            second_order: If second order derivatives should be included in loss function gradient.
        """
        super().__init__()
        self._maxiter = maxiter
        self._eta = eta
        self._eps = eps
        self._tol = tol
        self._param_tol = param_tol
        self._rho = rho
        self._second_order = second_order

    def get_support_level(self) -> Dict[str, OptimizerSupportLevel]:
        """ Support level dictionary

        Returns:
            Dict[str, int]: gradient, bounds and initial point
                            support information that is ignored/required.
        """
        return {
            'gradient': OptimizerSupportLevel.ignored,
            'bounds': OptimizerSupportLevel.ignored,
            'initial_point': OptimizerSupportLevel.required
        }

    def minimize(self, objective_function: Callable[[np.ndarray], float], initial_point: np.ndarray,
                 gradient_function: Callable[[np.ndarray], float]) -> Tuple[np.ndarray, float, int]:
        """Run the minimization.

        Args:
            objective_function: A function handle to the objective function.
            initial_point: The initial iteration point.
            gradient_function: A function handle to the gradient of the objective function.

        Returns:
            A tuple of (optimal parameters, optimal value, number of iterations).
        """

        def rast_grad(x):
            return np.array([2 * x_i + 2 * np.pi * x_i * np.cos(2 * np.pi * x_i) for x_i in x])

        def bukin_grad(x):
            nom = np.sign(x[1] - 0.01 * x[0] * x[0]) * 100
            den = 2 * np.sqrt(abs(x[1] - 0.01 * x[0] * x[0]))
            y = nom / den
            x = -0.02 * x[0] * y - 0.01 * np.sign(x[0] + 10)
            return np.array([x, y])

        # exact gradient of the rosenbrock function
        def gradient(x):
            x = np.asarray(x)
            xm = x[1:-1]
            xm_m1 = x[:-2]
            xm_p1 = x[2:]
            der = np.zeros_like(x)
            der[1:-1] = (200 * (xm - xm_m1 ** 2) -
                         400 * (xm_p1 - xm ** 2) * xm - 2 * (1 - xm))
            der[0] = -400 * x[0] * (x[1] - x[0] ** 2) - 2 * (1 - x[0])
            der[-1] = 200 * (x[-1] - x[-2] ** 2)
            return der

        # exact hessian of the rosenbrock function
        def hess(x):
            x = np.atleast_1d(x)
            H = np.diag(-400 * x[:-1], 1) - np.diag(400 * x[:-1], -1)
            diagonal = np.zeros(len(x), dtype=x.dtype)
            diagonal[0] = 1200 * x[0] ** 2 - 400 * x[1] + 2
            diagonal[-1] = 200
            diagonal[1:-1] = 202 + 1200 * x[1:-1] ** 2 - 400 * x[2:]
            H = H + np.diag(diagonal)
            return H

        # Naive implementation
        params = params_new = initial_point
        for it in range(self._maxiter):

            ob = objective_function(params)

            grad = bukin_grad(params)  # dL_s(w)/dw
            norm_grad = np.linalg.norm(grad)  # ||dL_s(w)||
            eps = self._rho * grad / norm_grad  # eq. 2  rho * dL_s(w)/dw / ||dL_s(w)/dw||

            grad_sam = bukin_grad(params + eps)  # eq. 3  dL^SAM_s(w+eps)/d(w+eps)

            if self._second_order:  # include second order of Taylor expansion
                hes = hess(params)

                # gradient of numerator, denominator unchanged
                d_eps1 = self._rho * hes / norm_grad

                # grad of denominator, numerator unchanged
                d_eps2 = - self._rho * np.outer(hes.dot(grad), grad) / (norm_grad ** 3)
                d_eps = d_eps1 + d_eps2
                grad_sam = grad_sam + d_eps.dot(grad_sam)

            params_new = params - self._eta * grad_sam / np.linalg.norm(grad_sam)  # Algorithm 1

            if np.linalg.norm(params - params_new) < self._tol:
                return params_new, objective_function(params_new), it + 1
            params = params_new

        return params_new, objective_function(params_new), self._maxiter

    def optimize(self, num_vars: int, objective_function: Callable[[np.ndarray], float],
                 gradient_function: Optional[Callable[[np.ndarray], float]] = None,
                 variable_bounds: Optional[List[Tuple[float, float]]] = None,
                 initial_point: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, float, int]:
        """Perform optimization.

        Args:
            num_vars: Number of parameters to be optimized.
            objective_function: Handle to a function that computes the objective function.
            gradient_function: Handle to a function that computes the gradient of the objective
                function.
            variable_bounds: deprecated
            initial_point: The initial point for the optimization.

        Returns:
            A tuple (point, value, nfev) where\n
                point: is a 1D numpy.ndarray[float] containing the solution\n
                value: is a float with the objective function value\n
                nfev: is the number of objective function calls
        """
        super().optimize(num_vars, objective_function, gradient_function,
                         variable_bounds, initial_point)
        if initial_point is None:
            initial_point = algorithm_globals.random.random(num_vars)
        if gradient_function is None:
            gradient_function = Optimizer.wrap_function(Optimizer.gradient_num_diff,
                                                        (objective_function, self._eps,
                                                         self._max_evals_grouped))

        point, value, nfev = self.minimize(objective_function, initial_point, gradient_function)
        return point, value, nfev
