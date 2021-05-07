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

"""Conjugate Gradient optimizer."""

from typing import Optional

from .optimizer import Optimizer
from .scipy_minimizer import ScipyMinimizer


class CG(ScipyMinimizer):
    """Conjugate Gradient optimizer.

    CG is an algorithm for the numerical solution of systems of linear equations whose matrices are
    symmetric and positive-definite. It is an *iterative algorithm* in that it uses an initial
    guess to generate a sequence of improving approximate solutions for a problem,
    in which each approximation is derived from the previous ones.  It is often used to solve
    unconstrained optimization problems, such as energy minimization.

    Uses scipy.optimize.minimize CG.
    For further detail, please refer to
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    _OPTIONS = ["maxiter", "disp", "gtol", "eps"]

    # pylint: disable=unused-argument
    def __init__(
        self,
        maxiter: int = 20,
        disp: bool = False,
        gtol: float = 1e-5,
        tol: Optional[float] = None,
        eps: float = 1.4901161193847656e-08,
        **kwargs,
    ) -> None:
        """
        Args:
            maxiter: Maximum number of iterations to perform.
            disp: Set to True to print convergence messages.
            gtol: Gradient norm must be less than gtol before successful termination.
            tol: Tolerance for termination.
            eps: If jac is approximated, use this value for the step size.
        """
        if "options" in kwargs:
            options = kwargs.pop("options")
        else:
            options = {}
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                options[k] = v
        super().__init__(method="CG", options=options, tol=tol, **kwargs)

    def optimize(
        self,
        num_vars,
        objective_function,
        gradient_function=None,
        variable_bounds=None,
        initial_point=None,
    ):
        if gradient_function is None and self._max_evals_grouped > 1:
            epsilon = self._options["eps"]
            gradient_function = Optimizer.wrap_function(
                Optimizer.gradient_num_diff, (objective_function, epsilon, self._max_evals_grouped)
            )

        return super().optimize(
            num_vars,
            objective_function,
            gradient_function=gradient_function,
            variable_bounds=variable_bounds,
            initial_point=initial_point,
        )
