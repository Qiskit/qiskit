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

from .scipy_optimizer import SciPyOptimizer


class SLSQP(SciPyOptimizer):
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

    _OPTIONS = ["maxiter", "disp", "ftol", "eps"]

    # pylint: disable=unused-argument
    def __init__(
        self,
        maxiter: int = 100,
        disp: bool = False,
        ftol: float = 1e-06,
        tol: Optional[float] = None,
        eps: float = 1.4901161193847656e-08,
        options: Optional[dict] = None,
        max_evals_grouped: int = 1,
        **kwargs,
    ) -> None:
        """
        Args:
            maxiter: Maximum number of iterations.
            disp: Set to True to print convergence messages.
            ftol: Precision goal for the value of f in the stopping criterion.
            tol: Tolerance for termination.
            eps: Step size used for numerical approximation of the Jacobian.
            options: A dictionary of solver options.
            max_evals_grouped: Max number of default gradient evaluations performed simultaneously.
            kwargs: additional kwargs for scipy.optimize.minimize.
        """
        if options is None:
            options = {}
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                options[k] = v
        super().__init__(
            "SLSQP",
            options=options,
            tol=tol,
            max_evals_grouped=max_evals_grouped,
            **kwargs,
        )
