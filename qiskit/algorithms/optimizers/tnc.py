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

"""Truncated Newton (TNC) optimizer. """

from typing import Optional

from .optimizer import Optimizer
from .scipy_minimizer import ScipyMinimizer


class TNC(ScipyMinimizer):
    """
    Truncated Newton (TNC) optimizer.

    TNC uses a truncated Newton algorithm to minimize a function with variables subject to bounds.
    This algorithm uses gradient information; it is also called Newton Conjugate-Gradient.
    It differs from the :class:`CG` method as it wraps a C implementation and allows each variable
    to be given upper and lower bounds.

    Uses scipy.optimize.minimize TNC
    For further detail, please refer to
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    _OPTIONS = ["maxiter", "disp", "accuracy", "ftol", "xtol", "gtol", "eps"]

    # pylint: disable=unused-argument
    def __init__(
        self,
        maxiter: int = 100,
        disp: bool = False,
        accuracy: float = 0,
        ftol: float = -1,
        xtol: float = -1,
        gtol: float = -1,
        tol: Optional[float] = None,
        eps: float = 1e-08,
        **kwargs,
    ) -> None:
        """
        Args:
            maxiter: Maximum number of function evaluation.
            disp: Set to True to print convergence messages.
            accuracy: Relative precision for finite difference calculations.
                If <= machine_precision, set to sqrt(machine_precision). Defaults to 0.
            ftol: Precision goal for the value of f in the stopping criterion.
                If ftol < 0.0, ftol is set to 0.0 defaults to -1.
            xtol: Precision goal for the value of x in the stopping criterion
                (after applying x scaling factors).
                If xtol < 0.0, xtol is set to sqrt(machine_precision). Defaults to -1.
            gtol: Precision goal for the value of the projected gradient in
                the stopping criterion (after applying x scaling factors).
                If gtol < 0.0, gtol is set to 1e-2 * sqrt(accuracy).
                Setting it to 0.0 is not recommended. Defaults to -1.
            tol: Tolerance for termination.
            eps: Step size used for numerical approximation of the Jacobian.
        """
        if "options" in kwargs:
            options = kwargs.pop("options")
        else:
            options = {}
        for k, v in list(locals().items()):
            if k in self._OPTIONS:
                options[k] = v
        super().__init__("TNC", options=options, tol=tol, **kwargs)

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
