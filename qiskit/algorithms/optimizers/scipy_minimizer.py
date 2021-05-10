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

"""Wrapper class of scipy.optimize.minimize."""

from collections.abc import Sequence
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
from scipy.optimize import Bounds, minimize

from .optimizer import Optimizer, OptimizerSupportLevel


class ScipyMinimizer(Optimizer):
    """
    Optimizer using scipy.optimize.minimize.

    For further detail, please refer to
    See https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html
    """

    def __init__(
        self,
        method: Union[str, Callable],
        tol: Optional[float] = None,
        options: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Args:
            method: Type of solver.
            tol: Tolerance for termination.
            options: A dictionary of solver options.
            kwargs: additional kwargs for scipy.optimize.minimize.
        """
        # pylint: disable=super-init-not-called
        self._method = method.lower() if isinstance(method, str) else method
        self._set_support_level()
        self._options = options if options is not None else {}
        self._tol = tol
        self._kwargs = kwargs
        self._max_evals_grouped = 1

    def get_support_level(self):
        """Return support level dictionary"""
        return {
            "gradient": self._gradient_support_level,
            "bounds": self._bounds_support_level,
            "initial_point": self._initial_point_support_level,
        }

    def _set_support_level(self):
        if self._method in {"l-bfgs-b", "tnc", "slsqp", "powell", "trust-constr"}:
            self._bounds_support_level = OptimizerSupportLevel.supported
        else:
            self._bounds_support_level = OptimizerSupportLevel.ignored

        if self._method in {
            "cg",
            "bfgs",
            "newton-cg",
            "l-bfgs-b",
            "tnc",
            "slsqp",
            "dogleg",
            "trust-ncg",
            "trust-krylov",
            "trust-exact",
            "trust-constr",
        }:
            self._gradient_support_level = OptimizerSupportLevel.supported
        else:
            self._gradient_support_level = OptimizerSupportLevel.ignored

        self._initial_point_support_level = OptimizerSupportLevel.required

    def optimize(
        self,
        num_vars,
        objective_function,
        gradient_function=None,
        variable_bounds: Optional[Union[Sequence, Bounds]] = None,
        initial_point=None,
    ):
        if self.is_bounds_ignored:
            variable_bounds = None
        if self.is_gradient_ignored:
            gradient_function = None
        if self.is_gradient_supported and gradient_function is None and self._max_evals_grouped > 1:
            if "eps" in self._options:
                epsilon = self._options["eps"]
            else:
                epsilon = (
                    1e-8 if self._method in {"l_bfgs_b", "tnc"} else np.sqrt(np.finfo(float).eps)
                )
            gradient_function = Optimizer.wrap_function(
                Optimizer.gradient_num_diff, (objective_function, epsilon, self._max_evals_grouped)
            )

        super().optimize(
            num_vars,
            objective_function,
            gradient_function=gradient_function,
            variable_bounds=variable_bounds,
            initial_point=initial_point,
        )  # Validate the input
        res = minimize(
            fun=objective_function,
            x0=initial_point,
            method=self._method,
            jac=gradient_function,
            bounds=variable_bounds,
            tol=self._tol,
            options=self._options,
            **self._kwargs,
        )
        return res.x, res.fun, res.nfev
